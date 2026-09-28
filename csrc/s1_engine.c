/* The model, its cache, and the two GPU calls. */
#include "s1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-backend.h"
#include "llama.h"

#define S1_N_BATCH  4096 /* most tokens accepted by one llama_decode call */
#define S1_N_UBATCH 1024 /* tokens per GPU pass */

/* Layer capture through libllama's eval callback. Armed only for the decode that holds the
 * decision positions and nothing else, so row j of a captured tensor is batch index j. */
struct capture {
    bool   armed;
    int    n;                          /* layers captured */
    int    layer[S1_LAYERS_MAX];       /* S1_LAYER_NORM or a block index */
    char   name[S1_LAYERS_MAX][32];    /* "result_embd_pooled" or "l_out-<layer>" */
    float *buf[S1_LAYERS_MAX];         /* max_rows x n_embd each */
    int    rows[S1_LAYERS_MAX];        /* rows copied by the last armed decode */
    int    n_embd;
    int    max_rows;
    bool   failed;
};

struct s1_engine {
    struct llama_model       *model;
    struct llama_context     *ctx;
    const struct llama_vocab *vocab;
    llama_memory_t            mem;
    struct llama_batch        batch;
    int32_t                  *out_index; /* n_seq entries: batch index of each decision position */
    int                       n_embd;
    int                       n_vocab;
    int                       n_seq;
    int                       n_ctx;
    int                       n_state; /* tokens held as sequence 0 */
    struct capture            cap;
};

static bool on_tensor(struct ggml_tensor *t, bool ask, void *user_data)
{
    struct capture *c = user_data;
    if (!c->armed) {
        return ask ? false : true;
    }
    int i = 0;
    while (i < c->n && strcmp(t->name, c->name[i]) != 0) {
        i++;
    }
    if (i == c->n) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    int rows = (int)t->ne[1];
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || rows > c->max_rows ||
        !ggml_is_contiguous(t)) {
        fprintf(stderr, "judgly: cannot capture %s (type %d, %lld x %lld)\n", t->name, (int)t->type,
                (long long)t->ne[0], (long long)t->ne[1]);
        c->failed = true;
        return true;
    }
    ggml_backend_tensor_get(t, c->buf[i], 0, (size_t)rows * (size_t)c->n_embd * sizeof(float));
    c->rows[i] = rows;
    return true;
}

int s1_engine_init(struct s1_engine **out, const struct s1_engine_params *p)
{
    if (p->n_seq < 2 || p->n_seq > 256) {
        fprintf(stderr, "judgly: n_seq is %d, must be 2 to 256\n", p->n_seq);
        return -1;
    }
    struct s1_engine *e = calloc(1, sizeof *e);
    if (!e) {
        fprintf(stderr, "judgly: out of memory\n");
        return -1;
    }
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p->n_gpu_layers;
    e->model = llama_model_load_from_file(p->model_path, mp);
    if (!e->model) {
        fprintf(stderr, "judgly: cannot load model %s\n", p->model_path);
        goto fail;
    }
    e->vocab   = llama_model_get_vocab(e->model);
    e->n_embd  = llama_model_n_embd(e->model);
    e->n_vocab = llama_vocab_n_tokens(e->vocab);
    e->n_seq   = p->n_seq;
    e->n_ctx   = p->n_ctx;

    if (p->n_layers < 0 || p->n_layers > S1_LAYERS_MAX) {
        fprintf(stderr, "judgly: %d layers to capture, at most %d\n", p->n_layers, S1_LAYERS_MAX);
        goto fail;
    }
    e->cap.n        = p->n_layers;
    e->cap.n_embd   = e->n_embd;
    e->cap.max_rows = p->n_seq;
    for (int i = 0; i < p->n_layers; i++) {
        int l = p->layers[i];
        if (l != S1_LAYER_NORM && (l < 0 || l >= llama_model_n_layer(e->model))) {
            fprintf(stderr, "judgly: layer %d does not exist (the model has %d)\n", l,
                    llama_model_n_layer(e->model));
            goto fail;
        }
        e->cap.layer[i] = l;
        if (l == S1_LAYER_NORM) {
            /* the final norm's output; libllama renames it from "result_norm" when embeddings are
               on without pooling, and it is what llama_get_embeddings_ith returns */
            snprintf(e->cap.name[i], sizeof e->cap.name[i], "result_embd_pooled");
        } else {
            snprintf(e->cap.name[i], sizeof e->cap.name[i], "l_out-%d", l);
        }
        e->cap.buf[i] = malloc((size_t)p->n_seq * (size_t)e->n_embd * sizeof(float));
        if (!e->cap.buf[i]) {
            fprintf(stderr, "judgly: out of memory\n");
            goto fail;
        }
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t)p->n_ctx;
    cp.n_batch      = S1_N_BATCH;
    cp.n_ubatch     = S1_N_UBATCH;
    cp.n_seq_max    = (uint32_t)p->n_seq;
    cp.kv_unified   = true; /* one cache for all sequences: branches share the state's entries */
    cp.swa_full     = !p->swa_pruned; /* keep the whole state in sliding-window layers */
    cp.embeddings   = false; /* switched on only around the decision positions: see read_last */
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (e->cap.n > 0) {
        cp.cb_eval           = on_tensor;
        cp.cb_eval_user_data = &e->cap;
    }
    e->ctx = llama_init_from_model(e->model, cp);
    if (!e->ctx) {
        fprintf(stderr, "judgly: cannot create context (n_ctx %d, n_seq %d)\n", p->n_ctx, p->n_seq);
        goto fail;
    }
    e->mem       = llama_get_memory(e->ctx);
    e->batch     = llama_batch_init(S1_N_BATCH, 0, 1);
    e->out_index = malloc((size_t)p->n_seq * sizeof *e->out_index);
    if (!e->out_index) {
        fprintf(stderr, "judgly: out of memory\n");
        goto fail;
    }
    fprintf(stderr, "judgly: model %s: n_embd %d, n_vocab %d, n_layer %d; n_ctx %d, n_seq %d, "
            "n_gpu_layers %d, %d layers captured\n", p->model_path, e->n_embd, e->n_vocab,
            llama_model_n_layer(e->model), p->n_ctx, p->n_seq, p->n_gpu_layers, e->cap.n);
    *out = e;
    return 0;

fail:
    s1_engine_free(e);
    return -1;
}

void s1_engine_free(struct s1_engine *e)
{
    if (!e) {
        return;
    }
    free(e->out_index);
    for (int i = 0; i < e->cap.n; i++) {
        free(e->cap.buf[i]);
    }
    if (e->batch.token) {
        llama_batch_free(e->batch);
    }
    if (e->ctx) {
        llama_free(e->ctx);
    }
    if (e->model) {
        llama_model_free(e->model);
    }
    llama_backend_free();
    free(e);
}

int s1_engine_n_embd(const struct s1_engine *e) { return e->n_embd; }

bool s1_engine_has_swa(const struct s1_engine *e) { return llama_model_n_swa(e->model) > 0; }
int s1_engine_n_vocab(const struct s1_engine *e) { return e->n_vocab; }
int s1_engine_n_seq(const struct s1_engine *e) { return e->n_seq; }
int s1_engine_n_ctx(const struct s1_engine *e) { return e->n_ctx; }
int s1_engine_h_stride(const struct s1_engine *e) { return e->n_embd * (1 + e->cap.n); }

int s1_tokenize(const struct s1_engine *e, const char *text, int len, bool parse_special,
                int32_t *tok, int cap)
{
    int32_t n = llama_tokenize(e->vocab, text, len, tok, cap, false, parse_special);
    if (n < 0) {
        fprintf(stderr, "judgly: tokenising %d bytes needs %d tokens, room for %d\n", len, -n, cap);
        return -1;
    }
    return n;
}

int s1_detokenize(const struct s1_engine *e, struct s1_span span, char *text, int cap)
{
    int32_t n = llama_detokenize(e->vocab, span.tok, span.n, text, cap - 1, false, true);
    if (n < 0) {
        fprintf(stderr, "judgly: detokenising %d tokens needs %d bytes, room for %d\n", span.n, -n,
                cap - 1);
        return -1;
    }
    text[n] = '\0';
    return n;
}

bool s1_token_is_control(const struct s1_engine *e, int32_t tok)
{
    return llama_vocab_is_control(e->vocab, tok);
}

static int decode(struct s1_engine *e)
{
    int32_t rc = llama_decode(e->ctx, e->batch);
    if (rc != 0) {
        fprintf(stderr, "judgly: llama_decode returned %d on a batch of %d tokens\n", rc,
                e->batch.n_tokens);
        return -1;
    }
    e->batch.n_tokens = 0;
    return 0;
}

/* Appends a run to the batch as sequence seq at positions pos0 onward, decoding whenever
 * the batch is full. The last partial batch is left for the caller to decode. With
 * last_index set, the final token is flagged for output and its batch index stored. */
static int feed(struct s1_engine *e, struct s1_span run, int pos0, int seq, int32_t *last_index)
{
    struct llama_batch *b = &e->batch;
    for (int i = 0; i < run.n; i++) {
        if (b->n_tokens == S1_N_BATCH && decode(e) != 0) {
            return -1;
        }
        bool    want_out = last_index && i == run.n - 1;
        int32_t at       = b->n_tokens++;
        b->token[at]     = run.tok[i];
        b->pos[at]       = pos0 + i;
        b->n_seq_id[at]  = 1;
        b->seq_id[at][0] = seq;
        b->logits[at]    = want_out;
        if (want_out) {
            *last_index = at;
        }
    }
    return 0;
}

/* Decodes a batch that holds exactly the `count` decision positions and copies out h, the
 * captured layers and the logits of each. */
static int decode_outputs(struct s1_engine *e, int count, struct s1_out *out)
{
    int n_tokens = e->batch.n_tokens;
    llama_set_embeddings(e->ctx, true);
    e->cap.armed  = true;
    e->cap.failed = false;
    for (int i = 0; i < e->cap.n; i++) {
        e->cap.rows[i] = 0;
    }
    int rc = decode(e);
    e->cap.armed = false;
    llama_set_embeddings(e->ctx, false);
    for (int i = 0; i < e->cap.n && rc == 0; i++) {
        if (e->cap.failed || e->cap.rows[i] != n_tokens) {
            fprintf(stderr, "judgly: captured %d rows of %s, expected %d\n", e->cap.rows[i],
                    e->cap.name[i], n_tokens);
            rc = -1;
        }
    }
    size_t n_embd = (size_t)e->n_embd;
    for (int j = 0; j < count && rc == 0; j++) {
        const float *h = llama_get_embeddings_ith(e->ctx, e->out_index[j]);
        const float *z = llama_get_logits_ith(e->ctx, e->out_index[j]);
        if (!h || !z) {
            fprintf(stderr, "judgly: no %s for batch index %d\n", h ? "logits" : "embeddings",
                    e->out_index[j]);
            return -1;
        }
        memcpy(out[j].h, h, n_embd * sizeof *h);
        memcpy(out[j].z, z, (size_t)e->n_vocab * sizeof *z);
        for (int i = 0; i < e->cap.n; i++) {
            const float *row = e->cap.buf[i] + (size_t)e->out_index[j] * n_embd;
            float       *dst = out[j].h + (size_t)(i + 1) * n_embd;
            double       ss  = 0.0;
            for (size_t k = 0; k < n_embd; k++) {
                ss += (double)row[k] * (double)row[k];
            }
            float scale = e->cap.layer[i] == S1_LAYER_NORM ? 1.0f
                                                           : (float)(1.0 / sqrt(ss / (double)n_embd + 1e-6));
            for (size_t k = 0; k < n_embd; k++) {
                dst[k] = row[k] * scale;
            }
        }
    }
    return rc;
}

/* Decodes the last token of each of `count` runs and copies out h and the logits there.
 *
 * With embeddings enabled, libllama at the pinned commit outputs every token of a batch,
 * whatever the per-token flags say, and computes the full vocabulary projection for each.
 * So everything but the last token of each run is decoded with embeddings off and no
 * outputs, and embeddings are switched on only for this batch of one token per run. */
static int read_last(struct s1_engine *e, const struct s1_span *run, int count, int pos0,
                     int seq0, struct s1_out *out)
{
    if (e->batch.n_tokens > 0 && decode(e) != 0) {
        return -1;
    }
    for (int j = 0; j < count; j++) {
        struct s1_span last = { run[j].tok + run[j].n - 1, 1 };
        if (feed(e, last, pos0 + run[j].n - 1, seq0 + j, &e->out_index[j]) != 0) {
            return -1;
        }
    }
    return decode_outputs(e, count, out);
}

/* How many of the next runs go into one call: at most max_seq sequences and one batch of
 * tokens. A run longer than a batch goes alone, and feed() splits it. */
static int group_size(const struct s1_span *run, int n_left, int max_seq)
{
    int count = 0;
    int n_tok = 0;
    while (count < n_left && count < max_seq &&
           (count == 0 || n_tok + run[count].n <= S1_N_BATCH)) {
        n_tok += run[count].n;
        count++;
    }
    return count;
}

int s1_engine_state(struct s1_engine *e, struct s1_span prefix)
{
    llama_memory_clear(e->mem, true);
    e->n_state = 0;
    if (feed(e, prefix, 0, 0, NULL) != 0 || (e->batch.n_tokens > 0 && decode(e) != 0)) {
        return -1;
    }
    e->n_state = prefix.n;
    return 0;
}

int s1_engine_branches(struct s1_engine *e, const struct s1_span *suffix, int n,
                       struct s1_out *out)
{
    if (e->n_state == 0) {
        fprintf(stderr, "judgly: s1_engine_branches called with no state\n");
        return -1;
    }
    for (int first = 0; first < n;) {
        int count = group_size(suffix + first, n - first, e->n_seq - 1);
        for (int j = 0; j < count; j++) {
            llama_memory_seq_cp(e->mem, 0, j + 1, -1, -1); /* branch j+1 sees the whole state */
        }
        for (int j = 0; j < count; j++) {
            struct s1_span body = { suffix[first + j].tok, suffix[first + j].n - 1 };
            if (feed(e, body, e->n_state, j + 1, NULL) != 0) {
                return -1;
            }
        }
        if (read_last(e, suffix + first, count, e->n_state, 1, out + first) != 0) {
            return -1;
        }
        for (int j = 0; j < count; j++) {
            llama_memory_seq_rm(e->mem, j + 1, -1, -1); /* drop the branch, keep sequence 0 */
        }
        first += count;
    }
    return 0;
}

/* One call for the prefixes, one for their branches. Sequence ids: prefix i is sequence i,
 * branch j is sequence n_prefix + j copied from sequence owner[j]. */
int s1_engine_shared(struct s1_engine *e, const struct s1_span *prefix, int n_prefix,
                     const struct s1_span *suffix, const int *owner, int n_suffix,
                     struct s1_out *out)
{
    if (n_prefix + n_suffix > e->n_seq) {
        fprintf(stderr, "judgly: %d prefixes and %d branches exceed n_seq %d\n", n_prefix, n_suffix,
                e->n_seq);
        return -1;
    }
    e->n_state = 0;
    llama_memory_clear(e->mem, true);
    for (int i = 0; i < n_prefix; i++) {
        if (feed(e, prefix[i], 0, i, NULL) != 0) {
            return -1;
        }
    }
    if (e->batch.n_tokens > 0 && decode(e) != 0) {
        return -1;
    }
    for (int j = 0; j < n_suffix; j++) {
        int seq = n_prefix + j;
        llama_memory_seq_cp(e->mem, owner[j], seq, -1, -1);
        struct s1_span body = { suffix[j].tok, suffix[j].n - 1 };
        if (feed(e, body, prefix[owner[j]].n, seq, NULL) != 0) {
            return -1;
        }
    }
    /* read_last needs one pos0 for all runs; the branches start at their own prefix's length,
       so it is called per distinct owner length below */
    if (e->batch.n_tokens > 0 && decode(e) != 0) {
        return -1;
    }
    for (int j = 0; j < n_suffix; j++) {
        struct s1_span last = { suffix[j].tok + suffix[j].n - 1, 1 };
        if (feed(e, last, prefix[owner[j]].n + suffix[j].n - 1, n_prefix + j, &e->out_index[j]) != 0) {
            return -1;
        }
    }
    int rc = decode_outputs(e, n_suffix, out);
    llama_memory_clear(e->mem, true);
    return rc;
}

int s1_engine_packed(struct s1_engine *e, const struct s1_span *seq, int n, struct s1_out *out)
{
    e->n_state = 0;
    for (int first = 0; first < n;) {
        int count = group_size(seq + first, n - first, e->n_seq);
        llama_memory_clear(e->mem, true);
        for (int j = 0; j < count; j++) {
            struct s1_span body = { seq[first + j].tok, seq[first + j].n - 1 };
            if (feed(e, body, 0, j, NULL) != 0) {
                return -1;
            }
        }
        if (read_last(e, seq + first, count, 0, 0, out + first) != 0) {
            return -1;
        }
        first += count;
    }
    llama_memory_clear(e->mem, true);
    return 0;
}
