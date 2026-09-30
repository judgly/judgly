/* Questions to probabilities: rotations, the content-free pass, and the averaging.
 * Order of application, per rotation: slot logits, content-free
 * subtraction, softmax (or the H1/H2 head), slots back to options; across rotations: the mean;
 * then, for a temperature head only, the temperature of the question's type. */
#include "s1.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SUFFIX_CAP 4096 /* tokens of one question; one batch */
#define CHUNK      64   /* branches whose full outputs are held at once */

static const char CONTENT_FREE_STATE[] = "N/A";

/* The tokens of one rotation of one question. */
struct branch {
    int32_t *tok;
    int      n;
};

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int s1_n_rotations(enum s1_type type, int K, struct s1_decide_opts opts)
{
    int n = type == S1_SCORE ? 1 : (opts.rotations || type == S1_BOOL) ? K : 1;
    return opts.max_rotations > 0 && n > opts.max_rotations ? opts.max_rotations : n;
}

int s1_rotation_of(int j, int n, int K)
{
    return (int)((long)j * K / n);
}

static int branch_init(struct branch *b, const struct s1_engine *e, const struct s1_template *t,
                       const struct s1_ask *a, int rotation, int32_t *scratch)
{
    const char *shown[S1_K_MAX];
    for (int s = 0; s < a->K; s++) {
        shown[s] = a->option[(s + rotation) % a->K];
    }
    struct s1_question q = { a->instructions, shown, a->K };
    int                n = s1_prompt_suffix(e, t, &q, scratch, SUFFIX_CAP);
    if (n < 0) {
        return -1;
    }
    b->n   = n;
    b->tok = malloc((size_t)n * sizeof *b->tok);
    if (!b->tok) {
        fprintf(stderr, "judgly: out of memory\n");
        return -1;
    }
    memcpy(b->tok, scratch, (size_t)n * sizeof *b->tok);
    return 0;
}

/* Keeps what a readout needs from one full backbone output: the slot logits, and for the
 * real state also the slot mass and h. */
static void reduce(struct s1_readouts *r, int at, const struct s1_out *out, const int32_t *slot,
                   int K, bool content_free, size_t n_embd, int n_vocab)
{
    struct s1_readout *item = &r->item[at];
    for (int s = 0; s < K; s++) {
        (content_free ? item->zc : item->z)[s] = out->z[slot[s]];
    }
    if (!content_free) {
        item->slot_mass = (float)s1_slot_mass(out->z, n_vocab, slot, K);
        memcpy(r->h_block + (size_t)at * n_embd, out->h, n_embd * sizeof(float));
    }
}

/* Processes one state and every branch against it. Fills z, slot_mass and h of each readout,
 * or zc when content_free is set. Adds the elapsed time to `timing`. */
static int run_pass(struct s1_engine *e, const struct s1_template *t, const int32_t *slot,
                    const char *state, const struct s1_ask *ask, const struct branch *branch,
                    struct s1_readouts *r, struct s1_out *out, bool content_free,
                    struct s1_timing *timing)
{
    int      cap    = (int)(strlen(t->open) + strlen(state)) + 16; /* a token is at least a byte */
    int32_t *prefix = malloc((size_t)cap * sizeof *prefix);
    int      n      = prefix ? s1_prompt_prefix(e, t, state, prefix, cap) : -1;
    double   t0     = now_ms();
    int      rc     = n < 0 ? -1 : s1_engine_state(e, (struct s1_span){ prefix, n });
    double   t1     = now_ms();
    free(prefix);

    size_t n_embd  = (size_t)s1_engine_h_stride(e); /* h and the captured layers */
    int    n_vocab = s1_engine_n_vocab(e);
    for (int first = 0; rc == 0 && first < r->n; first += CHUNK) {
        int            count = r->n - first < CHUNK ? r->n - first : CHUNK;
        struct s1_span suffix[CHUNK];
        for (int j = 0; j < count; j++) {
            suffix[j] = (struct s1_span){ branch[first + j].tok, branch[first + j].n };
        }
        rc = s1_engine_branches(e, suffix, count, out);
        for (int j = 0; rc == 0 && j < count; j++) {
            reduce(r, first + j, &out[j], slot, ask[r->item[first + j].ask].K, content_free,
                   n_embd, n_vocab);
            timing->n_question_tokens += content_free ? 0 : branch[first + j].n;
        }
    }
    if (!content_free) {
        timing->n_state_tokens = n;
    }
    timing->state_ms += t1 - t0;
    timing->questions_ms += now_ms() - t1;
    return rc;
}

static void softmax(const double *u, int K, double *p)
{
    double umax = u[0];
    for (int k = 1; k < K; k++) {
        umax = fmax(umax, u[k]);
    }
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        p[k] = exp(u[k] - umax);
        sum += p[k];
    }
    for (int k = 0; k < K; k++) {
        p[k] /= sum;
    }
}

/* Sets up one readout and one branch per rotation of every question. */
static int plan(struct s1_engine *e, const struct s1_template *t, const struct s1_ask *ask,
                int n_ask, struct s1_decide_opts opts, struct s1_readouts *r,
                struct branch **branch)
{
    size_t n_embd = (size_t)s1_engine_h_stride(e); /* h and the captured layers */
    for (int i = 0; i < n_ask; i++) {
        r->n += s1_n_rotations(ask[i].type, ask[i].K, opts);
    }
    r->item          = calloc((size_t)r->n, sizeof *r->item);
    r->h_block       = malloc((size_t)r->n * n_embd * sizeof *r->h_block);
    *branch          = calloc((size_t)r->n, sizeof **branch);
    int32_t *scratch = malloc(SUFFIX_CAP * sizeof *scratch);
    int      rc      = r->item && r->h_block && *branch && scratch ? 0 : -1;
    if (rc != 0) {
        fprintf(stderr, "judgly: out of memory\n");
    }
    int at = 0;
    for (int i = 0; i < n_ask && rc == 0; i++) {
        int n_rot = s1_n_rotations(ask[i].type, ask[i].K, opts);
        for (int j = 0; j < n_rot && rc == 0; j++, at++) {
            int rot              = s1_rotation_of(j, n_rot, ask[i].K);
            r->item[at].ask      = i;
            r->item[at].rotation = rot;
            r->item[at].h        = r->h_block + (size_t)at * n_embd;
            rc = branch_init(&(*branch)[at], e, t, &ask[i], rot, scratch);
        }
    }
    free(scratch);
    return rc;
}

static int outs_init(struct s1_out *out, const struct s1_engine *e)
{
    size_t n_embd  = (size_t)s1_engine_h_stride(e); /* h and the captured layers */
    size_t n_vocab = (size_t)s1_engine_n_vocab(e);
    for (int j = 0; j < CHUNK; j++) {
        out[j].h = malloc(n_embd * sizeof *out[j].h);
        out[j].z = malloc(n_vocab * sizeof *out[j].z);
        if (!out[j].h || !out[j].z) {
            fprintf(stderr, "judgly: out of memory\n");
            return -1;
        }
    }
    return 0;
}

static void outs_free(struct s1_out *out)
{
    for (int j = 0; j < CHUNK; j++) {
        free(out[j].h);
        free(out[j].z);
    }
}

static void branches_free(struct branch *branch, int n)
{
    for (int j = 0; branch && j < n; j++) {
        free(branch[j].tok);
    }
    free(branch);
}

int s1_read(struct s1_engine *e, const struct s1_template *t, const int32_t slot[S1_K_MAX],
            const char *state, const struct s1_ask *ask, int n_ask, struct s1_decide_opts opts,
            struct s1_readouts *r, struct s1_timing *timing)
{
    struct branch *branch     = NULL;
    struct s1_out  out[CHUNK] = { { 0 } };
    memset(r, 0, sizeof *r);
    memset(timing, 0, sizeof *timing);

    int rc = plan(e, t, ask, n_ask, opts, r, &branch);
    if (rc == 0) {
        rc = outs_init(out, e);
    }
    if (rc == 0) {
        rc = run_pass(e, t, slot, state, ask, branch, r, out, false, timing);
    }
    if (rc == 0 && opts.content_free) {
        rc = run_pass(e, t, slot, CONTENT_FREE_STATE, ask, branch, r, out, true, timing);
    }
    outs_free(out);
    branches_free(branch, r->n);
    if (rc != 0) {
        s1_readouts_free(r);
    }
    return rc;
}

/* One tokenised prefix. */
struct prefix {
    int32_t *tok;
    int      n;
};

static int prefix_init(struct prefix *p, const struct s1_engine *e, const struct s1_template *t,
                       const char *state)
{
    int cap = (int)(strlen(t->open) + strlen(state)) + 16; /* a token is at least a byte */
    p->tok  = malloc((size_t)cap * sizeof *p->tok);
    p->n    = p->tok ? s1_prompt_prefix(e, t, state, p->tok, cap) : -1;
    return p->n < 0 ? -1 : 0;
}

/* Runs the readouts in blocks through s1_engine_shared: each block holds up to CHUNK branches
 * and the prefixes they continue, within the engine's sequence limit and its cache cells. prefix[i] belongs to
 * ask i, or prefix[0] to every branch when `shared` (the content-free state). */
static int run_shared(struct s1_engine *e, const int32_t *slot, const struct s1_ask *ask,
                      const struct prefix *prefix, bool shared, const struct branch *branch,
                      struct s1_readouts *r, struct s1_out *out, struct s1_timing *timing)
{
    size_t n_embd  = (size_t)s1_engine_h_stride(e); /* h and the captured layers */
    int    n_vocab = s1_engine_n_vocab(e);
    int    n_seq   = s1_engine_n_seq(e);
    int    n_ctx   = s1_engine_n_ctx(e);
    int    rc      = 0;
    double t0      = now_ms();
    for (int first = 0; rc == 0 && first < r->n;) {
        struct s1_span block_prefix[CHUNK];
        struct s1_span suffix[CHUNK];
        int            owner[CHUNK];
        int            n_prefix = 0;
        int            count    = 0;
        int            last_ask = -1;
        long           cells    = 0; /* cache cells the block needs: its prefixes and branches */
        while (first + count < r->n && count < CHUNK) {
            int which = shared ? 0 : r->item[first + count].ask;
            bool new_prefix = n_prefix == 0 || (!shared && which != last_ask);
            long need = cells + (new_prefix ? prefix[which].n : 0) + branch[first + count].n;
            if (n_prefix + new_prefix + count + 1 > n_seq || (count > 0 && need > n_ctx)) {
                break;
            }
            cells = need;
            if (new_prefix) {
                block_prefix[n_prefix++] = (struct s1_span){ prefix[which].tok, prefix[which].n };
                last_ask                 = which;
                timing->n_state_tokens += shared ? 0 : prefix[which].n;
            }
            const struct branch *b = &branch[first + count];
            suffix[count]          = (struct s1_span){ b->tok, b->n };
            owner[count]           = n_prefix - 1;
            timing->n_question_tokens += shared ? 0 : b->n;
            count++;
        }
        rc = s1_engine_shared(e, block_prefix, n_prefix, suffix, owner, count, out);
        for (int j = 0; rc == 0 && j < count; j++) {
            reduce(r, first + j, &out[j], slot, ask[r->item[first + j].ask].K, shared, n_embd,
                   n_vocab);
        }
        first += count;
    }
    timing->questions_ms += now_ms() - t0;
    return rc;
}

int s1_read_packed(struct s1_engine *e, const struct s1_template *t,
                   const int32_t slot[S1_K_MAX], const char *const *state,
                   const struct s1_ask *ask, int n, struct s1_decide_opts opts,
                   struct s1_readouts *r, struct s1_timing *timing)
{
    struct branch *branch     = NULL;
    struct s1_out  out[CHUNK] = { { 0 } };
    struct prefix *prefix     = calloc((size_t)n + 1, sizeof *prefix); /* the last is "N/A" */
    memset(r, 0, sizeof *r);
    memset(timing, 0, sizeof *timing);

    int rc = prefix ? plan(e, t, ask, n, opts, r, &branch) : -1;
    for (int i = 0; i < n && rc == 0; i++) {
        rc = prefix_init(&prefix[i], e, t, state[i]);
    }
    if (rc == 0) {
        rc = prefix_init(&prefix[n], e, t, CONTENT_FREE_STATE);
    }
    if (rc == 0) {
        rc = outs_init(out, e);
    }
    if (rc == 0) {
        rc = run_shared(e, slot, ask, prefix, false, branch, r, out, timing);
    }
    if (rc == 0 && opts.content_free) {
        rc = run_shared(e, slot, ask, &prefix[n], true, branch, r, out, timing);
    }
    outs_free(out);
    branches_free(branch, r->n);
    for (int i = 0; prefix && i <= n; i++) {
        free(prefix[i].tok);
    }
    free(prefix);
    if (rc != 0) {
        s1_readouts_free(r);
    }
    return rc;
}

void s1_readouts_free(struct s1_readouts *r)
{
    free(r->item);
    free(r->h_block);
    memset(r, 0, sizeof *r);
}

void s1_readout_probs(const struct s1_readout *item, const struct s1_head *head,
                      enum s1_type type, int K, double *p)
{
    if (head && !head->temperature) { /* a temperature head acts after the mean (s1_combine) */
        s1_head_apply(head->x[type], head->h2, head->n_embd, item->z, item->zc, item->h, K, p);
        return;
    }
    double u[S1_K_MAX];
    for (int s = 0; s < K; s++) {
        u[s] = (double)item->z[s] - (double)item->zc[s];
    }
    softmax(u, K, p);
}

void s1_combine(const struct s1_ask *ask, int n_ask, const struct s1_readouts *r,
                const struct s1_head *head, struct s1_reply *reply)
{
    memset(reply, 0, (size_t)n_ask * sizeof *reply);
    for (int j = 0; j < r->n; j++) {
        const struct s1_readout *item = &r->item[j];
        const struct s1_ask     *a    = &ask[item->ask];
        struct s1_reply         *out  = &reply[item->ask];
        double                   p[S1_K_MAX];
        s1_readout_probs(item, head, a->type, a->K, p);
        int top = 0;
        for (int s = 0; s < a->K; s++) { /* slot s shows option (s + rotation) mod K */
            out->p[(s + item->rotation) % a->K] += p[s];
            top = p[s] > p[top] ? s : top;
        }
        out->top[out->n_rotation] = (top + item->rotation) % a->K;
        out->slot_mass += (double)item->slot_mass;
        out->n_rotation++;
    }
    for (int i = 0; i < n_ask; i++) {
        reply[i].slot_mass /= reply[i].n_rotation;
        for (int k = 0; k < ask[i].K; k++) {
            reply[i].p[k] /= reply[i].n_rotation;
        }
        if (head && head->temperature) {
            s1_temperature_apply(head->x[ask[i].type][0], ask[i].K, reply[i].p);
        }
    }
}

int s1_decide(struct s1_engine *e, const struct s1_template *t, const int32_t slot[S1_K_MAX],
              const char *state, const struct s1_ask *ask, int n_ask, struct s1_decide_opts opts,
              const struct s1_head *head, struct s1_reply *reply, struct s1_timing *timing)
{
    struct s1_readouts r;
    if (s1_read(e, t, slot, state, ask, n_ask, opts, &r, timing) != 0) {
        return -1;
    }
    s1_combine(ask, n_ask, &r, head, reply);
    s1_readouts_free(&r);
    return 0;
}
