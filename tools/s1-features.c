/* s1-features: examples (JSON Lines) in, a feature file out.
 *
 * Every example passes through the backbone once: its state, then every rotation of its
 * question as a branch, and with --content-free the same branches against the state "N/A".
 * One record per rotation. A sidecar FILE.names.tsv maps task and family ids to names. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "s1.h"

#define N_CTX 32768
#define N_SEQ 128 /* prefixes and their branches of one block */

struct tool {
    struct s1_engine     *engine;
    struct s1_template    tpl;
    struct s1_decide_opts opts;
    struct s1_feat_writer writer;
    struct s1_feat_writer layer_writer[S1_LAYERS_MAX]; /* one file per captured layer */
    int                   layer[S1_LAYERS_MAX];
    int                   n_layers;
    FILE                 *names;
    int32_t               slot[S1_K_MAX];
    uint64_t              seen[256]; /* name hashes already written to the sidecar */
    int                   n_seen;
    long                  n_tokens;
};

/* Writes "kind, id, name" to the sidecar the first time a name is seen. */
static void note_name(struct tool *t, const char *kind, uint64_t id, const char *name)
{
    uint64_t mark = s1_hash64(kind) ^ s1_hash64(name);
    for (int i = 0; i < t->n_seen; i++) {
        if (t->seen[i] == mark) {
            return;
        }
    }
    if (t->n_seen < (int)(sizeof t->seen / sizeof t->seen[0])) {
        t->seen[t->n_seen++] = mark;
    }
    fprintf(t->names, "%s\t%llu\t%s\n", kind, (unsigned long long)id, name);
}

#define BLOCK 32 /* examples handed to the backbone together */

static int write_record(struct tool *t, const struct s1_example *x, const struct s1_readout *item)
{
    struct s1_feat_record rec = { .id_hash   = s1_hash64(x->id),
                                  .task_id   = (uint32_t)s1_hash64(x->task),
                                  .family_id = (uint16_t)s1_hash64(x->family),
                                  .type      = (uint8_t)x->ask.type,
                                  .split     = (uint8_t)x->split,
                                  .K         = (uint8_t)x->ask.K,
                                  .rotation  = (uint8_t)item->rotation,
                                  .slot_mass = item->slot_mass };
    for (int s = 0; s < x->ask.K; s++) { /* slot s shows option (s + rotation) mod K */
        rec.perm[s] = (uint8_t)((s + item->rotation) % x->ask.K);
        if (rec.perm[s] == x->label) {
            rec.label = (uint8_t)s;
        }
    }
    rec.target[rec.label] = 1.0f;
    memcpy(rec.z, item->z, sizeof rec.z);
    memcpy(rec.zc, item->zc, sizeof rec.zc);
    int    rc     = s1_feat_append(&t->writer, &rec, item->h);
    size_t n_embd = (size_t)s1_engine_n_embd(t->engine);
    for (int i = 0; i < t->n_layers && rc == 0; i++) { /* the same record, that layer's vector */
        rc = s1_feat_append(&t->layer_writer[i], &rec, item->h + (size_t)(i + 1) * n_embd);
    }
    return rc;
}

/* Runs a block of examples through the backbone and writes a record per rotation. */
static int extract_block(struct tool *t, const struct s1_example *x, int n)
{
    const char        *state[BLOCK];
    struct s1_ask      ask[BLOCK];
    struct s1_readouts r;
    struct s1_timing   timing;
    for (int i = 0; i < n; i++) {
        state[i] = x[i].state;
        ask[i]   = x[i].ask;
        note_name(t, "task", (uint32_t)s1_hash64(x[i].task), x[i].task);
        note_name(t, "family", (uint16_t)s1_hash64(x[i].family), x[i].family);
    }
    if (s1_read_packed(t->engine, &t->tpl, t->slot, state, ask, n, t->opts, &r, &timing) != 0) {
        return -1;
    }
    t->n_tokens += timing.n_state_tokens + timing.n_question_tokens;
    int rc = 0;
    for (int j = 0; j < r.n && rc == 0; j++) {
        rc = write_record(t, &x[r.item[j].ask], &r.item[j]);
    }
    s1_readouts_free(&r);
    return rc;
}

static int extract_all(struct tool *t, FILE *in, long limit)
{
    struct s1_example x[BLOCK];
    char          *text = NULL;
    size_t         cap  = 0;
    long           line = 0;
    int            n    = 0;
    int            rc   = 0;
    bool           more = true;
    while (rc == 0 && more) {
        more = (limit == 0 || line < limit) && getline(&text, &cap, in) > 0;
        if (more) {
            rc = s1_example_parse(&x[n], text, ++line);
            n++; /* counted even on failure, so that its document is released below */
        }
        if (rc == 0 && n > 0 && (n == BLOCK || !more)) {
            rc = extract_block(t, x, n);
        }
        if (rc != 0 || n == BLOCK || !more) {
            for (int i = 0; i < n; i++) {
                s1_example_free(&x[i]);
            }
            n = 0;
            fprintf(stderr, "s1-features: %ld examples, %llu records\n", line,
                    (unsigned long long)t->writer.n_records);
        }
    }
    free(text);
    return rc;
}

static int open_outputs(struct tool *t, const char *model, const char *tpl_path, const char *out)
{
    struct s1_feat_header header = { .n_embd = (uint32_t)s1_engine_n_embd(t->engine),
                                     .flags  = t->opts.content_free ? S1_FEAT_HAS_ZC : 0 };
    char                  names[4096];
    memcpy(header.slot, t->slot, sizeof header.slot);
    if (s1_sha256_file(model, header.gguf_sha256) != 0 ||
        s1_sha256_file(tpl_path, header.template_sha256) != 0 ||
        snprintf(names, sizeof names, "%s.names.tsv", out) >= (int)sizeof names) {
        return -1;
    }
    t->names = fopen(names, "w");
    if (!t->names) {
        fprintf(stderr, "s1-features: cannot write %s\n", names);
        return -1;
    }
    if (s1_feat_create(&t->writer, out, &header) != 0) {
        return -1;
    }
    for (int i = 0; i < t->n_layers; i++) {
        char   path[4096];
        size_t stem = strlen(out) > 5 && strcmp(out + strlen(out) - 5, ".feat") == 0 ? strlen(out) - 5
                                                                                    : strlen(out);
        int    n    = t->layer[i] == S1_LAYER_NORM
                          ? snprintf(path, sizeof path, "%.*s.Lnorm.feat", (int)stem, out)
                          : snprintf(path, sizeof path, "%.*s.L%02d.feat", (int)stem, out, t->layer[i]);
        if (n >= (int)sizeof path || s1_feat_create(&t->layer_writer[i], path, &header) != 0) {
            return -1;
        }
    }
    return 0;
}

/* "9,18,norm" to layer ids; "norm" is the final-norm output itself (the capture check). */
static int parse_layers(struct tool *t, const char *list)
{
    for (const char *at = list; at && *at && t->n_layers < S1_LAYERS_MAX;) {
        char *end = NULL;
        if (strncmp(at, "norm", 4) == 0) {
            t->layer[t->n_layers++] = S1_LAYER_NORM;
            end = (char *)at + 4;
        } else {
            long l = strtol(at, &end, 10);
            if (end == at || l < 0) {
                fprintf(stderr, "s1-features: --layers takes numbers or \"norm\", comma-separated\n");
                return -1;
            }
            t->layer[t->n_layers++] = (int)l;
        }
        at = *end == ',' ? end + 1 : end;
        if (*end && *end != ',') {
            fprintf(stderr, "s1-features: cannot read --layers %s\n", list);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--model", "--template", "--examples", "--out", "--limit",
                                           "--max-rotations", "--layers", NULL };
    static const char *const flags[]   = { "--rotations", "--content-free", "--plain-slots", NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *model    = s1_arg_value(argc, argv, "--model");
    const char *tpl_path = s1_arg_value(argc, argv, "--template");
    const char *examples = s1_arg_value(argc, argv, "--examples");
    const char *out      = s1_arg_value(argc, argv, "--out");
    const char *limit    = s1_arg_value(argc, argv, "--limit");
    if (!model) {
        model = getenv("S1_MODEL");
    }
    if (!model || !tpl_path || !examples || !out) {
        fprintf(stderr, "usage: s1-features --model FILE.gguf --template FILE.tpl "
                        "--examples FILE.jsonl --out FILE.feat\n"
                        "         [--rotations] [--max-rotations N] [--content-free] [--limit N] [--plain-slots]\n"
                        "         [--layers L,L,...|norm]   also write FILE.Lnn.feat per layer\n");
        return 2;
    }
    struct tool t = { .opts = { s1_arg_flag(argc, argv, "--rotations"),
                                s1_arg_flag(argc, argv, "--content-free"),
                                (int)strtol(s1_arg_value(argc, argv, "--max-rotations")
                                                ? s1_arg_value(argc, argv, "--max-rotations") : "0",
                                            NULL, 10) } };
    if (parse_layers(&t, s1_arg_value(argc, argv, "--layers")) != 0) {
        return 2;
    }
    struct s1_engine_params ep = { model, N_CTX, N_SEQ, 999, false, t.layer, t.n_layers };
    FILE  *in     = fopen(examples, "r");
    int    status = 1;
    time_t start  = time(NULL);
    if (!in) {
        fprintf(stderr, "s1-features: cannot read %s\n", examples);
    } else if (s1_engine_init(&t.engine, &ep) == 0 && s1_template_load(&t.tpl, tpl_path) == 0 &&
               s1_slots_init(t.engine, !s1_arg_flag(argc, argv, "--plain-slots"), t.slot) == 0 &&
               open_outputs(&t, model, tpl_path, out) == 0 &&
               extract_all(&t, in, limit ? strtol(limit, NULL, 10) : 0) == 0) {
        status = 0;
    }
    if (t.writer.file && s1_feat_close(&t.writer) != 0) {
        status = 1;
    }
    for (int i = 0; i < t.n_layers; i++) {
        if (t.layer_writer[i].file && s1_feat_close(&t.layer_writer[i]) != 0) {
            status = 1;
        }
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double seconds = difftime(time(NULL), start);
    fprintf(stderr, "s1-features: %llu records, %ld tokens in %.0f s (%.0f tokens/s), peak "
                    "memory %.1f GB\n", (unsigned long long)t.writer.n_records, t.n_tokens, seconds,
            seconds > 0 ? (double)t.n_tokens / seconds : 0.0,
            (double)usage.ru_maxrss / (1024.0 * 1024.0 * 1024.0));
    if (t.names) {
        fclose(t.names);
    }
    if (in) {
        fclose(in);
    }
    s1_template_free(&t.tpl);
    s1_engine_free(t.engine);
    return status;
}
