/* s1-eval: the evaluation report from a feature file and an optional head.
 *
 * Conditions: --rotations averages an item's rotation records, otherwise only rotation 0 is
 * used; --content-free lets the content-free logits in, otherwise they are treated as zero.
 * The report goes to stdout as text, to --json as JSON, and --dump-items writes one line per
 * item for the paired comparison of section 20. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"

#define RESAMPLES 1000
#define SEED      20260922

static const char *const SPLIT_NAME[]  = { "train", "validation", "test", "heldout" };
static const double      THRESHOLD[]   = { 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 };
#define N_THRESHOLDS ((int)(sizeof THRESHOLD / sizeof THRESHOLD[0]))

/* Probabilities of one record over its slots: the head's, or softmax(z - zc) for H0. */
static void record_probs(const struct s1_feat *f, size_t i, const struct s1_head *head,
                         bool content_free, double *p)
{
    static const float           no_zc[S1_K_MAX] = { 0.0f };
    const struct s1_feat_record *r               = &f->rec[i];
    const float                 *zc              = content_free ? r->zc : no_zc;
    struct s1_head               h0              = { 0 };
    double                       identity[S1_HEAD_D] = { 0.0 };
    if (!head) { /* H0 is the identity head with c = 1 when the content-free logits are let in */
        identity[S1_HEAD_C] = 1.0;
        h0.x[r->type]       = identity;
        head                = &h0;
    }
    s1_head_apply(head->x[r->type], head->h2, (int)f->header.n_embd, r->z, zc,
                  f->h + i * f->header.n_embd, r->K, p);
}

/* Only H1 acts on slot logits alone, which every backbone produces in the same 26-slot
 * form; H2's row corrections are tied to one model's h and cannot transfer. */
static int head_transferable(const struct s1_head *head)
{
    if (head->h2) {
        fprintf(stderr, "s1-eval: --transfer needs an H1 head; H2 is tied to its backbone's h\n");
        return -1;
    }
    fprintf(stderr, "s1-eval: applying a head trained on model %.12s... to other features\n",
            head->gguf_sha256);
    return 0;
}

static int compare_by_id(const void *a, const void *b)
{
    const struct s1_pred *x = a;
    const struct s1_pred *y = b;
    return (x->id_hash > y->id_hash) - (x->id_hash < y->id_hash);
}

/* One prediction per item of the split: its records merged, slots mapped back to options.
 * Returns the number of items; *out is owned by the caller. */
static int predictions(const struct s1_feat *f, int split, const struct s1_head *head,
                       bool rotations, bool content_free, struct s1_pred **out)
{
    size_t          n_rec = (size_t)f->header.n_records;
    struct s1_pred *pred  = calloc(n_rec ? n_rec : 1, sizeof *pred);
    if (!pred) {
        fprintf(stderr, "s1-eval: out of memory\n");
        return -1;
    }
    int n = 0;
    for (size_t i = 0; i < n_rec; i++) { /* one entry per record used, merged below */
        const struct s1_feat_record *r = &f->rec[i];
        if (r->split != split || (!rotations && r->rotation != 0 && r->type != S1_BOOL) ||
            (r->type == S1_SCORE && r->rotation != 0)) { /* score: natural order only */
            continue;
        }
        struct s1_pred *x = &pred[n++];
        double          p[S1_K_MAX];
        record_probs(f, i, head, content_free, p);
        *x = (struct s1_pred){ .K = r->K, .label = r->perm[r->label], .type = r->type,
                               .slot_mass = r->slot_mass, .id_hash = r->id_hash,
                               .task_id = r->task_id, .family_id = r->family_id };
        for (int s = 0; s < r->K; s++) {
            x->p[r->perm[s]] = p[s];
        }
    }
    qsort(pred, (size_t)n, sizeof *pred, compare_by_id);
    int n_item = 0;
    for (int i = 0; i < n;) { /* average the records that share an id */
        struct s1_pred item = pred[i];
        int            j;
        for (j = i + 1; j < n && pred[j].id_hash == item.id_hash; j++) {
            for (int k = 0; k < item.K; k++) {
                item.p[k] += pred[j].p[k];
            }
            item.slot_mass += pred[j].slot_mass;
        }
        for (int k = 0; k < item.K; k++) {
            item.p[k] /= j - i;
        }
        item.slot_mass /= j - i;
        pred[n_item++] = item;
        i              = j;
    }
    *out = pred;
    return n_item;
}

static void print_text(const char *title, int n, const double *m, const double *lo,
                       const double *hi, const struct s1_bin *bin, const struct s1_pred *pred)
{
    printf("%s, %d items\n\n  %-14s %10s   95%% interval\n", title, n, "metric", "value");
    for (int i = 0; i < S1_N_METRICS; i++) {
        if (!isnan(m[i])) {
            printf("  %-14s %10.4f   [%.4f, %.4f]\n", S1_METRIC_NAME[i], m[i], lo[i], hi[i]);
        }
    }
    printf("\n  reliability: bin, n, mean confidence, accuracy, 95%% Wilson interval\n");
    for (int b = 0; b < S1_N_BINS; b++) {
        if (bin[b].n > 0) {
            printf("  %.1f-%.1f %6d   %.4f   %.4f   [%.4f, %.4f]\n", b / 10.0, (b + 1) / 10.0,
                   bin[b].n, bin[b].confidence, bin[b].accuracy, bin[b].wilson_lo, bin[b].wilson_hi);
        }
    }
    printf("\n  risk and coverage: threshold, accuracy above it, fraction of items above it\n");
    for (int t = 0; t < N_THRESHOLDS; t++) {
        double accuracy, coverage;
        s1_risk_coverage(pred, n, THRESHOLD[t], &accuracy, &coverage);
        printf("  %.2f   %.4f   %.4f\n", THRESHOLD[t], accuracy, coverage);
    }
}

static void json_number(FILE *out, double v)
{
    if (isnan(v)) {
        fprintf(out, "null");
    } else {
        fprintf(out, "%.10g", v);
    }
}

static int write_json(const char *path, const char *title, int n, const double *m,
                      const double *lo, const double *hi, const struct s1_bin *bin,
                      const struct s1_pred *pred)
{
    FILE *out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "s1-eval: cannot write %s\n", path);
        return -1;
    }
    fprintf(out, "{\"condition\": \"%s\", \"items\": %d,\n \"metrics\": {", title, n);
    for (int i = 0; i < S1_N_METRICS; i++) {
        fprintf(out, "%s\n  \"%s\": {\"value\": ", i ? "," : "", S1_METRIC_NAME[i]);
        json_number(out, m[i]);
        fprintf(out, ", \"lo\": ");
        json_number(out, lo[i]);
        fprintf(out, ", \"hi\": ");
        json_number(out, hi[i]);
        fprintf(out, "}");
    }
    fprintf(out, "},\n \"reliability\": [");
    for (int b = 0; b < S1_N_BINS; b++) {
        fprintf(out, "%s\n  {\"bin\": %d, \"n\": %d, \"confidence\": %.10g, \"accuracy\": %.10g, "
                     "\"wilson_lo\": %.10g, \"wilson_hi\": %.10g}", b ? "," : "", b, bin[b].n,
                bin[b].confidence, bin[b].accuracy, bin[b].wilson_lo, bin[b].wilson_hi);
    }
    fprintf(out, "],\n \"risk_coverage\": [");
    for (int t = 0; t < N_THRESHOLDS; t++) {
        double accuracy, coverage;
        s1_risk_coverage(pred, n, THRESHOLD[t], &accuracy, &coverage);
        fprintf(out, "%s\n  {\"threshold\": %.2f, \"accuracy\": ", t ? "," : "", THRESHOLD[t]);
        json_number(out, accuracy);
        fprintf(out, ", \"coverage\": %.10g}", coverage);
    }
    fprintf(out, "]}\n");
    return fclose(out) == 0 ? 0 : -1;
}

static int dump_items(const char *path, const struct s1_pred *pred, int n)
{
    FILE *out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "s1-eval: cannot write %s\n", path);
        return -1;
    }
    fprintf(out, "id_hash\ttask_id\tfamily_id\ttype\tK\tlabel\tp\n");
    for (int i = 0; i < n; i++) {
        fprintf(out, "%016llx\t%u\t%u\t%d\t%d\t%d\t", (unsigned long long)pred[i].id_hash,
                pred[i].task_id, pred[i].family_id, pred[i].type, pred[i].K, pred[i].label);
        for (int k = 0; k < pred[i].K; k++) {
            fprintf(out, "%s%.17g", k ? "," : "", pred[i].p[k]);
        }
        fprintf(out, "\n");
    }
    return fclose(out) == 0 ? 0 : -1;
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--features", "--head", "--split", "--json",
                                           "--dump-items", NULL };
    static const char *const flags[]   = { "--rotations", "--content-free", "--transfer", "--probe",
                                           NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *features  = s1_arg_value(argc, argv, "--features");
    const char *head_path = s1_arg_value(argc, argv, "--head");
    const char *split     = s1_arg_value(argc, argv, "--split");
    const char *json      = s1_arg_value(argc, argv, "--json");
    const char *dump      = s1_arg_value(argc, argv, "--dump-items");
    bool        rotations = s1_arg_flag(argc, argv, "--rotations");
    bool        cf        = s1_arg_flag(argc, argv, "--content-free");
    bool        transfer  = s1_arg_flag(argc, argv, "--transfer");
    int         code      = -1;
    for (int i = 0; split && i < 4; i++) {
        code = strcmp(split, SPLIT_NAME[i]) == 0 ? i : code;
    }
    if (!features || code < 0) {
        fprintf(stderr, "usage: s1-eval --features FILE.feat --split train|validation|test|heldout\n"
                        "         [--head HEAD.bin] [--rotations] [--content-free] "
                        "[--json FILE] [--dump-items FILE.tsv] [--transfer] [--probe]\n"
                        "       --probe evaluates a head fitted with s1-train --probe (slot logits zeroed).\n"
                        "       --transfer applies an H1 head trained on another backbone;\n"
                        "       the model-hash check is skipped and slot logits are used as they are.\n");
        return 2;
    }

    struct s1_feat  f    = { 0 };
    struct s1_head  head = { 0 };
    struct s1_pred *pred = NULL;
    int             n    = -1;
    int             status = 1;
    if (s1_feat_load(&f, features) == 0 &&
        (!s1_arg_flag(argc, argv, "--probe") || (s1_feat_drop_letters(&f), true)) &&
        (!head_path || (s1_head_load(&head, head_path) == 0 &&
                        (transfer ? head_transferable(&head)
                                  : s1_head_check(&head, f.header.gguf_sha256,
                                                  f.header.template_sha256, f.header.slot)) == 0)) &&
        (!cf || (f.header.flags & S1_FEAT_HAS_ZC) ||
         (fprintf(stderr, "s1-eval: %s carries no content-free logits\n", features), false))) {
        n = predictions(&f, code, head_path ? &head : NULL, rotations, cf, &pred);
    }
    if (n == 0) {
        fprintf(stderr, "s1-eval: no records in split %s\n", split);
    } else if (n > 0) {
        double        m[S1_N_METRICS], lo[S1_N_METRICS], hi[S1_N_METRICS];
        struct s1_bin bin[S1_N_BINS];
        char          title[256];
        snprintf(title, sizeof title, "%s%s%s on %s", head_path ? (head.h2 ? "H2" : "H1") : "H0",
                 rotations ? " + rotations" : "", cf ? " + content-free" : "", split);
        s1_metrics(pred, NULL, n, m);
        s1_reliability(pred, n, bin);
        if (s1_metrics_bootstrap(pred, n, RESAMPLES, SEED, lo, hi) == 0) {
            print_text(title, n, m, lo, hi, bin, pred);
            status = (json && write_json(json, title, n, m, lo, hi, bin, pred) != 0) ||
                             (dump && dump_items(dump, pred, n) != 0)
                         ? 1
                         : 0;
        }
    }
    free(pred);
    s1_head_free(&head);
    s1_feat_free(&f);
    return status;
}
