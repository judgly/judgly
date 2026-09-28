/* s1-selftest: the correctness tests T1 to T10.
 *
 * Prints one line per test with the measured value and the tolerance. Exits 0 only if
 * every test passes. Tolerances belong to the person who owns the project. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"
#include "yyjson.h"

/* Tolerances of the self-tests. T3 and T4 are measured in probability. */
#define TOL_T1 0.0  /* prompts that differ from the intended text, and forged markers */
#define TOL_T2 0.05 /* max |difference| in logit units, on an 8-bit or 16-bit file */
#define TOL_T3 0.02 /* max |difference| in any probability */
#define TOL_T4 0.02 /* max |difference| in any probability */
#define TOL_T5 0.01 /* max |difference| in any probability */
#define TOL_T6 0.05 /* max |difference| in slot log-probability, 16-bit or 8-bit GGUF */
#define TOL_T7 1e-4 /* relative error of analytic against numerical gradients */
#define TOL_T8 0.02 /* max |difference| in any probability */
#define TOL_T10 0.02 /* max |difference| in any probability; sliding-window models only */
#define TOL_T9 0.0  /* rotations whose top option is not the stated answer; sums off by > 1e-5 */

#define N_CTX       32768 /* the long state, its branches, and packed groups */
#define N_SEQ       17    /* 16 branches and the state */
#define PREFIX_CAP  16384
#define SUFFIX_CAP  512
#define Q_PER_STATE 8
#define TEXT_CAP    65536 /* bytes of one detokenised prompt piece */
#define T6_FIXTURES 5     /* the first four fixtures and the last, which has the long state */
#define T6_QUESTIONS 2

struct question {
    struct s1_question q;
    int32_t            tok[SUFFIX_CAP];
    int                n;
};

struct fixture {
    const char     *id;
    const char     *state;
    int32_t        *prefix; /* PREFIX_CAP tokens */
    int             n_prefix;
    struct question question[Q_PER_STATE];
};

struct suite {
    struct s1_engine         *engine;
    const struct s1_template *tpl;
    const char               *reference; /* path given with --reference, or NULL */
    const char               *t9_path;   /* requests whose state gives the answer outright */
    const char               *model_path;
    bool                      cpu;       /* --cpu: every layer on the CPU */
    int32_t                  *joined;    /* scratch for one prefix + suffix */
    yyjson_doc               *doc;       /* owns every fixture string */
    struct fixture           *fixture;
    int                       n_fixture;
    int32_t                   slot[S1_K_MAX];
    struct s1_rows            rows;
    struct s1_out             out_a[Q_PER_STATE]; /* scratch outputs, path A */
    struct s1_out             out_b[Q_PER_STATE]; /* and path B */
};

static void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "s1-selftest: out of memory (%zu bytes)\n", size);
        exit(2);
    }
    return p;
}

static int load_question(struct suite *s, const struct s1_template *tpl, yyjson_val *jq,
                         struct question *out)
{
    yyjson_val *options = yyjson_obj_get(jq, "options");
    size_t      K       = yyjson_arr_size(options);
    if (K < 2 || K > S1_K_MAX) {
        fprintf(stderr, "s1-selftest: a question has %zu options\n", K);
        return -1;
    }
    const char **option = xmalloc(K * sizeof *option);
    for (size_t k = 0; k < K; k++) {
        option[k] = yyjson_get_str(yyjson_arr_get(options, k));
    }
    out->q.instructions = yyjson_get_str(yyjson_obj_get(jq, "instructions"));
    out->q.option       = option;
    out->q.K            = (int)K;
    out->n              = s1_prompt_suffix(s->engine, tpl, &out->q, out->tok, SUFFIX_CAP);
    return out->n < 0 ? -1 : 0;
}

static int load_fixtures(struct suite *s, const struct s1_template *tpl, const char *path)
{
    yyjson_read_err err;
    s->doc = yyjson_read_file(path, 0, NULL, &err);
    if (!s->doc) {
        fprintf(stderr, "s1-selftest: cannot read fixtures %s: %s\n", path, err.msg);
        return -1;
    }
    yyjson_val *list = yyjson_obj_get(yyjson_doc_get_root(s->doc), "fixtures");
    s->n_fixture     = (int)yyjson_arr_size(list);
    s->fixture       = xmalloc((size_t)s->n_fixture * sizeof *s->fixture);
    memset(s->fixture, 0, (size_t)s->n_fixture * sizeof *s->fixture);
    for (int i = 0; i < s->n_fixture; i++) {
        struct fixture *f  = &s->fixture[i];
        yyjson_val     *jf = yyjson_arr_get(list, (size_t)i);
        yyjson_val     *jq = yyjson_obj_get(jf, "questions");
        const char     *state = yyjson_get_str(yyjson_obj_get(jf, "state"));
        if (!state || yyjson_arr_size(jq) != Q_PER_STATE) {
            fprintf(stderr, "s1-selftest: fixture %d needs a state and %d questions\n", i,
                    Q_PER_STATE);
            return -1;
        }
        f->id       = yyjson_get_str(yyjson_obj_get(jf, "id"));
        f->state    = state;
        f->prefix   = xmalloc(PREFIX_CAP * sizeof *f->prefix);
        f->n_prefix = s1_prompt_prefix(s->engine, tpl, state, f->prefix, PREFIX_CAP);
        if (f->n_prefix < 0) {
            return -1;
        }
        for (int j = 0; j < Q_PER_STATE; j++) {
            if (load_question(s, tpl, yyjson_arr_get(jq, (size_t)j), &f->question[j]) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* How far two outputs for one question disagree, over its K slots. */
struct gap {
    double p;    /* largest |difference| in probability: what the tests judge */
    double logp; /* largest |difference| in log-probability: reported for the record */
};

/* Widens `worst` by the disagreement between a and b. A probability difference above `loud`
 * is logged with both values, so that a failure shows which option it sits on. */
static void widen_gap(struct gap *worst, const struct suite *s, const struct question *q,
                      const struct s1_out *a, const struct s1_out *b, double loud,
                      const char *where)
{
    double la[S1_K_MAX];
    double lb[S1_K_MAX];
    s1_slot_logprobs(a->z, s->slot, q->q.K, la);
    s1_slot_logprobs(b->z, s->slot, q->q.K, lb);
    for (int k = 0; k < q->q.K; k++) {
        double dp = fabs(exp(la[k]) - exp(lb[k]));
        if (dp > loud) {
            fprintf(stderr, "s1-selftest: %s slot %c: p %.6f against %.6f, |d p| %.6f\n", where,
                    'A' + k, exp(la[k]), exp(lb[k]), dp);
        }
        worst->p    = fmax(worst->p, dp);
        worst->logp = fmax(worst->logp, fabs(la[k] - lb[k]));
    }
}

/* Prefix and suffix as one sequence, in the suite's scratch buffer. */
static struct s1_span join(struct suite *s, const struct fixture *f, const struct question *q)
{
    memcpy(s->joined, f->prefix, (size_t)f->n_prefix * sizeof *s->joined);
    memcpy(s->joined + f->n_prefix, q->tok, (size_t)q->n * sizeof *s->joined);
    return (struct s1_span){ s->joined, f->n_prefix + q->n };
}

static int run_branches(struct suite *s, const struct fixture *f, struct s1_out *out)
{
    struct s1_span suffix[Q_PER_STATE];
    for (int j = 0; j < Q_PER_STATE; j++) {
        suffix[j] = (struct s1_span){ f->question[j].tok, f->question[j].n };
    }
    if (s1_engine_state(s->engine, (struct s1_span){ f->prefix, f->n_prefix }) != 0) {
        return -1;
    }
    return s1_engine_branches(s->engine, suffix, Q_PER_STATE, out);
}

/* A letter after "Answer:" may be expected with or without a leading
 * space, depending on the model. Keeps the variant with the higher mean slot mass over
 * the first SLOT_PROBE fixtures. */
#define SLOT_PROBE 3

static int choose_slots(struct suite *s)
{
    int32_t slot[2][S1_K_MAX];
    bool    valid[2];
    double  mass[2] = { 0.0, 0.0 };
    for (int v = 0; v < 2; v++) {
        valid[v] = s1_slots_init(s->engine, v == 1, slot[v]) == 0;
    }
    if (!valid[0] && !valid[1]) {
        fprintf(stderr, "s1-selftest: neither slot variant is single-token for this model\n");
        return -1;
    }
    int n_vocab = s1_engine_n_vocab(s->engine);
    for (int i = 0; i < SLOT_PROBE && i < s->n_fixture; i++) {
        if (run_branches(s, &s->fixture[i], s->out_a) != 0) {
            return -1;
        }
        for (int j = 0; j < Q_PER_STATE; j++) {
            for (int v = 0; v < 2; v++) {
                if (valid[v]) {
                    mass[v] += s1_slot_mass(s->out_a[j].z, n_vocab, slot[v],
                                            s->fixture[i].question[j].q.K);
                }
            }
        }
    }
    int best = (valid[1] && (!valid[0] || mass[1] > mass[0])) ? 1 : 0;
    memcpy(s->slot, slot[best], sizeof s->slot);
    printf("INFO slots  \"%sA\" chosen  mean slot mass: plain %.4f, leading space %.4f\n",
           best ? " " : "", mass[0] / (SLOT_PROBE * Q_PER_STATE),
           mass[1] / (SLOT_PROBE * Q_PER_STATE));
    return 0;
}

static int report(const char *name, double measured, double tolerance, const char *what)
{
    bool pass = measured <= tolerance;
    printf("%-4s %s  measured %.6f  tolerance %.6f  %s\n", pass ? "PASS" : "FAIL", name,
           measured, tolerance, what);
    return pass ? 0 : 1;
}

static int report_gap(const char *name, struct gap worst, double tolerance, const char *what)
{
    char text[128];
    snprintf(text, sizeof text, "%s, max |d p| (max |d logp| %.6f)", what, worst.logp);
    return report(name, worst.p, tolerance, text);
}

/* T2: dot(w[k], h) from the GGUF rows, through the soft-cap if the model has one, against
 * libllama's logit for the same slot token. All 26 slots at every decision position. */
static int test_t2(struct suite *s)
{
    double worst = 0.0;
    for (int i = 0; i < s->n_fixture; i++) {
        if (run_branches(s, &s->fixture[i], s->out_a) != 0) {
            return -1;
        }
        for (int j = 0; j < Q_PER_STATE; j++) {
            for (int k = 0; k < S1_K_MAX; k++) {
                double ours   = s1_rows_logit(&s->rows, k, s->out_a[j].h);
                double theirs = (double)s->out_a[j].z[s->slot[k]];
                worst         = fmax(worst, fabs(ours - theirs));
            }
        }
    }
    return report("T2", worst, TOL_T2, "rows against logits, max |d logit|");
}

/* T3: the two-call path against a fresh single-sequence run of prefix + suffix. With
 * 8 questions per state and the batch limit, long states span more than one group. */
static int test_t3(struct suite *s)
{
    struct gap worst = { 0.0, 0.0 };
    for (int i = 0; i < s->n_fixture; i++) {
        struct fixture *f = &s->fixture[i];
        if (run_branches(s, f, s->out_a) != 0) {
            return -1;
        }
        for (int j = 0; j < Q_PER_STATE; j++) {
            struct s1_span whole = join(s, f, &f->question[j]);
            if (s1_engine_packed(s->engine, &whole, 1, &s->out_b[j]) != 0) {
                return -1;
            }
            widen_gap(&worst, s, &f->question[j], &s->out_a[j], &s->out_b[j], TOL_T3, f->id);
        }
    }
    return report_gap("T3", worst, TOL_T3, "branch against recompute");
}

/* 1 if the tokens do not detokenise to exactly the intended text. */
static int text_differs(struct suite *s, struct s1_span span, const char *intended,
                        const char *id, const char *piece)
{
    char *text = xmalloc(TEXT_CAP);
    int   n    = s1_detokenize(s->engine, span, text, TEXT_CAP);
    int   bad  = n < 0 || strcmp(text, intended) != 0;
    if (bad) {
        fprintf(stderr, "s1-selftest: T1 %s: %s does not detokenise to the intended text\n", id,
                piece);
    }
    free(text);
    return bad;
}

/* Control tokens inside the state span of a prefix. The span lies between the tokens of
 * OPEN and the tokens of the separator, both of which are tokenised on their own. */
static int forged_markers(struct suite *s, const struct fixture *f)
{
    int32_t tok[SUFFIX_CAP];
    int     n_open = s1_tokenize(s->engine, s->tpl->open, (int)strlen(s->tpl->open), true, tok,
                                 SUFFIX_CAP);
    int     n_sep  = s1_tokenize(s->engine, "\n\n", 2, false, tok, SUFFIX_CAP);
    if (n_open < 0 || n_sep < 0) {
        return 1;
    }
    int forged = 0;
    for (int i = n_open; i < f->n_prefix - n_sep; i++) {
        forged += s1_token_is_control(s->engine, f->prefix[i]);
    }
    return forged;
}

/* T1: every prompt detokenises to the intended text, written out here independently of
 * s1_prompt.c, and no state, including one that spells out chat markers, puts a control
 * token inside the state span. The fixtures have no trailing whitespace in a state and no
 * newline in an option, so the intended text needs neither rule. */
static int test_t1(struct suite *s)
{
    int   bad      = 0;
    char *intended = xmalloc(TEXT_CAP);
    for (int i = 0; i < s->n_fixture; i++) {
        const struct fixture *f = &s->fixture[i];
        snprintf(intended, TEXT_CAP, "%s%s\n\n", s->tpl->open, f->state);
        bad += text_differs(s, (struct s1_span){ f->prefix, f->n_prefix }, intended, f->id,
                            "prefix");
        bad += forged_markers(s, f);
        for (int j = 0; j < Q_PER_STATE; j++) {
            const struct question *q = &f->question[j];
            int at = snprintf(intended, TEXT_CAP, "QUESTION\n%s\nOPTIONS", q->q.instructions);
            for (int k = 0; k < q->q.K; k++) {
                at += snprintf(intended + at, (size_t)(TEXT_CAP - at), "\n%c. %s", 'A' + k,
                               q->q.option[k]);
            }
            snprintf(intended + at, (size_t)(TEXT_CAP - at), "%s", s->tpl->close);
            bad += text_differs(s, (struct s1_span){ q->tok, q->n }, intended, f->id, "suffix");
        }
    }
    free(intended);
    return report("T1", bad, TOL_T1, "prompt integrity, mismatches and forged markers");
}

/* T4: the two feature-extraction paths against one sequence per call: packed (8 whole
 * sequences per call) and shared (8 states, then 8 branches). Group g takes question m of
 * fixture g+m, so a call mixes 8 different states. */
static int test_t4(struct suite *s)
{
    struct gap worst = { 0.0, 0.0 };
    int32_t   *buf   = xmalloc((size_t)Q_PER_STATE * (PREFIX_CAP + SUFFIX_CAP) * sizeof *buf);
    for (int g = 0; g < s->n_fixture; g++) {
        struct s1_span seq[Q_PER_STATE];
        for (int m = 0; m < Q_PER_STATE; m++) {
            const struct fixture *f   = &s->fixture[(g + m) % s->n_fixture];
            struct s1_span        one = join(s, f, &f->question[m]);
            int32_t              *at  = buf + (size_t)m * (PREFIX_CAP + SUFFIX_CAP);
            memcpy(at, one.tok, (size_t)one.n * sizeof *at);
            seq[m] = (struct s1_span){ at, one.n };
        }
        if (s1_engine_packed(s->engine, seq, Q_PER_STATE, s->out_a) != 0) {
            return -1;
        }
        for (int m = 0; m < Q_PER_STATE; m++) {
            const struct fixture *f = &s->fixture[(g + m) % s->n_fixture];
            if (s1_engine_packed(s->engine, &seq[m], 1, &s->out_b[m]) != 0) {
                return -1;
            }
            char where[32];
            snprintf(where, sizeof where, "T4 packed %s", f->id);
            widen_gap(&worst, s, &f->question[m], &s->out_a[m], &s->out_b[m], TOL_T4, where);
        }
        /* the shared-state path: 8 prefixes in one call, one branch each, against the same
           single-sequence results */
        struct s1_span prefix[Q_PER_STATE];
        struct s1_span suffix[Q_PER_STATE];
        int            owner[Q_PER_STATE];
        for (int m = 0; m < Q_PER_STATE; m++) {
            const struct fixture *f = &s->fixture[(g + m) % s->n_fixture];
            prefix[m] = (struct s1_span){ f->prefix, f->n_prefix };
            suffix[m] = (struct s1_span){ f->question[m].tok, f->question[m].n };
            owner[m]  = m;
        }
        if (s1_engine_shared(s->engine, prefix, Q_PER_STATE, suffix, owner, Q_PER_STATE,
                             s->out_a) != 0) {
            return -1;
        }
        for (int m = 0; m < Q_PER_STATE; m++) {
            const struct fixture *f = &s->fixture[(g + m) % s->n_fixture];
            char                  where[32];
            snprintf(where, sizeof where, "T4 shared %s", f->id);
            widen_gap(&worst, s, &f->question[m], &s->out_a[m], &s->out_b[m], TOL_T4, where);
        }
    }
    free(buf);
    return report_gap("T4", worst, TOL_T4, "packed and shared against single");
}

/* T5: the answer to question 0 with three different sets of sibling questions: none,
 * questions 1 to 3, and questions 4 to 7. */
static int test_t5(struct suite *s)
{
    static const int sibling_from[3] = { 0, 1, 4 };
    static const int sibling_to[3]   = { 0, 4, 8 };
    double           worst           = 0.0;
    for (int i = 0; i < s->n_fixture; i++) {
        const struct fixture *f = &s->fixture[i];
        double                p[3][S1_K_MAX];
        if (s1_engine_state(s->engine, (struct s1_span){ f->prefix, f->n_prefix }) != 0) {
            return -1;
        }
        for (int set = 0; set < 3; set++) {
            struct s1_span suffix[Q_PER_STATE];
            int            n = 0;
            suffix[n++]      = (struct s1_span){ f->question[0].tok, f->question[0].n };
            for (int j = sibling_from[set]; j < sibling_to[set]; j++) {
                suffix[n++] = (struct s1_span){ f->question[j].tok, f->question[j].n };
            }
            if (s1_engine_branches(s->engine, suffix, n, s->out_a) != 0) {
                return -1;
            }
            s1_slot_logprobs(s->out_a[0].z, s->slot, f->question[0].q.K, p[set]);
        }
        for (int set = 1; set < 3; set++) {
            for (int k = 0; k < f->question[0].q.K; k++) {
                worst = fmax(worst, fabs(exp(p[set][k]) - exp(p[0][k])));
            }
        }
    }
    return report("T5", worst, TOL_T5, "isolation from sibling questions, max |d p|");
}

/* The T6 items: the first T6_QUESTIONS questions of the first four fixtures and of the
 * last one, whose state is about 8,000 tokens long. */
static const struct fixture *t6_fixture(const struct suite *s, int m)
{
    return &s->fixture[m < T6_FIXTURES - 1 ? m : s->n_fixture - 1];
}

/* Writes the token ids and slot ids of the T6 items for scripts/reference_logits.py. Token
 * ids, never text: the reference must see the prompt token for token. */
static int dump_tokens(struct suite *s, const char *path)
{
    FILE *out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "s1-selftest: cannot write %s\n", path);
        return -1;
    }
    fprintf(out, "{\"items\": [");
    for (int m = 0; m < T6_FIXTURES; m++) {
        const struct fixture *f = t6_fixture(s, m);
        for (int j = 0; j < T6_QUESTIONS; j++) {
            struct s1_span whole = join(s, f, &f->question[j]);
            fprintf(out, "%s\n {\"fixture\": \"%s\", \"question\": %d, \"slots\": [",
                    m + j ? "," : "", f->id, j);
            for (int k = 0; k < f->question[j].q.K; k++) {
                fprintf(out, "%s%d", k ? ", " : "", s->slot[k]);
            }
            fprintf(out, "], \"tokens\": [");
            for (int i = 0; i < whole.n; i++) {
                fprintf(out, "%s%d", i ? ", " : "", whole.tok[i]);
            }
            fprintf(out, "]}");
        }
    }
    fprintf(out, "\n]}\n");
    if (fclose(out) != 0) {
        fprintf(stderr, "s1-selftest: error writing %s\n", path);
        return -1;
    }
    fprintf(stderr, "s1-selftest: wrote %d items to %s\n", T6_FIXTURES * T6_QUESTIONS, path);
    return 0;
}

/* Largest |difference| between our slot log-probabilities and one reference item. The
 * reference must have been computed from exactly the tokens this build produces. */
static int reference_gap(struct suite *s, const struct fixture *f, int j, yyjson_val *item,
                         const struct s1_out *ours, double *worst)
{
    struct s1_span whole  = join(s, f, &f->question[j]);
    yyjson_val    *tokens = yyjson_obj_get(item, "tokens");
    yyjson_val    *logp   = yyjson_obj_get(item, "logp");
    int            K      = f->question[j].q.K;
    bool           same   = (int)yyjson_arr_size(tokens) == whole.n &&
                (int)yyjson_arr_size(logp) == K;
    for (int i = 0; same && i < whole.n; i++) {
        same = yyjson_get_int(yyjson_arr_get(tokens, (size_t)i)) == whole.tok[i];
    }
    if (!same) {
        fprintf(stderr, "s1-selftest: T6 %s question %d: the reference was computed from "
                        "different tokens; regenerate it\n", f->id, j);
        return -1;
    }
    double mine[S1_K_MAX];
    s1_slot_logprobs(ours->z, s->slot, K, mine);
    for (int k = 0; k < K; k++) {
        double theirs = yyjson_get_num(yyjson_arr_get(logp, (size_t)k));
        if (fabs(mine[k] - theirs) > TOL_T6) {
            fprintf(stderr, "s1-selftest: T6 %s q%d (%d tokens) slot %c: logp %.4f against "
                            "reference %.4f, |d logp| %.4f, |d p| %.6f\n", f->id, j, whole.n,
                    'A' + k, mine[k], theirs, fabs(mine[k] - theirs),
                    fabs(exp(mine[k]) - exp(theirs)));
        }
        *worst = fmax(*worst, fabs(mine[k] - theirs));
    }
    return 0;
}

/* The branch results of a second engine, opened with `params`, against those of the main
 * engine, over the first `n_fixture` fixtures. */
static int compare_engines(struct suite *s, struct s1_engine_params params, int n_fixture,
                           const char *what, double *worst)
{
    struct s1_engine *other = NULL;
    struct s1_out    *out   = s->out_b;
    struct s1_engine *main_engine = s->engine;
    int               rc    = s1_engine_init(&other, &params);
    for (int i = 0; rc == 0 && i < n_fixture && i < s->n_fixture; i++) {
        const struct fixture *f = &s->fixture[i];
        rc                      = run_branches(s, f, s->out_a);
        s->engine               = other;
        rc                      = rc == 0 ? run_branches(s, f, out) : rc;
        s->engine               = main_engine;
        for (int j = 0; rc == 0 && j < Q_PER_STATE; j++) {
            struct gap gap = { 0.0, 0.0 };
            widen_gap(&gap, s, &f->question[j], &s->out_a[j], &out[j], 1.0, f->id);
            *worst = fmax(*worst, gap.p);
        }
        fprintf(stderr, "s1-selftest: %s %s: worst so far %.6f\n", what, f->id, *worst);
    }
    s1_engine_free(other);
    return rc;
}

/* T6 substitute for a model too large to run in 32-bit: libllama with every layer on the CPU
 * against libllama on Metal, on the same five fixtures T6 uses. Selected by --reference cpu. */
static int test_t6_cpu(struct suite *s)
{
    struct s1_engine_params params = { s->model_path, N_CTX, N_SEQ, s->cpu ? 999 : 0, false, NULL, 0 };
    double                  worst  = 0.0;
    if (compare_engines(s, params, T6_FIXTURES - 1, "T6", &worst) != 0) {
        return -1;
    }
    struct s1_engine_params long_params = params; /* the long state, alone */
    struct suite            tail        = *s;
    tail.fixture                        = s->fixture + s->n_fixture - 1;
    tail.n_fixture                      = 1;
    if (compare_engines(&tail, long_params, 1, "T6", &worst) != 0) {
        return -1;
    }
    return report("T6", worst, TOL_T6, "CPU against Metal (substitute), max |d p|");
}

/* T10: the pruned sliding-window cache against the full one, on states longer than the
 * window. Sliding-window models only; a failure is reported and swa_full stays true. */
static int test_t10(struct suite *s)
{
    if (!s1_engine_has_swa(s->engine)) {
        printf("SKIP T10  the model has no sliding-window layers\n");
        return 0;
    }
    struct s1_engine_params params = { s->model_path, N_CTX, N_SEQ, s->cpu ? 0 : 999, true, NULL, 0 };
    struct suite            tail   = *s;
    double                  worst  = 0.0;
    tail.fixture   = s->fixture + s->n_fixture - 1; /* the long state */
    tail.n_fixture = 1;
    if (compare_engines(&tail, params, 1, "T10", &worst) != 0) {
        return -1;
    }
    return report("T10", worst, TOL_T10, "pruned against full sliding-window cache, max |d p|");
}

/* T6: the two-call path against an independent implementation, Hugging Face transformers
 * in 32-bit on the CPU, run on the same token ids by scripts/reference_logits.py. */
static int test_t6(struct suite *s)
{
    if (!s->reference) {
        printf("SKIP T6  no --reference given; the acceptance check always gives one\n");
        return 0;
    }
    if (strcmp(s->reference, "cpu") == 0) {
        return test_t6_cpu(s);
    }
    yyjson_read_err err;
    yyjson_doc     *doc = yyjson_read_file(s->reference, 0, NULL, &err);
    if (!doc) {
        fprintf(stderr, "s1-selftest: cannot read reference %s: %s\n", s->reference, err.msg);
        return -1;
    }
    yyjson_val *items = yyjson_obj_get(yyjson_doc_get_root(doc), "items");
    double      worst = 0.0;
    int         rc    = yyjson_arr_size(items) == T6_FIXTURES * T6_QUESTIONS ? 0 : -1;
    if (rc != 0) {
        fprintf(stderr, "s1-selftest: reference has %zu items, expected %d\n",
                yyjson_arr_size(items), T6_FIXTURES * T6_QUESTIONS);
    }
    for (int m = 0; rc == 0 && m < T6_FIXTURES; m++) {
        const struct fixture *f = t6_fixture(s, m);
        rc                      = run_branches(s, f, s->out_a);
        for (int j = 0; rc == 0 && j < T6_QUESTIONS; j++) {
            yyjson_val *item = yyjson_arr_get(items, (size_t)(m * T6_QUESTIONS + j));
            rc               = reference_gap(s, f, j, item, &s->out_a[j], &worst);
        }
    }
    yyjson_doc_free(doc);
    return rc != 0 ? -1 : report("T6", worst, TOL_T6, "independent reference, max |d logp|");
}

/* A small deterministic generator for synthetic records: uniform in [-1, 1). */
static double next_uniform(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

#define T7_RECORDS 200
#define T7_EMBD    24
#define T7_PROBES  50
#define T7_STEP    1e-4

/* Largest relative error between the analytic gradient and central differences, over
 * T7_PROBES randomly chosen parameters of a fit at a random point. */
static double gradient_error(struct s1_fit *fit, uint64_t *rng)
{
    int     n = s1_head_n_param(fit->h2, T7_EMBD);
    double *x = xmalloc((size_t)n * sizeof *x);
    double *g = xmalloc((size_t)n * sizeof *g);
    double *scratch = xmalloc((size_t)n * sizeof *scratch);
    for (int i = 0; i < n; i++) {
        x[i] = 0.3 * next_uniform(rng);
    }
    s1_head_loss(x, g, n, fit);
    double worst = 0.0;
    for (int probe = 0; probe < T7_PROBES; probe++) {
        int    i    = (int)((next_uniform(rng) + 1.0) / 2.0 * n) % n;
        double kept = x[i];
        x[i]        = kept + T7_STEP;
        double up   = s1_head_loss(x, scratch, n, fit);
        x[i]        = kept - T7_STEP;
        double down = s1_head_loss(x, scratch, n, fit);
        x[i]        = kept;
        double numeric = (up - down) / (2.0 * T7_STEP);
        worst = fmax(worst, fabs(g[i] - numeric) / fmax(1e-8, fabs(g[i]) + fabs(numeric)));
    }
    free(x);
    free(g);
    free(scratch);
    return worst;
}

/* T7: analytic against numerical gradients for H1 and H2 on synthetic records with soft
 * targets, content-free logits, a penalty and, in a second pass, record weights, so that every
 * term of the loss is exercised.
 * Also requires that L-BFGS drives the H1 gradient to zero on the same records. */
static int test_t7(struct suite *s)
{
    (void)s;
    uint64_t rng = 20260922;
    float   *h      = xmalloc(T7_RECORDS * T7_EMBD * sizeof *h);
    float   *z      = xmalloc(T7_RECORDS * S1_K_MAX * sizeof *z);
    float   *zc     = xmalloc(T7_RECORDS * S1_K_MAX * sizeof *zc);
    float   *target = xmalloc(T7_RECORDS * S1_K_MAX * sizeof *target);
    uint8_t *K      = xmalloc(T7_RECORDS * sizeof *K);
    float   *weight = xmalloc(T7_RECORDS * sizeof *weight);
    for (int r = 0; r < T7_RECORDS; r++) {
        K[r]      = (uint8_t)(2 + r % 9);
        weight[r] = (float)(1.25 + next_uniform(&rng));
        float sum = 0.0f;
        for (int k = 0; k < S1_K_MAX; k++) {
            z[r * S1_K_MAX + k]      = (float)(4.0 * next_uniform(&rng));
            zc[r * S1_K_MAX + k]     = (float)(2.0 * next_uniform(&rng));
            target[r * S1_K_MAX + k] = k < K[r] ? (float)(next_uniform(&rng) + 1.0) : 0.0f;
            sum += target[r * S1_K_MAX + k];
        }
        for (int k = 0; k < S1_K_MAX; k++) {
            target[r * S1_K_MAX + k] /= sum;
        }
        for (int i = 0; i < T7_EMBD; i++) {
            h[r * T7_EMBD + i] = (float)next_uniform(&rng);
        }
    }
    struct s1_records data     = { T7_RECORDS, T7_EMBD, h, z, zc, target, K, NULL };
    struct s1_records weighted = { T7_RECORDS, T7_EMBD, h, z, zc, target, K, weight };
    struct s1_fit     h1       = { &data, false, true, 0.0, false };
    struct s1_fit     h2       = { &data, true, true, 1e-2, false };
    struct s1_fit     h1w      = { &weighted, false, true, 0.0, false };
    struct s1_fit     h2w      = { &weighted, true, true, 1e-2, false };
    double            worst    = fmax(gradient_error(&h1, &rng), gradient_error(&h2, &rng));
    worst = fmax(worst, fmax(gradient_error(&h1w, &rng), gradient_error(&h2w, &rng)));

    double                 x[S1_HEAD_D] = { 0.0 };
    struct s1_lbfgs_opts   opts         = { 500, 1e-5, NULL, NULL };
    struct s1_lbfgs_result fitted       = { 0 };
    int rc = s1_lbfgs(s1_head_loss, &h1, x, S1_HEAD_D, &opts, &fitted);
    fprintf(stderr, "s1-selftest: T7 L-BFGS on H1: %d iterations, loss %.6f, |g|inf %.2e\n",
            fitted.n_iter, fitted.f, fitted.g_inf);
    free(h);
    free(z);
    free(zc);
    free(target);
    free(K);
    free(weight);
    if (rc != 0 || !fitted.converged) {
        fprintf(stderr, "s1-selftest: T7 L-BFGS did not converge\n");
        return rc != 0 ? -1 : report("T7", INFINITY, TOL_T7, "optimiser did not converge");
    }
    return report("T7", worst, TOL_T7, "gradient check H1 and H2, max relative error");
}

#define T8_ITEMS 24

/* Reads the first T8_ITEMS examples of the pipeline fixture. Returns how many were read. */
static int t8_examples(struct s1_example *x)
{
    FILE  *in   = fopen("tests/fixtures/pipeline.jsonl", "r");
    char  *text = NULL;
    size_t cap  = 0;
    int    n    = 0;
    while (in && n < T8_ITEMS && getline(&text, &cap, in) > 0) {
        if (s1_example_parse(&x[n], text, n + 1) != 0) {
            s1_example_free(&x[n]);
            break;
        }
        n++;
    }
    free(text);
    if (in) {
        fclose(in);
    }
    return n;
}

/* T8: the training path against the serving path. The same items go through packed feature
 * extraction and through the live two-call engine, both followed by the same H2 head, whose
 * parameters are pseudo-random so that z, zc and h all matter. */
static int test_t8(struct suite *s)
{
    struct s1_example     x[T8_ITEMS];
    const char           *state[T8_ITEMS];
    struct s1_ask         ask[T8_ITEMS];
    struct s1_reply       cached[T8_ITEMS];
    struct s1_decide_opts opts = { .rotations = true, .content_free = true };
    struct s1_head        head = { 0 };
    struct s1_readouts    r    = { 0 };
    struct s1_timing      timing;
    uint64_t              rng   = 8;
    double                worst = 0.0;
    int                   n     = t8_examples(x);
    int rc = n == T8_ITEMS ? s1_head_init(&head, true, s1_engine_n_embd(s->engine)) : -1;
    if (n != T8_ITEMS) {
        fprintf(stderr, "s1-selftest: T8 needs %d examples in tests/fixtures/pipeline.jsonl\n",
                T8_ITEMS);
    }
    for (int type = 0; rc == 0 && type < 3; type++) {
        int n_param = s1_head_n_param(true, head.n_embd);
        for (int i = 0; i < n_param; i++) {
            head.x[type][i] = (i < S1_HEAD_D ? 0.3 : 0.01) * next_uniform(&rng);
        }
    }
    for (int i = 0; i < n; i++) {
        state[i] = x[i].state;
        ask[i]   = x[i].ask;
    }
    if (rc == 0) {
        rc = s1_read_packed(s->engine, s->tpl, s->slot, state, ask, n, opts, &r, &timing);
    }
    if (rc == 0) {
        s1_combine(ask, n, &r, &head, cached);
    }
    for (int i = 0; rc == 0 && i < n; i++) {
        struct s1_reply live;
        rc = s1_decide(s->engine, s->tpl, s->slot, state[i], &ask[i], 1, opts, &head, &live,
                       &timing);
        for (int k = 0; rc == 0 && k < ask[i].K; k++) {
            worst = fmax(worst, fabs(live.p[k] - cached[i].p[k]));
        }
    }
    s1_readouts_free(&r);
    s1_head_free(&head);
    for (int i = 0; i < n; i++) {
        s1_example_free(&x[i]);
    }
    return rc != 0 ? -1 : report("T8", worst, TOL_T8, "cached features against live engine, max |d p|");
}

/* Violations in one answered T9 request: a question not read in as many orders as the
 * engine's rotation rule gives, a rotation whose top option is not the expected one, or
 * probabilities that do not sum to 1 within 1e-5. */
static int t9_violations(const struct s1_request *req, const struct s1_reply *reply,
                         struct s1_decide_opts opts)
{
    yyjson_val *expect = yyjson_obj_get(yyjson_doc_get_root(req->doc), "expect");
    int         bad    = 0;
    for (int i = 0; i < req->n_ask; i++) {
        const struct s1_ask *a    = &req->ask[i];
        const char          *name = yyjson_get_str(yyjson_obj_get(expect, a->name));
        int                  want = name ? s1_ask_option_index(a, name) : -1;
        double               sum  = 0.0;
        for (int k = 0; k < a->K; k++) {
            sum += reply[i].p[k];
        }
        bad += want < 0 || reply[i].n_rotation != s1_n_rotations(a->type, a->K, opts) ||
               fabs(sum - 1.0) > 1e-5;
        for (int r = 0; r < reply[i].n_rotation; r++) {
            if (reply[i].top[r] != want) {
                fprintf(stderr, "s1-selftest: T9 \"%s\" rotation %d: top option %d, expected %d\n",
                        a->name, r, reply[i].top[r], want);
                bad++;
            }
        }
    }
    return bad;
}

/* T9: for states that give the answer outright, the top slot of every rotation maps back to
 * the same option, and the averaged probabilities sum to 1. */
static int test_t9(struct suite *s)
{
    size_t len = 0;
    FILE  *f   = fopen(s->t9_path, "rb");
    if (!f) {
        fprintf(stderr, "s1-selftest: cannot read %s\n", s->t9_path);
        return -1;
    }
    char *text = xmalloc(TEXT_CAP);
    len        = fread(text, 1, TEXT_CAP - 8, f);
    fclose(f);
    memset(text + len, 0, 8);

    struct s1_decide_opts opts = { .rotations = true, .content_free = false };
    int                   bad  = 0;
    int                   rc   = 0;
    for (size_t at = 0; rc == 0 && at < len;) {
        struct s1_request req;
        struct s1_reply   reply[8];
        struct s1_timing  timing;
        size_t            used = 0;
        rc = s1_request_parse(&req, text + at, len - at, &used);
        if (rc == 0 && req.n_ask <= 8) {
            rc = s1_decide(s->engine, s->tpl, s->slot, req.state, req.ask, req.n_ask, opts, NULL,
                           reply, &timing);
            bad += rc == 0 ? t9_violations(&req, reply, opts) : 0;
        }
        s1_request_free(&req);
        at += used;
        while (at < len && strchr(" \t\r\n", text[at])) {
            at++;
        }
    }
    free(text);
    return rc != 0 ? -1 : report("T9", bad, TOL_T9, "rotation bookkeeping, violations");
}

/* Releases everything the suite owns. Safe on a suite that was only partly set up. */
static void suite_free(struct suite *s)
{
    for (int i = 0; i < s->n_fixture; i++) {
        free(s->fixture[i].prefix);
        for (int j = 0; j < Q_PER_STATE; j++) {
            free((void *)s->fixture[i].question[j].q.option);
        }
    }
    for (int j = 0; j < Q_PER_STATE; j++) {
        free(s->out_a[j].h);
        free(s->out_a[j].z);
        free(s->out_b[j].h);
        free(s->out_b[j].z);
    }
    free(s->fixture);
    free(s->joined);
    s1_rows_free(&s->rows);
    s1_engine_free(s->engine); /* on every path: Metal aborts at exit with resources held */
    yyjson_doc_free(s->doc);
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--model", "--template", "--fixtures", "--reference", "--dump-tokens", NULL };
    static const char *const flags[]   = { "--cpu", NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *model    = s1_arg_value(argc, argv, "--model");
    const char *tpl_path = s1_arg_value(argc, argv, "--template");
    const char *fixtures = s1_arg_value(argc, argv, "--fixtures");
    const char *dump     = s1_arg_value(argc, argv, "--dump-tokens");
    if (!model || !tpl_path) {
        fprintf(stderr, "usage: s1-selftest --model FILE.gguf --template FILE.tpl\n"
                        "         [--fixtures FILE.json] [--reference FILE.json|cpu] [--cpu]\n"
                        "       --reference cpu runs T6 as CPU against Metal, for a model too large\n"
                        "       to run in 32-bit\n"
                        "       --cpu keeps every layer off the GPU\n"
                        "       s1-selftest ... --dump-tokens FILE.json   (input to "
                        "scripts/reference_logits.py; runs no tests)\n");
        return 2;
    }
    if (!fixtures) {
        fixtures = "tests/fixtures/selftest.json";
    }

    struct suite       s      = { 0 };
    struct s1_template tpl    = { 0 };
    int                status = 2; /* 0 all passed, 1 a test failed, 2 a test could not run */
    struct s1_engine_params ep = {
        .model_path = model, .n_ctx = N_CTX, .n_seq = N_SEQ,
        .n_gpu_layers = s1_arg_flag(argc, argv, "--cpu") ? 0 : 999, .swa_pruned = false
    };
    s.tpl       = &tpl;
    s.reference = s1_arg_value(argc, argv, "--reference");
    s.t9_path   = "tests/fixtures/t9.json";
    s.model_path = model;
    s.cpu        = s1_arg_flag(argc, argv, "--cpu");
    s.joined    = xmalloc((PREFIX_CAP + SUFFIX_CAP) * sizeof *s.joined);
    if (s1_engine_init(&s.engine, &ep) != 0 || s1_template_load(&tpl, tpl_path) != 0 ||
        load_fixtures(&s, &tpl, fixtures) != 0) {
        goto done;
    }
    size_t n_embd  = (size_t)s1_engine_n_embd(s.engine);
    size_t n_vocab = (size_t)s1_engine_n_vocab(s.engine);
    for (int j = 0; j < Q_PER_STATE; j++) {
        s.out_a[j] = (struct s1_out){ xmalloc(n_embd * sizeof(float)),
                                      xmalloc(n_vocab * sizeof(float)) };
        s.out_b[j] = (struct s1_out){ xmalloc(n_embd * sizeof(float)),
                                      xmalloc(n_vocab * sizeof(float)) };
    }
    if (choose_slots(&s) != 0 ||
        s1_rows_load(&s.rows, model, (int)n_embd, (int)n_vocab, s.slot) != 0) {
        goto done;
    }

    if (dump) {
        status = dump_tokens(&s, dump) == 0 ? 0 : 2;
        goto done;
    }

    int (*const test[])(struct suite *) = { test_t1, test_t2, test_t3,
                                            test_t4, test_t5, test_t6, test_t7,
                                            test_t8, test_t9, test_t10 };
    int failed = 0;
    for (size_t i = 0; i < sizeof test / sizeof test[0]; i++) {
        int rc = test[i](&s);
        if (rc < 0) {
            goto done;
        }
        failed += rc;
    }
    printf("%s: %d test(s) failed\n", failed ? "FAIL" : "PASS", failed);
    status = failed ? 1 : 0;

done:
    suite_free(&s);
    s1_template_free(&tpl);
    return status;
}
