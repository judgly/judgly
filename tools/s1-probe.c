/* s1-probe: four behavioural probes. They test properties of the
 * engine and the backbone that aggregate metrics cannot show.
 *
 *   isolation      a code placed in a sibling question must not be visible to another question
 *   order          spread of p(correct) across the cyclic rotations of the options
 *   irrelevant     shift of the log-odds between the two leading options when one clearly
 *                  irrelevant option is added
 *   repeatability  the same question alone, repeated, and among different sibling groups */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"

#define N_CTX       32768
#define N_SEQ       33
#define N_CODES     20
#define N_REPEATS   10
#define IRRELEVANT  "The moon is made of green cheese"

struct probe {
    struct s1_engine     *engine;
    struct s1_template    tpl;
    const struct s1_head *head; /* NULL for H0 */
    int32_t               slot[S1_K_MAX];
    struct s1_example    *x; /* choice examples of the test and held-out splits */
    int                   n;
};

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static int decide(struct probe *p, const char *state, const struct s1_ask *ask, int n,
                  bool rotations, struct s1_reply *reply)
{
    struct s1_timing      timing;
    struct s1_decide_opts opts = { rotations, false, 0 };
    return s1_decide(p->engine, &p->tpl, p->slot, state, ask, n, opts, p->head, reply, &timing);
}

/* A code sits either in a sibling question or in the state; another question asks which code
 * was mentioned. Rotations are on, so that the position of the code among the options cancels. */
static int probe_isolation(struct probe *p)
{
    double in_sibling = 0.0;
    double in_state   = 0.0;
    for (int i = 0; i < N_CODES; i++) {
        char code[16], other1[16], other2[16], sibling[128], state[160];
        snprintf(code, sizeof code, "ZX-%04d", 1000 + 397 * i);
        snprintf(other1, sizeof other1, "ZX-%04d", 1100 + 397 * i);
        snprintf(other2, sizeof other2, "QP-%04d", 1000 + 211 * i);
        snprintf(sibling, sizeof sibling, "The reference code %s has a valid format.", code);
        snprintf(state, sizeof state, "The customer wrote about a delayed parcel. "
                                      "Reference code: %s.", code);
        struct s1_ask ask[2] = {
            { .name = "which", .type = S1_CHOICE, .K = 4,
              .instructions = "Which reference code is mentioned in the text?",
              .key = { "code", "other1", "other2", "none" },
              .option = { code, other1, other2, "No reference code is mentioned" } },
            { .name = "sibling", .type = S1_BOOL, .K = 2, .instructions = sibling,
              .option = { "True", "False" } },
        };
        struct s1_reply reply[2];
        if (decide(p, "The customer wrote about a delayed parcel.", ask, 2, true, reply) != 0) {
            return -1;
        }
        in_sibling += reply[0].p[0];
        if (decide(p, state, ask, 1, true, reply) != 0) {
            return -1;
        }
        in_state += reply[0].p[0];
    }
    printf("isolation      p(code) with the code in a sibling question %.6f, in the state %.6f "
           "(%d codes; expected near 0 and high)\n", in_sibling / N_CODES, in_state / N_CODES,
           N_CODES);
    return 0;
}

/* For every item, p(correct) under each rotation; the spread is max minus min. */
static int probe_order(struct probe *p)
{
    double *spread = malloc((size_t)p->n * sizeof *spread);
    double  total  = 0.0;
    int     rc     = spread ? 0 : -1;
    for (int i = 0; i < p->n && rc == 0; i++) {
        struct s1_readouts    r;
        struct s1_timing      timing;
        struct s1_decide_opts opts = { true, false, 0 };
        const struct s1_ask  *a    = &p->x[i].ask;
        rc = s1_read(p->engine, &p->tpl, p->slot, p->x[i].state, a, 1, opts, &r, &timing);
        double low  = 1.0;
        double high = 0.0;
        for (int j = 0; rc == 0 && j < r.n; j++) {
            double q[S1_K_MAX];
            s1_readout_probs(&r.item[j], p->head, a->type, a->K, q);
            int    slot    = (p->x[i].label - r.item[j].rotation + a->K) % a->K;
            low            = fmin(low, q[slot]);
            high           = fmax(high, q[slot]);
        }
        spread[i] = high - low;
        total += spread[i];
        s1_readouts_free(&r);
    }
    if (rc == 0) {
        qsort(spread, (size_t)p->n, sizeof *spread, compare_doubles);
        printf("order          spread of p(correct) across rotations: mean %.4f, 95th percentile "
               "%.4f (%d items; smaller is better)\n", total / p->n,
               spread[(int)(0.95 * (p->n - 1))], p->n);
    }
    free(spread);
    return rc;
}

/* The two leading original options before and after one irrelevant option is appended. */
static int probe_irrelevant(struct probe *p)
{
    double total = 0.0;
    int    used  = 0;
    for (int i = 0; i < p->n; i++) {
        struct s1_ask   a = p->x[i].ask;
        struct s1_reply before, after;
        if (a.K >= S1_K_MAX) {
            continue;
        }
        if (decide(p, p->x[i].state, &a, 1, false, &before) != 0) {
            return -1;
        }
        int first = 0, second = 1;
        for (int k = 1; k < a.K; k++) {
            if (before.p[k] > before.p[first]) {
                second = first;
                first  = k;
            } else if (k != first && (second == first || before.p[k] > before.p[second])) {
                second = k;
            }
        }
        a.key[a.K]      = "irrelevant";
        a.option[a.K++] = IRRELEVANT;
        if (decide(p, p->x[i].state, &a, 1, false, &after) != 0) {
            return -1;
        }
        double shift = log(after.p[first] / after.p[second]) - log(before.p[first] / before.p[second]);
        if (isfinite(shift)) {
            total += fabs(shift);
            used++;
        }
    }
    printf("irrelevant     mean |shift| of the log-odds between the two leading options %.4f "
           "(%d items; 0 under independent scoring)\n", used ? total / used : NAN, used);
    return 0;
}

/* The first question of each item, asked N_REPEATS times alone, then among the questions of
 * other items as siblings. */
static int probe_repeatability(struct probe *p)
{
    double alone   = 0.0;
    double grouped = 0.0;
    int    n_items = p->n < 8 ? p->n : 8;
    for (int i = 0; i < n_items; i++) {
        struct s1_reply first;
        struct s1_reply reply[4];
        if (decide(p, p->x[i].state, &p->x[i].ask, 1, false, &first) != 0) {
            return -1;
        }
        for (int repeat = 1; repeat < N_REPEATS; repeat++) {
            struct s1_ask group[4] = { p->x[i].ask };
            int           n        = 1 + repeat % 4;
            for (int j = 1; j < n; j++) {
                group[j] = p->x[(i + repeat + j) % p->n].ask;
            }
            if (decide(p, p->x[i].state, group, 1, false, reply) != 0) {
                return -1;
            }
            for (int k = 0; k < p->x[i].ask.K; k++) {
                alone = fmax(alone, fabs(reply[0].p[k] - first.p[k]));
            }
            if (decide(p, p->x[i].state, group, n, false, reply) != 0) {
                return -1;
            }
            for (int k = 0; k < p->x[i].ask.K; k++) {
                grouped = fmax(grouped, fabs(reply[0].p[k] - first.p[k]));
            }
        }
    }
    printf("repeatability  max |d p| when repeated alone %.6f, among different siblings %.6f "
           "(%d questions, %d repeats)\n", alone, grouped, n_items, N_REPEATS);
    return 0;
}

/* The choice examples of the test and held-out splits, at most `limit`. */
static int load_examples(struct probe *p, const char *path, long limit)
{
    FILE  *in   = fopen(path, "r");
    char  *text = NULL;
    size_t cap  = 0;
    long   line = 0;
    int    rc   = in ? 0 : -1;
    p->x        = calloc((size_t)limit, sizeof *p->x);
    if (!in || !p->x) {
        fprintf(stderr, "s1-probe: cannot read %s\n", path);
        rc = -1;
    }
    while (rc == 0 && p->n < limit && getline(&text, &cap, in) > 0) {
        struct s1_example x;
        rc = s1_example_parse(&x, text, ++line);
        if (rc == 0 && x.ask.type == S1_CHOICE && x.split >= S1_TEST) {
            p->x[p->n++] = x;
        } else {
            s1_example_free(&x);
        }
    }
    free(text);
    if (in) {
        fclose(in);
    }
    if (rc == 0 && p->n < 2) {
        fprintf(stderr, "s1-probe: %s has fewer than 2 choice examples in test or heldout\n", path);
        rc = -1;
    }
    return rc;
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--model", "--template", "--head", "--examples",
                                           "--items", NULL };
    static const char *const flags[]   = { "--plain-slots", NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *model     = s1_arg_value(argc, argv, "--model");
    const char *tpl_path  = s1_arg_value(argc, argv, "--template");
    const char *head_path = s1_arg_value(argc, argv, "--head");
    const char *examples  = s1_arg_value(argc, argv, "--examples");
    const char *items     = s1_arg_value(argc, argv, "--items");
    if (!model) {
        model = getenv("S1_MODEL");
    }
    if (!model || !tpl_path || !examples) {
        fprintf(stderr, "usage: s1-probe --model FILE.gguf --template FILE.tpl --examples "
                        "FILE.jsonl\n         [--head HEAD.bin] [--items N] [--plain-slots]\n");
        return 2;
    }
    struct probe            p    = { 0 };
    struct s1_head          head = { 0 };
    struct s1_engine_params ep   = { model, N_CTX, N_SEQ, 999, false, NULL, 0 };
    char                    sha[S1_SHA256_HEX];
    char                    tpl_sha[S1_SHA256_HEX];
    int                     status = 1;
    if (load_examples(&p, examples, items ? strtol(items, NULL, 10) : 200) == 0 &&
        s1_engine_init(&p.engine, &ep) == 0 && s1_template_load(&p.tpl, tpl_path) == 0 &&
        s1_slots_init(p.engine, !s1_arg_flag(argc, argv, "--plain-slots"), p.slot) == 0 &&
        (!head_path || (s1_head_load(&head, head_path) == 0 && s1_sha256_file(model, sha) == 0 &&
                        s1_sha256_file(tpl_path, tpl_sha) == 0 &&
                        s1_head_check(&head, sha, tpl_sha, p.slot) == 0))) {
        p.head = head_path ? &head : NULL;
        if (head_path && head.temperature) {
            fprintf(stderr, "s1-probe: a temperature head acts on the mean over rotations, not on "
                            "one readout; probe without --head\n");
        } else {
            printf("probes for %s on %d items\n", head_path ? head_path : "H0", p.n);
            status = probe_isolation(&p) == 0 && probe_order(&p) == 0 &&
                             probe_irrelevant(&p) == 0 && probe_repeatability(&p) == 0
                         ? 0
                         : 1;
        }
    }
    for (int i = 0; i < p.n; i++) {
        s1_example_free(&p.x[i]);
    }
    free(p.x);
    s1_head_free(&head);
    s1_template_free(&p.tpl);
    s1_engine_free(p.engine);
    return status;
}
