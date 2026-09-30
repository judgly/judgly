/* s1-train: fits head H1 or H2, or the per-type temperature, on a feature file.
 *
 * The temperature (--head temperature): per question type, the one temperature T that minimises
 * the mean log loss of the training items, each item's probabilities being the raw readout
 * (softmax(z - zc) per rotation, mapped back to the options, averaged over the rotations the
 * engine settings read) with T applied to the average (s1_temperature_apply). Every item counts
 * once, unweighted. T is rounded to three decimals, the precision of the values confirmed on
 * untouched data (docs/calibration.md); the unrounded value is recorded in the sidecar. The
 * validation split is used only for the verdict below, as for H2.
 *
 * Per question type: H1 from the identity by L-BFGS; for H2, from the fitted H1 for each
 * lambda of the grid, stopping early on the validation loss, keeping the best lambda. The
 * test and held-out splits are never read here.
 *
 * Score records are weighted so that every level carries the same total weight, in training and
 * in validation. A level means something different in every score question (level 1 is "not
 * toxic" in one, "one star" in another), so the head must not learn how often each level was
 * the answer in the fit data: unweighted, its per-level bias takes the level frequencies of the
 * fit data and pulls every score question towards the most common level there. Choice and bool
 * need no weights: their options are rotated through the slots. Score heads stay H1 (no row
 * corrections) also when H2 is asked for: the score fit data are few, and the weights raise
 * the rare levels to the weight of the common ones, which the n_embd-wide row corrections of
 * H2 overfit. */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"

#define MAX_ITER 500
#define G_TOL    1e-5
#define PATIENCE 20 /* H2: iterations without a better validation loss before stopping */

static const double      LAMBDA[]    = { 1e-4, 1e-3, 1e-2, 1e-1 };
static const char *const TYPE_NAME[] = { "choice", "bool", "score" };

/* The records of one question type and one split, copied into parallel arrays. */
struct block {
    struct s1_records view;
    float            *h;
    float            *z;
    float            *zc;
    float            *target;
    uint8_t          *K;
    float            *weight; /* score only: level-balanced weights; NULL otherwise */
};

/* Marks the training items kept under --limit-train: the same number from every task in
 * turn, each task's items in the order of their id hash, so that a smaller limit is a subset
 * of a larger one. Records of one item (its rotations) are kept or dropped together. */
struct item_key {
    uint64_t id_hash;
    uint32_t task_id;
};

static int compare_items(const void *a, const void *b)
{
    const struct item_key *x = a;
    const struct item_key *y = b;
    if (x->task_id != y->task_id) {
        return x->task_id < y->task_id ? -1 : 1;
    }
    return (x->id_hash > y->id_hash) - (x->id_hash < y->id_hash);
}

static int compare_hash(const void *a, const void *b)
{
    const struct item_key *x = a;
    const struct item_key *y = b;
    return (x->id_hash > y->id_hash) - (x->id_hash < y->id_hash);
}

/* Writes into `keep` (one flag per record) which training records survive the limit. */
static int mark_kept(const struct s1_feat *f, long limit, bool *keep)
{
    size_t           n_rec = (size_t)f->header.n_records;
    struct item_key *items = malloc((n_rec ? n_rec : 1) * sizeof *items);
    size_t           n     = 0;
    if (!items) {
        return -1;
    }
    for (size_t i = 0; i < n_rec; i++) {
        keep[i] = f->rec[i].split != S1_TRAIN;
        if (f->rec[i].split == S1_TRAIN && f->rec[i].rotation == 0) {
            items[n++] = (struct item_key){ f->rec[i].id_hash, f->rec[i].task_id };
        }
    }
    qsort(items, n, sizeof *items, compare_items); /* grouped by task, hash order within */
    /* round-robin over tasks: the j-th item of every task before the (j+1)-th of any */
    struct item_key *chosen = malloc((n ? n : 1) * sizeof *chosen);
    size_t           n_chosen = 0;
    if (!chosen) {
        free(items);
        return -1;
    }
    for (size_t round = 0; n_chosen < (size_t)limit && n_chosen < n; round++) {
        size_t before = n_chosen;
        for (size_t i = 0; i < n && n_chosen < (size_t)limit; i++) {
            size_t rank = 0; /* position of item i within its task */
            for (size_t k = i; k > 0 && items[k - 1].task_id == items[i].task_id; k--) {
                rank++;
            }
            if (rank == round) {
                chosen[n_chosen++] = items[i];
            }
        }
        if (n_chosen == before) {
            break;
        }
    }
    qsort(chosen, n_chosen, sizeof *chosen, compare_hash);
    for (size_t i = 0; i < n_rec; i++) {
        if (f->rec[i].split == S1_TRAIN) {
            struct item_key key = { f->rec[i].id_hash, f->rec[i].task_id };
            keep[i] = bsearch(&key, chosen, n_chosen, sizeof key, compare_hash) != NULL;
        }
    }
    free(chosen);
    free(items);
    return 0;
}

/* The level of a score record: the slot its target puts the most weight on. */
static int record_level(const float *target, int K)
{
    int best = 0;
    for (int k = 1; k < K; k++) {
        if (target[k] > target[best]) {
            best = k;
        }
    }
    return best;
}

/* Weights w = n / (L * n_level), L the number of levels that occur: every level then has total
 * weight n / L, and the weights sum to n. */
static int balance_levels(struct block *b)
{
    int    n                = b->view.n;
    size_t count[S1_K_MAX]  = { 0 };
    int    n_levels         = 0;
    b->weight               = malloc((size_t)n * sizeof *b->weight);
    if (!b->weight) {
        fprintf(stderr, "s1-train: out of memory for %d weights\n", n);
        return -1;
    }
    for (int r = 0; r < n; r++) {
        count[record_level(b->target + (size_t)r * S1_K_MAX, b->K[r])]++;
    }
    for (int k = 0; k < S1_K_MAX; k++) {
        n_levels += count[k] > 0;
    }
    for (int r = 0; r < n; r++) {
        size_t c     = count[record_level(b->target + (size_t)r * S1_K_MAX, b->K[r])];
        b->weight[r] = (float)((double)n / ((double)n_levels * (double)c));
    }
    b->view.weight = b->weight;
    return 0;
}

static int block_init(struct block *b, const struct s1_feat *f, int type, int split,
                      const bool *keep)
{
    size_t n_embd = f->header.n_embd;
    size_t n      = 0;
    for (uint64_t i = 0; i < f->header.n_records; i++) {
        n += f->rec[i].type == type && f->rec[i].split == split && keep[i];
    }
    memset(b, 0, sizeof *b);
    size_t room = n ? n : 1;
    b->h        = malloc(room * n_embd * sizeof *b->h);
    b->z        = malloc(room * S1_K_MAX * sizeof *b->z);
    b->zc       = malloc(room * S1_K_MAX * sizeof *b->zc);
    b->target   = malloc(room * S1_K_MAX * sizeof *b->target);
    b->K        = malloc(room);
    if (!b->h || !b->z || !b->zc || !b->target || !b->K) {
        fprintf(stderr, "s1-train: out of memory for %zu records\n", n);
        return -1;
    }
    size_t at = 0;
    for (uint64_t i = 0; i < f->header.n_records; i++) {
        const struct s1_feat_record *r = &f->rec[i];
        if (r->type != type || r->split != split || !keep[i]) {
            continue;
        }
        memcpy(b->h + at * n_embd, f->h + i * n_embd, n_embd * sizeof *b->h);
        memcpy(b->z + at * S1_K_MAX, r->z, sizeof r->z);
        memcpy(b->zc + at * S1_K_MAX, r->zc, sizeof r->zc);
        memcpy(b->target + at * S1_K_MAX, r->target, sizeof r->target);
        b->K[at++] = r->K;
    }
    b->view = (struct s1_records){ (int)n, (int)n_embd, b->h, b->z, b->zc, b->target, b->K, NULL };
    if (type == S1_SCORE && n > 0) {
        return balance_levels(b);
    }
    return 0;
}

static void block_free(struct block *b)
{
    free(b->h);
    free(b->z);
    free(b->zc);
    free(b->target);
    free(b->K);
    free(b->weight);
}

/* Early stopping for H2: remembers the parameters with the lowest validation loss. */
struct watch {
    const struct s1_records *validation;
    double                  *best_x;
    double                   best_loss;
    int                      n;
    int                      since_best;
};

static int watch_validation(const double *x, int iter, double f, void *ctx)
{
    struct watch *w    = ctx;
    double        loss = s1_head_mean_loss(x, true, w->validation);
    printf("    iter %3d  train %.6f  validation %.6f\n", iter, f, loss);
    if (loss < w->best_loss) {
        w->best_loss  = loss;
        w->since_best = 0;
        memcpy(w->best_x, x, (size_t)w->n * sizeof *x);
    } else {
        w->since_best++;
    }
    return w->since_best >= PATIENCE;
}

static int print_h1(const double *x, int iter, double f, void *ctx)
{
    printf("    iter %3d  train %.6f  validation %.6f\n", iter, f,
           s1_head_mean_loss(x, false, ctx));
    return 0;
}

/* Fits H1 into x[0 .. S1_HEAD_D). Returns the validation loss through *loss and whether the
 * temperature had to be held at a bound through *bounded. */
static int fit_h1(double *x, const struct s1_records *train, const struct s1_records *validation,
                  bool fit_c, double *loss, bool *bounded)
{
    struct s1_fit          fit  = { train, false, fit_c, 0.0, false };
    struct s1_lbfgs_opts   opts = { MAX_ITER, G_TOL, print_h1, (void *)validation };
    struct s1_lbfgs_result res;
    double                 start[S1_HEAD_D];
    memcpy(start, x, sizeof start);
    if (s1_lbfgs(s1_head_loss, &fit, x, S1_HEAD_D, &opts, &res) != 0) {
        return -1;
    }
    *bounded = s1_theta_clamp(x[S1_HEAD_THETA]) != x[S1_HEAD_THETA];
    if (*bounded) { /* refit with the temperature held at the bound it crossed */
        printf("  H1: temperature %.4g is outside [%g, %g]; refitting with it held at %g\n",
               exp(-x[S1_HEAD_THETA]), S1_TEMP_MIN, S1_TEMP_MAX,
               exp(-s1_theta_clamp(x[S1_HEAD_THETA])));
        double theta = s1_theta_clamp(x[S1_HEAD_THETA]);
        memcpy(x, start, sizeof start);
        x[S1_HEAD_THETA] = theta;
        fit.fix_theta    = true;
        if (s1_lbfgs(s1_head_loss, &fit, x, S1_HEAD_D, &opts, &res) != 0) {
            return -1;
        }
    }
    *loss = s1_head_mean_loss(x, false, validation);
    printf("  H1: %d iterations, train %.6f, validation %.6f, temperature %.4f, c %.4f%s\n",
           res.n_iter, res.f, *loss, exp(-x[S1_HEAD_THETA]), x[S1_HEAD_C],
           res.converged ? "" : " (not converged)");
    return 0;
}

/* One H2 fit from `start` at one lambda; the best parameters by validation loss land in
 * watch->best_x. */
static int fit_h2_once(double *trial, const double *start, int n, const struct s1_records *train,
                       bool fit_c, double lambda, bool fix_theta, struct watch *watch)
{
    struct s1_fit          fit  = { train, true, fit_c, lambda, fix_theta };
    struct s1_lbfgs_opts   opts = { MAX_ITER, G_TOL, watch_validation, watch };
    struct s1_lbfgs_result res;
    memcpy(trial, start, (size_t)n * sizeof *trial);
    return s1_lbfgs(s1_head_loss, &fit, trial, n, &opts, &res);
}

/* Fits H2 for every lambda from the H1 solution in x (its temperature capped by
 * s1_head_h2_start), and leaves the best in x. A fit whose best temperature leaves the bounds
 * is redone with the temperature held at the bound. */
static int fit_h2(double *x, int n, const struct s1_records *train,
                  const struct s1_records *validation, bool fit_c, bool fix_theta, double *loss,
                  double *lambda)
{
    double *start = malloc((size_t)n * sizeof *start);
    double *trial = malloc((size_t)n * sizeof *trial);
    double *best  = malloc((size_t)n * sizeof *best);
    int     rc    = start && trial && best ? 0 : -1;
    if (rc == 0) {
        memcpy(start, x, (size_t)n * sizeof *x);
        if (!fix_theta) {
            s1_head_h2_start(start);
        }
        if (start[S1_HEAD_THETA] != x[S1_HEAD_THETA]) {
            printf("  H2 starts at temperature %g, not H1's %.4g\n", exp(-start[S1_HEAD_THETA]),
                   exp(-x[S1_HEAD_THETA]));
        }
    }
    for (size_t l = 0; rc == 0 && l < sizeof LAMBDA / sizeof LAMBDA[0]; l++) {
        struct watch watch = { validation, best, *loss, n, 0 };
        printf("  H2 lambda %g\n", LAMBDA[l]);
        rc = fit_h2_once(trial, start, n, train, fit_c, LAMBDA[l], fix_theta, &watch);
        if (rc == 0 && watch.best_loss < *loss &&
            s1_theta_clamp(best[S1_HEAD_THETA]) != best[S1_HEAD_THETA]) {
            double theta = s1_theta_clamp(best[S1_HEAD_THETA]);
            printf("  H2 lambda %g: temperature %.4g is outside [%g, %g]; refitting with it held "
                   "at %g\n", LAMBDA[l], exp(-best[S1_HEAD_THETA]), S1_TEMP_MIN, S1_TEMP_MAX,
                   exp(-theta));
            double kept = start[S1_HEAD_THETA];
            start[S1_HEAD_THETA] = theta;
            watch = (struct watch){ validation, best, *loss, n, 0 };
            rc = fit_h2_once(trial, start, n, train, fit_c, LAMBDA[l], true, &watch);
            start[S1_HEAD_THETA] = kept;
        }
        if (rc == 0 && watch.best_loss < *loss) {
            *loss   = watch.best_loss;
            *lambda = LAMBDA[l];
            memcpy(x, best, (size_t)n * sizeof *x);
        }
    }
    if (rc != 0) {
        fprintf(stderr, "s1-train: H2 fit failed\n");
    }
    free(start);
    free(trial);
    free(best);
    return rc;
}

static int train_type(struct s1_head *head, const struct s1_feat *f, int type, FILE *report,
                      int *n_reported, const bool *keep)
{
    struct block train      = { 0 };
    struct block validation = { 0 };
    int          rc         = block_init(&train, f, type, S1_TRAIN, keep);
    rc              = rc == 0 ? block_init(&validation, f, type, S1_VALIDATION, keep) : rc;
    if (rc == 0 && (train.view.n == 0 || validation.view.n == 0)) {
        printf("%s: %d training and %d validation records; the head stays the identity\n",
               TYPE_NAME[type], train.view.n, validation.view.n);
        fprintf(report, "%s  \"%s\": {\"train\": %d, \"validation\": %d, \"fallback\": \"identity\", "
                        "\"reason\": \"no records of this type\"}",
                (*n_reported)++ ? ",\n" : "", TYPE_NAME[type], train.view.n, validation.view.n);
    } else if (rc == 0) {
        bool   fit_c   = (f->header.flags & S1_FEAT_HAS_ZC) != 0;
        double h0      = s1_head_mean_loss(head->x[type], false, &validation.view);
        double loss    = 0.0;
        double lambda  = 0.0;
        bool   bounded = false;
        printf("%s: %d training and %d validation records, H0 validation %.6f\n", TYPE_NAME[type],
               train.view.n, validation.view.n, h0);
        rc = fit_h1(head->x[type], &train.view, &validation.view, fit_c, &loss, &bounded);
        if (rc == 0 && head->h2 && type != S1_SCORE) {
            rc = fit_h2(head->x[type], s1_head_n_param(true, head->n_embd), &train.view,
                        &validation.view, fit_c, bounded, &loss, &lambda);
            printf("  H2: validation %.6f at lambda %g\n", loss, lambda);
        }
        /* The verdict: a head that ignores its input, or does no better than the raw readout
         * on validation, is replaced by the identity (H0) for this type. */
        double      sd          = s1_head_prob_sd(head->x[type], head->h2, &validation.view);
        double      temperature = exp(-head->x[type][S1_HEAD_THETA]);
        const char *reason      = sd < S1_MIN_PROB_SD ? "input-independent on validation"
                                  : !(loss < h0)      ? "no better than the raw readout on validation"
                                                      : NULL;
        if (reason) {
            memset(head->x[type], 0, (size_t)s1_head_n_param(head->h2, head->n_embd) * sizeof(double));
            printf("  %s: %s (probability sd %.3g, loss %.6f vs raw %.6f); the head is the identity\n",
                   TYPE_NAME[type], reason, sd, loss, h0);
        } else {
            printf("  %s: probability sd %.4f on validation, temperature %.4f\n", TYPE_NAME[type],
                   sd, temperature);
        }
        fprintf(report, "%s  \"%s\": {\"train\": %d, \"validation\": %d, \"h0_validation_loss\": "
                        "%.6f, \"validation_loss\": %.6f, \"lambda\": %g, \"temperature\": %.6g, "
                        "\"temperature_bounded\": %s, \"prob_sd\": %.6g, \"fallback\": %s%s%s, "
                        "\"reason\": %s%s%s}",
                (*n_reported)++ ? ",\n" : "", TYPE_NAME[type], train.view.n, validation.view.n, h0,
                loss, lambda, temperature, bounded ? "true" : "false", sd,
                reason ? "\"" : "", reason ? "identity" : "null", reason ? "\"" : "",
                reason ? "\"" : "", reason ? reason : "null", reason ? "\"" : "");
    }
    block_free(&train);
    block_free(&validation);
    return rc;
}

/* Largest, over option positions k, standard deviation of p_T[k] across the items (those with
 * K > k): the temperature form of s1_head_prob_sd. */
static double temperature_prob_sd(double t, const struct s1_pred *pred, const int *index, int n)
{
    double mean[S1_K_MAX] = { 0.0 };
    double m2[S1_K_MAX]   = { 0.0 };
    int    count[S1_K_MAX] = { 0 };
    for (int i = 0; i < n; i++) {
        const struct s1_pred *x = &pred[index[i]];
        double                p[S1_K_MAX];
        memcpy(p, x->p, sizeof p);
        s1_temperature_apply(t, x->K, p);
        for (int k = 0; k < x->K; k++) {
            double delta = p[k] - mean[k];
            count[k]++;
            mean[k] += delta / count[k];
            m2[k] += delta * (p[k] - mean[k]);
        }
    }
    double worst = 0.0;
    for (int k = 0; k < S1_K_MAX; k++) {
        if (count[k] > 1) {
            worst = fmax(worst, sqrt(m2[k] / count[k]));
        }
    }
    return worst;
}

/* The positions in pred[0 .. n) of the items of one question type. Returns their number. */
static int items_of_type(const struct s1_pred *pred, int n, int type, int *index)
{
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (pred[i].type == type) {
            index[m++] = i;
        }
    }
    return m;
}

/* Fits the temperature of every question type into head (a temperature head) and writes the
 * per-type entries of the sidecar. */
static int fit_temperatures(struct s1_head *head, const struct s1_feat *f,
                            struct s1_decide_opts engine, FILE *report)
{
    struct s1_pred *train  = NULL;
    struct s1_pred *valid  = NULL;
    int             n_t    = s1_feat_predictions(f, S1_TRAIN, NULL, engine.rotations,
                                                 engine.content_free, &train);
    int             n_v    = n_t < 0 ? -1 : s1_feat_predictions(f, S1_VALIDATION, NULL,
                                                                engine.rotations,
                                                                engine.content_free, &valid);
    int            *it     = n_v < 0 ? NULL : malloc((size_t)(n_t + 1) * sizeof *it);
    int            *iv     = it ? malloc((size_t)(n_v + 1) * sizeof *iv) : NULL;
    int             rc     = iv ? 0 : -1;
    int             n_done = 0;
    if (rc != 0) {
        fprintf(stderr, "s1-train: cannot read the items of the feature file\n");
    }
    for (int type = 0; rc == 0 && type < 3; type++) {
        int m_t = items_of_type(train, n_t, type, it);
        int m_v = items_of_type(valid, n_v, type, iv);
        fprintf(report, "%s  \"%s\": ", n_done++ ? ",\n" : "", TYPE_NAME[type]);
        if (m_t == 0 || m_v == 0) {
            /* A fit needs training items, and its verdict validation items. */
            const char *why = m_t == 0 && m_v == 0 ? "no items of this type"
                              : m_t == 0           ? "no training items of this type"
                                                   : "no validation items of this type";
            printf("%s: %d training and %d validation items; the temperature stays 1 (identity)\n",
                   TYPE_NAME[type], m_t, m_v);
            fprintf(report, "{\"train_items\": %d, \"validation_items\": %d, \"temperature\": 1, "
                            "\"fallback\": \"identity\", \"reason\": \"%s\"}",
                    m_t, m_v, why);
            continue;
        }
        bool   bounded = false;
        double exact   = s1_temperature_fit(train, it, m_t, &bounded);
        double t       = round(exact * 1000.0) / 1000.0; /* three decimals */
        t              = fmin(fmax(t, S1_TEMP_MIN), S1_TEMP_MAX);
        double train_loss = s1_temperature_loss(t, train, it, m_t);
        double h0         = s1_temperature_loss(1.0, valid, iv, m_v);
        double loss       = s1_temperature_loss(t, valid, iv, m_v);
        double sd         = temperature_prob_sd(t, valid, iv, m_v);
        const char *reason = sd < S1_MIN_PROB_SD ? "input-independent on validation"
                             : !(loss < h0)      ? "no better than the raw readout on validation"
                                                 : NULL;
        printf("%s: %d training and %d validation items, raw validation %.6f\n"
               "  temperature %.3f (unrounded %.9g%s), train %.6f, validation %.6f, "
               "probability sd %.4f\n",
               TYPE_NAME[type], m_t, m_v, h0, t, exact, bounded ? ", at a bound" : "", train_loss,
               loss, sd);
        if (reason) {
            printf("  %s: %s; the temperature is 1 (identity)\n", TYPE_NAME[type], reason);
        }
        head->x[type][0] = reason ? 1.0 : t;
        fprintf(report, "{\"train_items\": %d, \"validation_items\": %d, \"h0_validation_loss\": "
                        "%.6f, \"train_loss\": %.6f, \"validation_loss\": %.6f, \"temperature\": "
                        "%.3f, \"temperature_unrounded\": %.9g, \"temperature_bounded\": %s, "
                        "\"prob_sd\": %.6g, \"fallback\": %s%s%s, \"reason\": %s%s%s}",
                m_t, m_v, h0, train_loss, loss, reason ? 1.0 : t, exact,
                bounded ? "true" : "false", sd, reason ? "\"" : "", reason ? "identity" : "null",
                reason ? "\"" : "", reason ? "\"" : "", reason ? reason : "null",
                reason ? "\"" : "");
    }
    free(it);
    free(iv);
    free(train);
    free(valid);
    return rc;
}

static int compare_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* The feature file must have been extracted with the engine settings the head is declared
 * for: content-free logits present exactly when content_free is set, and every item read in
 * exactly s1_n_rotations orders. */
static int check_engine(const struct s1_feat *f, struct s1_decide_opts opts)
{
    if (((f->header.flags & S1_FEAT_HAS_ZC) != 0) != opts.content_free) {
        fprintf(stderr, "s1-train: the features %s content-free logits, but --content-free is %s\n",
                (f->header.flags & S1_FEAT_HAS_ZC) ? "carry" : "carry no",
                opts.content_free ? "set" : "not set");
        return -1;
    }
    size_t    n   = (size_t)f->header.n_records;
    uint64_t *ids = malloc((n ? n : 1) * sizeof *ids);
    if (!ids) {
        fprintf(stderr, "s1-train: out of memory\n");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        ids[i] = f->rec[i].id_hash;
    }
    qsort(ids, n, sizeof *ids, compare_u64);
    int rc = 0;
    for (size_t i = 0; i < n && rc == 0; i++) {
        const struct s1_feat_record *r = &f->rec[i];
        size_t lo = 0, hi = n; /* the run of r's id in the sorted ids */
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (ids[mid] < r->id_hash) lo = mid + 1; else hi = mid;
        }
        size_t end = lo;
        while (end < n && ids[end] == r->id_hash) {
            end++;
        }
        int expected = s1_n_rotations((enum s1_type)r->type, r->K, opts);
        if ((int)(end - lo) != expected) {
            fprintf(stderr, "s1-train: an item of type %s with %d options has %d records, but the "
                            "engine settings read it in %d orders; extract the features with the "
                            "same settings\n", TYPE_NAME[r->type], r->K, (int)(end - lo), expected);
            rc = -1;
        }
    }
    free(ids);
    return rc;
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--features", "--head", "--out", "--limit-train",
                                           "--max-rotations", NULL };
    static const char *const flags[]   = { "--probe", "--rotations", "--content-free", NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *features = s1_arg_value(argc, argv, "--features");
    const char *kind     = s1_arg_value(argc, argv, "--head");
    const char *out      = s1_arg_value(argc, argv, "--out");
    const char *limit    = s1_arg_value(argc, argv, "--limit-train");
    bool        h2       = kind && strcmp(kind, "h2") == 0;
    bool        temp     = kind && strcmp(kind, "temperature") == 0;
    const char *max_rot  = s1_arg_value(argc, argv, "--max-rotations");
    struct s1_decide_opts engine = { s1_arg_flag(argc, argv, "--rotations"),
                                     s1_arg_flag(argc, argv, "--content-free"),
                                     max_rot ? (int)strtol(max_rot, NULL, 10) : 0 };
    if (!features || !out || !kind || (!h2 && !temp && strcmp(kind, "h1") != 0) ||
        (temp && (limit || s1_arg_flag(argc, argv, "--probe")))) {
        fprintf(stderr, "usage: s1-train --features FILE.feat --head h1|h2|temperature --out HEAD.bin "
                        "[--limit-train N] [--probe]\n"
                        "         [--rotations] [--max-rotations N] [--content-free]\n"
                        "       The engine settings the features were extracted with, and that the\n"
                        "       head is served under; checked against the features, recorded in\n"
                        "       HEAD.bin.json.\n"
                        "       --probe fits on h alone: the slot logits are set to zero.\n"
                        "       --limit-train keeps N training items, evenly across tasks, nested.\n"
                        "       --head temperature fits one temperature per question type on the\n"
                        "       rotation-averaged raw readout (no --limit-train, no --probe).\n"
                        "       The gradient check is test T7 of s1-selftest.\n");
        return 2;
    }

    struct s1_feat f      = { 0 };
    struct s1_head head   = { 0 };
    char           json[4096];
    char           sha[S1_SHA256_HEX];
    FILE          *report = NULL;
    bool          *keep   = NULL;
    int            status = 1;
    if (s1_feat_load(&f, features) == 0 && check_engine(&f, engine) == 0 &&
        (!s1_arg_flag(argc, argv, "--probe") || (s1_feat_drop_letters(&f), true)) &&
        s1_sha256_file(features, sha) == 0 &&
        (keep = malloc((size_t)(f.header.n_records ? f.header.n_records : 1) * sizeof *keep)) &&
        mark_kept(&f, limit ? strtol(limit, NULL, 10) : LONG_MAX, keep) == 0 &&
        (temp ? s1_temperature_head_init(&head) : s1_head_init(&head, h2, (int)f.header.n_embd)) == 0 &&
        snprintf(json, sizeof json, "%s.json", out) < (int)sizeof json &&
        (report = fopen(json, "w")) != NULL) {
        memcpy(head.gguf_sha256, f.header.gguf_sha256, sizeof head.gguf_sha256);
        memcpy(head.template_sha256, f.header.template_sha256, sizeof head.template_sha256);
        memcpy(head.slot, f.header.slot, sizeof head.slot);
        fprintf(report, "{\"head\": \"%s\", \"features\": \"%s\", \"features_sha256\": \"%s\",\n"
                        " \"limit_train\": %ld,\n"
                        " \"engine\": {\"rotations\": %s, \"max_rotations\": %d, \"content_free\": %s},\n"
                        " \"types\": {\n", kind, features, sha,
                limit ? strtol(limit, NULL, 10) : -1L, engine.rotations ? "true" : "false",
                engine.max_rotations, engine.content_free ? "true" : "false");
        int n_reported = 0;
        status         = 0;
        if (temp) {
            status = fit_temperatures(&head, &f, engine, report) == 0 ? 0 : 1;
        }
        for (int type = 0; !temp && type < 3 && status == 0; type++) {
            status = train_type(&head, &f, type, report, &n_reported, keep) == 0 ? 0 : 1;
        }
        fprintf(report, "\n }}\n");
        if (status == 0 && s1_head_save(&head, out) != 0) {
            status = 1;
        }
    }
    if (report) {
        fclose(report);
    }
    free(keep);
    s1_head_free(&head);
    s1_feat_free(&f);
    return status;
}
