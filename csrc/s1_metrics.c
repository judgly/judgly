/* Accuracy, proper scoring rules, calibration and their bootstrap intervals.
 * Pure: predictions in, numbers out. */
#include "s1.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define P_FLOOR 1e-300 /* log loss of an item whose correct option got probability zero */

const char *const S1_METRIC_NAME[S1_N_METRICS] = { "accuracy",  "log_loss",      "brier", "ece",
                                                   "slot_mass", "slot_mass_p05", "rps" };

static int top_option(const struct s1_pred *x)
{
    int top = 0;
    for (int k = 1; k < x->K; k++) {
        top = x->p[k] > x->p[top] ? k : top;
    }
    return top;
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The q-quantile of v[0..n), which is sorted in place. */
static double quantile(double *v, int n, double q)
{
    qsort(v, (size_t)n, sizeof *v, compare_doubles);
    double at   = q * (n - 1);
    int    low  = (int)at;
    int    high = low + 1 < n ? low + 1 : low;
    return v[low] + (at - low) * (v[high] - v[low]);
}

/* Sum over levels of the squared difference between predicted and observed cumulative
 * distributions: a proper scoring rule that respects the order of the levels. */
static double ranked_probability_score(const struct s1_pred *x)
{
    double cumulative = 0.0;
    double score      = 0.0;
    for (int k = 0; k < x->K - 1; k++) {
        cumulative += x->p[k];
        double observed = k >= x->label ? 1.0 : 0.0;
        score += (cumulative - observed) * (cumulative - observed);
    }
    return score;
}

void s1_metrics(const struct s1_pred *pred, const int *index, int n, double m[S1_N_METRICS])
{
    double  bin_conf[S1_N_BINS] = { 0.0 };
    double  bin_hit[S1_N_BINS]  = { 0.0 };
    int     bin_n[S1_N_BINS]    = { 0 };
    int     n_score             = 0;
    double *mass                = malloc((size_t)(n ? n : 1) * sizeof *mass);
    for (int i = 0; i < S1_N_METRICS; i++) {
        m[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        const struct s1_pred *x   = &pred[index ? index[i] : i];
        int                   top = top_option(x);
        int                   b   = (int)(x->p[top] * S1_N_BINS);
        b                         = b >= S1_N_BINS ? S1_N_BINS - 1 : b;
        bin_n[b]++;
        bin_conf[b] += x->p[top];
        bin_hit[b] += top == x->label;
        m[S1_M_ACCURACY] += top == x->label;
        m[S1_M_LOG_LOSS] -= log(fmax(x->p[x->label], P_FLOOR));
        for (int k = 0; k < x->K; k++) {
            double d = x->p[k] - (k == x->label);
            m[S1_M_BRIER] += d * d;
        }
        m[S1_M_SLOT_MASS] += x->slot_mass;
        if (mass) {
            mass[i] = x->slot_mass;
        }
        if (x->type == S1_SCORE) {
            m[S1_M_RPS] += ranked_probability_score(x);
            n_score++;
        }
    }
    for (int b = 0; b < S1_N_BINS; b++) { /* item-weighted mean |confidence - accuracy| */
        m[S1_M_ECE] += fabs(bin_conf[b] - bin_hit[b]);
    }
    for (int i = 0; i <= S1_M_SLOT_MASS; i++) {
        m[i] = n > 0 ? m[i] / n : NAN;
    }
    m[S1_M_SLOT_MASS_P05] = n > 0 && mass ? quantile(mass, n, 0.05) : NAN;
    m[S1_M_RPS]           = n_score > 0 ? m[S1_M_RPS] / n_score : NAN;
    free(mass);
}

static uint64_t next_random(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 33;
}

int s1_metrics_bootstrap(const struct s1_pred *pred, int n, int resamples, uint64_t seed,
                         double lo[S1_N_METRICS], double hi[S1_N_METRICS])
{
    int    *index = malloc((size_t)(n ? n : 1) * sizeof *index);
    double *draws = malloc((size_t)resamples * S1_N_METRICS * sizeof *draws);
    double *one   = malloc((size_t)resamples * sizeof *one);
    if (!index || !draws || !one) {
        fprintf(stderr, "judgly: out of memory in the bootstrap\n");
        free(index);
        free(draws);
        free(one);
        return -1;
    }
    for (int r = 0; r < resamples; r++) {
        for (int i = 0; i < n; i++) {
            index[i] = (int)(next_random(&seed) % (uint64_t)n);
        }
        s1_metrics(pred, index, n, draws + (size_t)r * S1_N_METRICS);
    }
    for (int i = 0; i < S1_N_METRICS; i++) {
        int kept = 0;
        for (int r = 0; r < resamples; r++) { /* a resample without score items has no RPS */
            double v = draws[(size_t)r * S1_N_METRICS + (size_t)i];
            if (!isnan(v)) {
                one[kept++] = v;
            }
        }
        lo[i] = kept > 0 ? quantile(one, kept, 0.025) : NAN;
        hi[i] = kept > 0 ? quantile(one, kept, 0.975) : NAN;
    }
    free(index);
    free(draws);
    free(one);
    return 0;
}

void s1_reliability(const struct s1_pred *pred, int n, struct s1_bin bin[S1_N_BINS])
{
    const double z = 1.959964; /* 97.5th percentile of the standard normal */
    memset(bin, 0, S1_N_BINS * sizeof *bin);
    for (int i = 0; i < n; i++) {
        int top = top_option(&pred[i]);
        int b   = (int)(pred[i].p[top] * S1_N_BINS);
        b       = b >= S1_N_BINS ? S1_N_BINS - 1 : b;
        bin[b].n++;
        bin[b].confidence += pred[i].p[top];
        bin[b].accuracy += top == pred[i].label;
    }
    for (int b = 0; b < S1_N_BINS; b++) {
        if (bin[b].n == 0) {
            continue;
        }
        double count = bin[b].n;
        bin[b].confidence /= count;
        bin[b].accuracy /= count;
        double p      = bin[b].accuracy;
        double centre = (p + z * z / (2 * count)) / (1 + z * z / count);
        double half   = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) /
                      (1 + z * z / count);
        bin[b].wilson_lo = centre - half;
        bin[b].wilson_hi = centre + half;
    }
}

void s1_risk_coverage(const struct s1_pred *pred, int n, double threshold, double *accuracy,
                      double *coverage)
{
    int kept = 0;
    int hit  = 0;
    for (int i = 0; i < n; i++) {
        int top = top_option(&pred[i]);
        if (pred[i].p[top] > threshold) {
            kept++;
            hit += top == pred[i].label;
        }
    }
    *accuracy = kept > 0 ? (double)hit / kept : NAN;
    *coverage = n > 0 ? (double)kept / n : NAN;
}
