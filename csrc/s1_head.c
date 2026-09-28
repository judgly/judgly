/* Heads H1 and H2: forward pass, loss and analytic gradient.
 *
 *   u[k] = a * (z[k] + dot(d[k], h)) - c * zc[k] + b[k],   a = exp(theta),   p = softmax(u)
 *   L    = -sum_k t[k] * log(p[k]),                        dL/du[k] = p[k] - t[k]
 *
 * The mean over records is weighted when the records carry weights: sum_r w[r] L[r] / sum_r w[r].
 *
 * Sums run in double although the records are float. Pure: no model, no
 * files; test T7 checks every gradient here against central differences. */
#include "s1.h"

#include <math.h>
#include <string.h>

int s1_head_n_param(bool h2, int n_embd) { return S1_HEAD_D + (h2 ? S1_K_MAX * n_embd : 0); }

static double dot_df(const double *d, const float *h, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += d[i] * (double)h[i];
    }
    return sum;
}

/* s[k] = z[k] + dot(d[k], h): the slot logit with the row correction, before scaling. */
static void corrected_logits(const double *x, bool h2, int n_embd, const float *z, const float *h,
                             int K, double *s)
{
    for (int k = 0; k < K; k++) {
        s[k] = (double)z[k];
        if (h2) {
            s[k] += dot_df(x + S1_HEAD_D + (size_t)k * (size_t)n_embd, h, n_embd);
        }
    }
}

/* p = softmax(u) and logp = log(p), the latter computed without taking log of a p that may
 * have underflowed to zero. */
static void softmax(const double *u, int K, double *p, double *logp)
{
    double umax = u[0];
    for (int k = 1; k < K; k++) {
        umax = fmax(umax, u[k]);
    }
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += exp(u[k] - umax);
    }
    double lse = umax + log(sum);
    for (int k = 0; k < K; k++) {
        logp[k] = u[k] - lse;
        p[k]    = exp(logp[k]);
    }
}

static void probabilities(const double *x, const double *s, const float *zc, int K, double *p,
                          double *logp)
{
    double a = exp(x[S1_HEAD_THETA]);
    double u[S1_K_MAX];
    for (int k = 0; k < K; k++) {
        u[k] = a * s[k] - x[S1_HEAD_C] * (double)zc[k] + x[S1_HEAD_B + k];
    }
    softmax(u, K, p, logp);
}

void s1_head_apply(const double *x, bool h2, int n_embd, const float *z, const float *zc,
                   const float *h, int K, double *p)
{
    double s[S1_K_MAX];
    double logp[S1_K_MAX];
    corrected_logits(x, h2, n_embd, z, h, K, s);
    probabilities(x, s, zc, K, p, logp);
}

double s1_head_loss(const double *x, double *g, int n, void *ctx)
{
    const struct s1_fit     *fit    = ctx;
    const struct s1_records *data   = fit->data;
    int                      n_embd = data->n_embd;
    double                   a      = exp(x[S1_HEAD_THETA]);
    double                   loss   = 0.0;
    double                   w_sum  = 0.0;
    memset(g, 0, (size_t)n * sizeof *g);

    for (int r = 0; r < data->n; r++) {
        const float *z  = data->z + (size_t)r * S1_K_MAX;
        const float *zc = data->zc + (size_t)r * S1_K_MAX;
        const float *t  = data->target + (size_t)r * S1_K_MAX;
        const float *h  = data->h + (size_t)r * (size_t)n_embd;
        int          K  = data->K[r];
        double       w  = data->weight ? (double)data->weight[r] : 1.0;
        double       s[S1_K_MAX];
        double       p[S1_K_MAX];
        double       logp[S1_K_MAX];
        w_sum += w;
        corrected_logits(x, fit->h2, n_embd, z, h, K, s);
        probabilities(x, s, zc, K, p, logp);
        for (int k = 0; k < K; k++) {
            loss -= w * (double)t[k] * logp[k];
            double gu = w * (p[k] - (double)t[k]); /* w * dL/du[k] */
            g[S1_HEAD_THETA] += gu * a * s[k]; /* chain rule through a = exp(theta) */
            g[S1_HEAD_C] -= gu * (double)zc[k];
            g[S1_HEAD_B + k] += gu;
            if (fit->h2) {
                double *gd = g + S1_HEAD_D + (size_t)k * (size_t)n_embd;
                for (int i = 0; i < n_embd; i++) {
                    gd[i] += gu * a * (double)h[i];
                }
            }
        }
    }

    double scale = w_sum > 0.0 ? 1.0 / w_sum : 0.0;
    loss *= scale;
    for (int i = 0; i < n; i++) {
        g[i] *= scale;
    }
    if (!fit->fit_c) {
        g[S1_HEAD_C] = 0.0;
    }
    if (fit->fix_theta) {
        g[S1_HEAD_THETA] = 0.0;
    }
    for (int i = S1_HEAD_D; i < n; i++) { /* the penalty lambda * sum(d * d) */
        loss += fit->lambda * x[i] * x[i];
        g[i] += 2.0 * fit->lambda * x[i];
    }
    return loss;
}

double s1_head_mean_loss(const double *x, bool h2, const struct s1_records *data)
{
    double loss  = 0.0;
    double w_sum = 0.0;
    for (int r = 0; r < data->n; r++) {
        const float *t = data->target + (size_t)r * S1_K_MAX;
        int          K = data->K[r];
        double       w = data->weight ? (double)data->weight[r] : 1.0;
        double       s[S1_K_MAX];
        double       p[S1_K_MAX];
        double       logp[S1_K_MAX];
        corrected_logits(x, h2, data->n_embd, data->z + (size_t)r * S1_K_MAX,
                         data->h + (size_t)r * (size_t)data->n_embd, K, s);
        probabilities(x, s, data->zc + (size_t)r * S1_K_MAX, K, p, logp);
        for (int k = 0; k < K; k++) {
            loss -= w * (double)t[k] * logp[k];
        }
        w_sum += w;
    }
    return w_sum > 0.0 ? loss / w_sum : 0.0;
}

double s1_theta_clamp(double theta)
{
    double lo = -log(S1_TEMP_MAX); /* temperature exp(-theta) at most S1_TEMP_MAX */
    double hi = -log(S1_TEMP_MIN);
    return theta < lo ? lo : theta > hi ? hi : theta;
}

void s1_head_h2_start(double *x)
{
    double lo = -log(S1_H2_START_TEMP);
    if (x[S1_HEAD_THETA] < lo) {
        x[S1_HEAD_THETA] = lo;
    }
}

double s1_head_prob_sd(const double *x, bool h2, const struct s1_records *data)
{
    double mean[S1_K_MAX] = { 0.0 }; /* Welford: running mean and sum of squared deviations */
    double m2[S1_K_MAX]   = { 0.0 };
    int    n[S1_K_MAX]    = { 0 };
    for (int r = 0; r < data->n; r++) {
        double p[S1_K_MAX];
        int    K = data->K[r];
        s1_head_apply(x, h2, data->n_embd, data->z + (size_t)r * S1_K_MAX,
                      data->zc + (size_t)r * S1_K_MAX, data->h + (size_t)r * (size_t)data->n_embd,
                      K, p);
        for (int k = 0; k < K; k++) {
            double delta = p[k] - mean[k];
            n[k]++;
            mean[k] += delta / n[k];
            m2[k] += delta * (p[k] - mean[k]);
        }
    }
    double worst = 0.0;
    for (int k = 0; k < S1_K_MAX; k++) {
        if (n[k] > 1) {
            worst = fmax(worst, sqrt(m2[k] / n[k]));
        }
    }
    return worst;
}
