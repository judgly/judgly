/* Full-batch L-BFGS with Armijo backtracking.
 *
 * Nocedal and Wright, Numerical Optimization, algorithm 7.4 (two-loop recursion). The
 * direction is -H g, where H approximates the inverse Hessian from the last M pairs
 * s = x' - x and y = g' - g. No dependencies: arrays in, arrays out. */
#include "s1.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define M           10   /* correction pairs kept */
#define ARMIJO      1e-4 /* sufficient-decrease constant */
#define MAX_HALVING 40   /* backtracking steps before the search is declared failed */

static double dot(const double *a, const double *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static double norm_inf(const double *a, int n)
{
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        worst = fmax(worst, fabs(a[i]));
    }
    return worst;
}

/* dir = -H g by the two-loop recursion over the `count` newest pairs, oldest at `first`. */
static void direction(const double *g, double *dir, int n, double *const *s, double *const *y,
                      const double *rho, int first, int count)
{
    double alpha[M];
    memcpy(dir, g, (size_t)n * sizeof *dir);
    for (int j = count - 1; j >= 0; j--) {
        int i    = (first + j) % M;
        alpha[j] = rho[i] * dot(s[i], dir, n);
        for (int k = 0; k < n; k++) {
            dir[k] -= alpha[j] * y[i][k];
        }
    }
    if (count > 0) {
        int    newest = (first + count - 1) % M;
        double scale  = 1.0 / (rho[newest] * dot(y[newest], y[newest], n)); /* s.y / y.y */
        for (int k = 0; k < n; k++) {
            dir[k] *= scale;
        }
    }
    for (int j = 0; j < count; j++) {
        int    i    = (first + j) % M;
        double beta = rho[i] * dot(y[i], dir, n);
        for (int k = 0; k < n; k++) {
            dir[k] += (alpha[j] - beta) * s[i][k];
        }
    }
    for (int k = 0; k < n; k++) {
        dir[k] = -dir[k];
    }
}

/* Backtracks from `step` until f has decreased sufficiently along dir. On success x_new,
 * g_new and *f_new hold the accepted point. */
static bool line_search(s1_objective f, void *ctx, const double *x, double fx, const double *g,
                        const double *dir, int n, double step, double *x_new, double *g_new,
                        double *f_new)
{
    double slope = dot(g, dir, n);
    for (int halving = 0; slope < 0.0 && halving < MAX_HALVING; halving++, step *= 0.5) {
        for (int k = 0; k < n; k++) {
            x_new[k] = x[k] + step * dir[k];
        }
        *f_new = f(x_new, g_new, n, ctx);
        if (isfinite(*f_new) && *f_new <= fx + ARMIJO * step * slope) {
            return true;
        }
    }
    return false;
}

int s1_lbfgs(s1_objective f, void *ctx, double *x, int n, const struct s1_lbfgs_opts *opts,
             struct s1_lbfgs_result *result)
{
    /* one block: s[M], y[M], g, g_new, x_new, dir */
    double *block = malloc((size_t)(2 * M + 4) * (size_t)n * sizeof *block);
    if (!block) {
        fprintf(stderr, "judgly: out of memory in s1_lbfgs (%d parameters)\n", n);
        return -1;
    }
    double *s[M];
    double *y[M];
    for (int i = 0; i < M; i++) {
        s[i] = block + (size_t)(2 * i) * (size_t)n;
        y[i] = block + (size_t)(2 * i + 1) * (size_t)n;
    }
    double *g     = block + (size_t)(2 * M) * (size_t)n;
    double *g_new = g + n;
    double *x_new = g_new + n;
    double *dir   = x_new + n;
    double rho[M];
    int    first = 0;
    int    count = 0;

    double fx = f(x, g, n, ctx);
    int    iter;
    bool   stopped = false;
    for (iter = 0; iter < opts->max_iter && norm_inf(g, n) >= opts->g_tol && !stopped; iter++) {
        direction(g, dir, n, s, y, rho, first, count);
        double f_new;
        double step = count == 0 ? 1.0 / fmax(1.0, norm_inf(g, n)) : 1.0;
        if (!line_search(f, ctx, x, fx, g, dir, n, step, x_new, g_new, &f_new)) {
            break;
        }
        int slot = (first + count) % M;
        if (count == M) {
            first = (first + 1) % M; /* the oldest pair is overwritten */
        } else {
            count++;
        }
        for (int k = 0; k < n; k++) {
            s[slot][k] = x_new[k] - x[k];
            y[slot][k] = g_new[k] - g[k];
        }
        double sy = dot(s[slot], y[slot], n);
        if (sy > 1e-12) {
            rho[slot] = 1.0 / sy;
        } else {
            count = 0; /* curvature condition failed: restart from steepest descent */
            first = 0;
        }
        memcpy(x, x_new, (size_t)n * sizeof *x);
        memcpy(g, g_new, (size_t)n * sizeof *g);
        fx      = f_new;
        stopped = opts->progress && opts->progress(x, iter + 1, fx, opts->progress_ctx) != 0;
    }
    *result = (struct s1_lbfgs_result){ iter, fx, norm_inf(g, n), norm_inf(g, n) < opts->g_tol };
    free(block);
    return 0;
}
