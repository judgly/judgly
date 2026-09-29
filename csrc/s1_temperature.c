/* The per-type temperature, applied after the mean over rotations, and the per-item
 * predictions of a feature file that both s1-eval and the temperature fit read.
 *
 * Pure apart from allocation: no model, no files. */
#include "s1.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int s1_temperature_head_init(struct s1_head *head)
{
    memset(head, 0, sizeof *head);
    head->temperature = true;
    for (int type = 0; type < 3; type++) {
        head->x[type] = malloc(sizeof *head->x[type]);
        if (!head->x[type]) {
            fprintf(stderr, "judgly: out of memory\n");
            s1_head_free(head);
            return -1;
        }
        head->x[type][0] = 1.0;
    }
    return 0;
}

/* log p[k], floored (and capped at 0, for a sum that rounds above 1). */
static double floored_log(double p)
{
    return log(fmin(fmax(p, S1_TEMP_FLOOR), 1.0));
}

void s1_temperature_apply(double temperature, int K, double *p)
{
    if (temperature == 1.0) {
        return; /* the identity */
    }
    double u[S1_K_MAX];
    double umax = -INFINITY;
    for (int k = 0; k < K; k++) {
        u[k] = floored_log(p[k]) / temperature;
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

double s1_temperature_loss(double temperature, const struct s1_pred *pred, const int *index,
                           int n)
{
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        const struct s1_pred *x = &pred[index ? index[i] : i];
        double                p[S1_K_MAX];
        memcpy(p, x->p, sizeof p);
        s1_temperature_apply(temperature, x->K, p);
        loss -= floored_log(p[x->label]);
    }
    return n > 0 ? loss / n : 0.0;
}

/* Derivative of the mean loss with respect to beta = 1/T:
 *   d/d beta [-beta L_y + log sum_k exp(beta L_k)] = sum_k q_k L_k - L_y,   q = softmax(beta L),
 * with L_k = log p_k (floored). Its own derivative, the variance of L under q, is not negative,
 * so the loss is convex in beta and the derivative increases with beta. */
static double slope(double beta, const struct s1_pred *pred, const int *index, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const struct s1_pred *x = &pred[index ? index[i] : i];
        double                L[S1_K_MAX];
        double                umax = -INFINITY;
        for (int k = 0; k < x->K; k++) {
            L[k] = floored_log(x->p[k]);
            umax = fmax(umax, beta * L[k]);
        }
        double z = 0.0, m = 0.0;
        for (int k = 0; k < x->K; k++) {
            double w = exp(beta * L[k] - umax);
            z += w;
            m += w * L[k];
        }
        sum += m / z - L[x->label];
    }
    return n > 0 ? sum / n : 0.0;
}

double s1_temperature_fit(const struct s1_pred *pred, const int *index, int n, bool *bounded)
{
    double lo = 1.0 / S1_TEMP_MAX; /* beta = 1/T */
    double hi = 1.0 / S1_TEMP_MIN;
    *bounded  = false;
    if (n == 0) {
        return 1.0;
    }
    if (slope(lo, pred, index, n) >= 0.0) {
        *bounded = true;
        return S1_TEMP_MAX;
    }
    if (slope(hi, pred, index, n) <= 0.0) {
        *bounded = true;
        return S1_TEMP_MIN;
    }
    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) {
            break; /* adjacent doubles: machine precision */
        }
        if (slope(mid, pred, index, n) < 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 1.0 / (0.5 * (lo + hi));
}

/* ---- per-item predictions of a feature file ------------------------------------------------ */

/* Probabilities of one record over its slots: the head's, or softmax(z - zc) for H0 and for a
 * temperature head, which acts only after the mean over rotations. */
static void record_probs(const struct s1_feat *f, size_t i, const struct s1_head *head,
                         bool content_free, double *p)
{
    static const float           no_zc[S1_K_MAX] = { 0.0f };
    const struct s1_feat_record *r               = &f->rec[i];
    const float                 *zc              = content_free ? r->zc : no_zc;
    struct s1_head               h0              = { 0 };
    double                       identity[S1_HEAD_D] = { 0.0 };
    if (!head || head->temperature) { /* H0: the identity head, c = 1 lets zc in */
        identity[S1_HEAD_C] = 1.0;
        h0.x[r->type]       = identity;
        head                = &h0;
    }
    s1_head_apply(head->x[r->type], head->h2, (int)f->header.n_embd, r->z, zc,
                  f->h + i * f->header.n_embd, r->K, p);
}

static int compare_by_id(const void *a, const void *b)
{
    const struct s1_pred *x = a;
    const struct s1_pred *y = b;
    return (x->id_hash > y->id_hash) - (x->id_hash < y->id_hash);
}

int s1_feat_predictions(const struct s1_feat *f, int split, const struct s1_head *head,
                        bool rotations, bool content_free, struct s1_pred **out)
{
    size_t          n_rec = (size_t)f->header.n_records;
    struct s1_pred *pred  = calloc(n_rec ? n_rec : 1, sizeof *pred);
    if (!pred) {
        fprintf(stderr, "judgly: out of memory\n");
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
        if (head && head->temperature) {
            s1_temperature_apply(head->x[item.type][0], item.K, item.p);
        }
        pred[n_item++] = item;
        i              = j;
    }
    *out = pred;
    return n_item;
}
