/* s1-compare: a results table from per-item dumps.
 *
 * Each input is one row: a dump written by `s1-eval --dump-items`, named on the command line
 * as NAME=FILE.tsv. Every metric carries a bootstrap interval. Against the row named with
 * --baseline, the differences are paired: computed item by item on the items both rows
 * share, then bootstrapped over those items, which gives far tighter intervals than two
 * independent averages. Text on stdout; --tsv writes the same table as TSV. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"

#define RESAMPLES 1000
#define SEED      20260922
#define MAX_ROWS  64

struct row {
    const char     *name;
    struct s1_pred *pred;
    int             n;
    double          m[S1_N_METRICS];
    double          lo[S1_N_METRICS];
    double          hi[S1_N_METRICS];
};

static int compare_by_id(const void *a, const void *b)
{
    const struct s1_pred *x = a;
    const struct s1_pred *y = b;
    return (x->id_hash > y->id_hash) - (x->id_hash < y->id_hash);
}

/* Reads one dump into a row, sorted by item id. */
static int row_load(struct row *r, const char *spec)
{
    const char *eq = strchr(spec, '=');
    if (!eq || eq == spec) {
        fprintf(stderr, "s1-compare: expected NAME=FILE.tsv, got \"%s\"\n", spec);
        return -1;
    }
    char *name = strndup(spec, (size_t)(eq - spec));
    FILE *in   = fopen(eq + 1, "r");
    if (!name || !in) {
        fprintf(stderr, "s1-compare: cannot read %s\n", eq + 1);
        free(name);
        return -1;
    }
    r->name = name;
    int    cap  = 1024;
    char  *line = NULL;
    size_t len  = 0;
    r->pred     = malloc((size_t)cap * sizeof *r->pred);
    r->n        = 0;
    int rc      = r->pred && getline(&line, &len, in) > 0 ? 0 : -1; /* header line */
    while (rc == 0 && getline(&line, &len, in) > 0) {
        if (r->n == cap) {
            cap *= 2;
            struct s1_pred *grown = realloc(r->pred, (size_t)cap * sizeof *grown);
            if (!grown) {
                rc = -1;
                break;
            }
            r->pred = grown;
        }
        struct s1_pred *x = &r->pred[r->n];
        unsigned long long id;
        unsigned            task, family;
        int                 type, K, label, used;
        memset(x, 0, sizeof *x);
        if (sscanf(line, "%llx\t%u\t%u\t%d\t%d\t%d\t%n", &id, &task, &family, &type, &K, &label,
                   &used) != 6 || K < 2 || K > S1_K_MAX || label < 0 || label >= K) {
            rc = -1;
            break;
        }
        x->id_hash   = id;
        x->task_id   = task;
        x->family_id = (uint16_t)family;
        x->type      = type;
        x->K         = K;
        x->label     = label;
        const char *at = line + used;
        for (int k = 0; k < K && rc == 0; k++) {
            char *end;
            x->p[k] = strtod(at, &end);
            rc      = end == at ? -1 : 0;
            at      = *end == ',' ? end + 1 : end;
        }
        x->slot_mass = 1.0; /* not in the dump; the slot-mass columns come from s1-eval */
        r->n++;
    }
    if (rc != 0) {
        fprintf(stderr, "s1-compare: %s: bad line %d\n", eq + 1, r->n + 2);
    }
    free(line);
    fclose(in);
    qsort(r->pred, (size_t)r->n, sizeof *r->pred, compare_by_id);
    return rc;
}

/* Indices into a and b of the items both hold, in id order. Returns the count. */
static int shared_items(const struct row *a, const struct row *b, int *ia, int *ib)
{
    int n = 0;
    for (int i = 0, j = 0; i < a->n && j < b->n;) {
        if (a->pred[i].id_hash == b->pred[j].id_hash) {
            ia[n]   = i;
            ib[n++] = j;
            i++;
            j++;
        } else if (a->pred[i].id_hash < b->pred[j].id_hash) {
            i++;
        } else {
            j++;
        }
    }
    return n;
}

static uint64_t next_random(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 33;
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Paired difference row minus baseline on the shared items, with a bootstrap interval over
 * those items. */
static int paired(const struct row *r, const struct row *base, double d[S1_N_METRICS],
                  double lo[S1_N_METRICS], double hi[S1_N_METRICS], int *n_shared)
{
    int    *ir     = malloc((size_t)r->n * sizeof *ir);
    int    *ib     = malloc((size_t)r->n * sizeof *ib);
    int    *sr     = malloc((size_t)r->n * sizeof *sr);
    int    *sb     = malloc((size_t)r->n * sizeof *sb);
    double *draws  = malloc((size_t)RESAMPLES * S1_N_METRICS * sizeof *draws);
    double *column = malloc((size_t)RESAMPLES * sizeof *column);
    int     rc     = ir && ib && sr && sb && draws && column ? 0 : -1;
    int     n      = rc == 0 ? shared_items(r, base, ir, ib) : 0;
    *n_shared      = n;
    if (rc == 0 && n > 0) {
        double mr[S1_N_METRICS], mb[S1_N_METRICS];
        s1_metrics(r->pred, ir, n, mr);
        s1_metrics(base->pred, ib, n, mb);
        uint64_t seed = SEED;
        for (int s = 0; s < RESAMPLES; s++) {
            for (int i = 0; i < n; i++) {
                int pick = (int)(next_random(&seed) % (uint64_t)n);
                sr[i]    = ir[pick];
                sb[i]    = ib[pick];
            }
            double a[S1_N_METRICS], b[S1_N_METRICS];
            s1_metrics(r->pred, sr, n, a);
            s1_metrics(base->pred, sb, n, b);
            for (int i = 0; i < S1_N_METRICS; i++) {
                draws[(size_t)s * S1_N_METRICS + (size_t)i] = a[i] - b[i];
            }
        }
        for (int i = 0; i < S1_N_METRICS; i++) {
            d[i]     = mr[i] - mb[i];
            int kept = 0;
            for (int s = 0; s < RESAMPLES; s++) {
                double v = draws[(size_t)s * S1_N_METRICS + (size_t)i];
                if (!isnan(v)) {
                    column[kept++] = v;
                }
            }
            qsort(column, (size_t)kept, sizeof *column, compare_doubles);
            lo[i] = kept ? column[(int)(0.025 * (kept - 1))] : NAN;
            hi[i] = kept ? column[(int)(0.975 * (kept - 1))] : NAN;
        }
    }
    free(ir);
    free(ib);
    free(sr);
    free(sb);
    free(draws);
    free(column);
    return rc;
}

static const int SHOWN[] = { S1_M_ACCURACY, S1_M_LOG_LOSS, S1_M_BRIER, S1_M_ECE };
#define N_SHOWN ((int)(sizeof SHOWN / sizeof SHOWN[0]))

int main(int argc, char **argv)
{
    const char *baseline = s1_arg_value(argc, argv, "--baseline");
    const char *tsv_path = s1_arg_value(argc, argv, "--tsv");
    struct row  row[MAX_ROWS];
    int         n_row  = 0;
    int         status = 1;
    int         base   = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--baseline") == 0 || strcmp(argv[i], "--tsv") == 0) {
            i++;
        } else if (n_row == MAX_ROWS || row_load(&row[n_row], argv[i]) != 0) {
            goto done;
        } else {
            base = strcmp(row[n_row].name, baseline ? baseline : "") == 0 ? n_row : base;
            n_row++;
        }
    }
    if (n_row == 0 || !baseline || base < 0) {
        fprintf(stderr, "usage: s1-compare NAME=items.tsv [NAME=items.tsv ...] --baseline NAME "
                        "[--tsv FILE]\n");
        goto done;
    }
    FILE *tsv = tsv_path ? fopen(tsv_path, "w") : NULL;
    if (tsv_path && !tsv) {
        fprintf(stderr, "s1-compare: cannot write %s\n", tsv_path);
        goto done;
    }

    printf("%-28s %6s", "row", "items");
    for (int i = 0; i < N_SHOWN; i++) {
        printf("  %-26s", S1_METRIC_NAME[SHOWN[i]]);
    }
    printf("\n");
    if (tsv) {
        fprintf(tsv, "row\titems");
        for (int i = 0; i < N_SHOWN; i++) {
            const char *m = S1_METRIC_NAME[SHOWN[i]];
            fprintf(tsv, "\t%s\t%s_lo\t%s_hi\t%s_diff\t%s_diff_lo\t%s_diff_hi", m, m, m, m, m, m);
        }
        fprintf(tsv, "\tshared_with_baseline\n");
    }
    for (int r = 0; r < n_row; r++) {
        double d[S1_N_METRICS], dlo[S1_N_METRICS], dhi[S1_N_METRICS];
        int    n_shared = 0;
        s1_metrics(row[r].pred, NULL, row[r].n, row[r].m);
        if (s1_metrics_bootstrap(row[r].pred, row[r].n, RESAMPLES, SEED, row[r].lo, row[r].hi) != 0 ||
            paired(&row[r], &row[base], d, dlo, dhi, &n_shared) != 0) {
            goto done;
        }
        printf("%-28s %6d", row[r].name, row[r].n);
        for (int i = 0; i < N_SHOWN; i++) {
            int k = SHOWN[i];
            printf("  %.4f [%.4f, %.4f]", row[r].m[k], row[r].lo[k], row[r].hi[k]);
        }
        printf("\n%-28s %6d", r == base ? "  (baseline)" : "  vs baseline, paired", n_shared);
        for (int i = 0; i < N_SHOWN; i++) {
            int k = SHOWN[i];
            printf("  %+.4f [%+.4f, %+.4f]", d[k], dlo[k], dhi[k]);
        }
        printf("\n");
        if (tsv) {
            fprintf(tsv, "%s\t%d", row[r].name, row[r].n);
            for (int i = 0; i < N_SHOWN; i++) {
                int k = SHOWN[i];
                fprintf(tsv, "\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g", row[r].m[k], row[r].lo[k],
                        row[r].hi[k], d[k], dlo[k], dhi[k]);
            }
            fprintf(tsv, "\t%d\n", n_shared);
        }
    }
    status = tsv && fclose(tsv) != 0 ? 1 : 0;

done:
    for (int r = 0; r < n_row; r++) {
        free((void *)row[r].name);
        free(row[r].pred);
    }
    return status;
}
