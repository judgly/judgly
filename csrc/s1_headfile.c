/* A trained head on disk. Little-endian, field by field. Head type 1 is H1, 2 is H2, 3 a
 * per-type temperature (n_embd 0, one float64 per question type). */
#include "s1.h"

#include <stdlib.h>
#include <string.h>

#define MAGIC   "S1HEAD\0"
#define VERSION 1u
#define N_TYPES 3

static void put32(FILE *f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
}

static uint32_t get32(FILE *f)
{
    uint8_t b[4] = { 0 };
    if (fread(b, 1, 4, f) != 4) {
        return 0; /* a short file fails the checks of the caller */
    }
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static void put_double(FILE *f, double d)
{
    uint64_t bits;
    memcpy(&bits, &d, sizeof bits);
    put32(f, (uint32_t)bits);
    put32(f, (uint32_t)(bits >> 32));
}

static double get_double(FILE *f)
{
    uint64_t bits = get32(f);
    bits |= (uint64_t)get32(f) << 32;
    double d;
    memcpy(&d, &bits, sizeof d);
    return d;
}

int s1_head_init(struct s1_head *head, bool h2, int n_embd)
{
    memset(head, 0, sizeof *head);
    head->h2     = h2;
    head->n_embd = n_embd;
    for (int type = 0; type < N_TYPES; type++) {
        head->x[type] = calloc((size_t)S1_HEAD_D + (h2 ? (size_t)S1_K_MAX * (size_t)n_embd : 0),
                               sizeof *head->x[type]);
        if (!head->x[type]) {
            fprintf(stderr, "judgly: out of memory\n");
            s1_head_free(head);
            return -1;
        }
    }
    return 0;
}

void s1_head_free(struct s1_head *head)
{
    for (int type = 0; type < N_TYPES; type++) {
        free(head->x[type]);
        head->x[type] = NULL;
    }
}

int s1_head_save(const struct s1_head *head, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "judgly: cannot write head file %s\n", path);
        return -1;
    }
    fwrite(MAGIC, 1, 8, f);
    put32(f, VERSION);
    put32(f, head->temperature ? 3 : head->h2 ? 2 : 1);
    put32(f, head->temperature ? 0 : (uint32_t)head->n_embd);
    fwrite(head->gguf_sha256, 1, S1_SHA256_HEX, f);
    fwrite(head->template_sha256, 1, S1_SHA256_HEX, f);
    for (int k = 0; k < S1_K_MAX; k++) {
        put32(f, (uint32_t)head->slot[k]);
    }
    int n = head->temperature ? 1 : s1_head_n_param(head->h2, head->n_embd);
    for (int type = 0; type < N_TYPES; type++) {
        for (int i = 0; i < n; i++) {
            put_double(f, head->x[type][i]);
        }
    }
    int failed = ferror(f);
    if (fclose(f) != 0 || failed) {
        fprintf(stderr, "judgly: write error in head file %s\n", path);
        return -1;
    }
    return 0;
}

int s1_head_load(struct s1_head *head, const char *path)
{
    char  magic[8] = { 0 };
    FILE *f        = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "judgly: cannot open head file %s\n", path);
        return -1;
    }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8) != 0 || get32(f) != VERSION) {
        fprintf(stderr, "judgly: %s is not a version %u head file\n", path, VERSION);
        fclose(f);
        return -1;
    }
    uint32_t kind   = get32(f);
    uint32_t n_embd = get32(f);
    if (kind == 3 && n_embd != 0) {
        fprintf(stderr, "judgly: head file %s: a temperature head has hidden size 0, not %u\n",
                path, n_embd);
        fclose(f);
        return -1;
    }
    if (kind != 3 && (n_embd < 1 || n_embd > S1_HEAD_N_EMBD_MAX)) {
        fprintf(stderr, "judgly: head file %s: hidden size %u is outside 1 to %d\n", path, n_embd,
                S1_HEAD_N_EMBD_MAX);
        fclose(f);
        return -1;
    }
    int rc = kind == 1 || kind == 2 ? s1_head_init(head, kind == 2, (int)n_embd)
             : kind == 3            ? s1_temperature_head_init(head)
                                    : -1;
    if (rc == 0) {
        rc = fread(head->gguf_sha256, 1, S1_SHA256_HEX, f) == S1_SHA256_HEX &&
                     fread(head->template_sha256, 1, S1_SHA256_HEX, f) == S1_SHA256_HEX
                 ? 0
                 : -1;
        head->gguf_sha256[S1_SHA256_HEX - 1]     = '\0';
        head->template_sha256[S1_SHA256_HEX - 1] = '\0';
        for (int k = 0; k < S1_K_MAX; k++) {
            head->slot[k] = (int32_t)get32(f);
        }
        int n = head->temperature ? 1 : s1_head_n_param(head->h2, head->n_embd);
        for (int type = 0; type < N_TYPES; type++) {
            for (int i = 0; i < n; i++) {
                head->x[type][i] = get_double(f);
            }
        }
        if (rc != 0 || feof(f) || ferror(f)) {
            fprintf(stderr, "judgly: head file %s is truncated\n", path);
            s1_head_free(head);
            rc = -1;
        } else if (fgetc(f) != EOF) {
            fprintf(stderr, "judgly: head file %s has bytes after its end\n", path);
            s1_head_free(head);
            rc = -1;
        }
        for (int type = 0; rc == 0 && head->temperature && type < N_TYPES; type++) {
            double t = head->x[type][0]; /* written as a finite number in the bounds, or 1 */
            if (!(t == 1.0 || (t >= S1_TEMP_MIN && t <= S1_TEMP_MAX))) {
                fprintf(stderr, "judgly: head file %s: temperature %g of question type %d is "
                                "outside [%g, %g]\n", path, t, type, S1_TEMP_MIN, S1_TEMP_MAX);
                s1_head_free(head);
                rc = -1;
            }
        }
    } else {
        fprintf(stderr, "judgly: head file %s has an unknown head type\n", path);
    }
    fclose(f);
    return rc;
}

int s1_head_check(const struct s1_head *head, const char *gguf_sha256,
                  const char *template_sha256, const int32_t slot[S1_K_MAX])
{
    if (strcmp(head->gguf_sha256, gguf_sha256) != 0) {
        fprintf(stderr, "judgly: the head was trained on another model file (%s, not %s)\n",
                head->gguf_sha256, gguf_sha256);
        return -1;
    }
    if (strcmp(head->template_sha256, template_sha256) != 0) {
        fprintf(stderr, "judgly: the head was trained with another template file\n");
        return -1;
    }
    if (memcmp(head->slot, slot, sizeof head->slot) != 0) {
        fprintf(stderr, "judgly: the head was trained with another slot table\n");
        return -1;
    }
    return 0;
}
