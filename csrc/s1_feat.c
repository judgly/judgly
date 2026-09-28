/* The feature file: cached backbone outputs, one record per rotation of one example.
 * Written and read field by field in little-endian byte order, so the file
 * does not depend on the compiler's struct layout or on the host's byte order. */
#include "s1.h"

#include <stdlib.h>
#include <string.h>

#define MAGIC        "S1FEAT\0"
#define VERSION      1u
#define HEADER_BYTES (96 + 4 * S1_K_MAX)
#define RECORD_BYTES (8 + 4 + 2 + 5 + S1_K_MAX + 4 * (3 * S1_K_MAX + 1))
#define COUNT_OFFSET 24 /* byte offset of n_records in the header */

uint64_t s1_hash64(const char *text)
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (const unsigned char *c = (const unsigned char *)text; *c; c++) {
        hash = (hash ^ *c) * 0x100000001b3ULL;
    }
    return hash;
}

static uint8_t *put(uint8_t *at, uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; i++) {
        *at++ = (uint8_t)(value >> (8 * i));
    }
    return at;
}

static const uint8_t *get(const uint8_t *at, uint64_t *value, int bytes)
{
    *value = 0;
    for (int i = 0; i < bytes; i++) {
        *value |= (uint64_t)*at++ << (8 * i);
    }
    return at;
}

static uint8_t *put_floats(uint8_t *at, const float *f, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &f[i], sizeof bits);
        at = put(at, bits, 4);
    }
    return at;
}

static const uint8_t *get_floats(const uint8_t *at, float *f, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        at            = get(at, &bits, 4);
        uint32_t word = (uint32_t)bits;
        memcpy(&f[i], &word, sizeof word);
    }
    return at;
}

static int hex_to_bytes(const char *hex, uint8_t *bytes)
{
    for (int i = 0; i < 32; i++) {
        unsigned value;
        if (sscanf(hex + 2 * i, "%2x", &value) != 1) {
            return -1;
        }
        bytes[i] = (uint8_t)value;
    }
    return 0;
}

static void bytes_to_hex(const uint8_t *bytes, char *hex)
{
    for (int i = 0; i < 32; i++) {
        snprintf(hex + 2 * i, 3, "%02x", bytes[i]);
    }
}

int s1_feat_create(struct s1_feat_writer *w, const char *path, const struct s1_feat_header *h)
{
    uint8_t  buf[HEADER_BYTES] = { 0 };
    uint8_t *at                = buf;
    memcpy(at, MAGIC, 8);
    at = put(at + 8, VERSION, 4);
    at = put(at, h->n_embd, 4);
    at = put(at, S1_K_MAX, 4);
    at = put(at, h->flags, 4);
    at = put(at, 0, 8); /* n_records, written by s1_feat_close */
    if (hex_to_bytes(h->gguf_sha256, at) != 0 || hex_to_bytes(h->template_sha256, at + 32) != 0) {
        fprintf(stderr, "judgly: feature header needs two SHA-256 hex strings\n");
        return -1;
    }
    at += 64;
    for (int k = 0; k < S1_K_MAX; k++) {
        at = put(at, (uint32_t)h->slot[k], 4);
    }
    w->file      = fopen(path, "wb");
    w->n_embd    = h->n_embd;
    w->n_records = 0;
    if (!w->file || fwrite(buf, 1, sizeof buf, w->file) != sizeof buf) {
        fprintf(stderr, "judgly: cannot write feature file %s\n", path);
        return -1;
    }
    return 0;
}

int s1_feat_append(struct s1_feat_writer *w, const struct s1_feat_record *r, const float *h)
{
    size_t   bytes = RECORD_BYTES + 4 * (size_t)w->n_embd;
    uint8_t *buf   = malloc(bytes);
    if (!buf) {
        fprintf(stderr, "judgly: out of memory\n");
        return -1;
    }
    uint8_t *at = put(buf, r->id_hash, 8);
    at          = put(at, r->task_id, 4);
    at          = put(at, r->family_id, 2);
    at          = put(at, r->type, 1);
    at          = put(at, r->split, 1);
    at          = put(at, r->K, 1);
    at          = put(at, r->label, 1);
    at          = put(at, r->rotation, 1);
    memcpy(at, r->perm, S1_K_MAX);
    at = put_floats(at + S1_K_MAX, r->target, S1_K_MAX);
    at = put_floats(at, r->z, S1_K_MAX);
    at = put_floats(at, r->zc, S1_K_MAX);
    at = put_floats(at, &r->slot_mass, 1);
    put_floats(at, h, w->n_embd);
    int rc = fwrite(buf, 1, bytes, w->file) == bytes ? 0 : -1;
    if (rc != 0) {
        fprintf(stderr, "judgly: write error in the feature file\n");
    }
    free(buf);
    w->n_records++;
    return rc;
}

int s1_feat_close(struct s1_feat_writer *w)
{
    uint8_t count[8];
    put(count, w->n_records, 8);
    int rc = fseek(w->file, COUNT_OFFSET, SEEK_SET) == 0 &&
                     fwrite(count, 1, sizeof count, w->file) == sizeof count
                 ? 0
                 : -1;
    if (fclose(w->file) != 0 || rc != 0) {
        fprintf(stderr, "judgly: cannot finish the feature file\n");
        rc = -1;
    }
    w->file = NULL;
    return rc;
}

static int read_header(struct s1_feat_header *h, FILE *file, const char *path)
{
    uint8_t  buf[HEADER_BYTES];
    uint64_t version, k_max, value;
    if (fread(buf, 1, sizeof buf, file) != sizeof buf || memcmp(buf, MAGIC, 8) != 0) {
        fprintf(stderr, "judgly: %s is not a feature file\n", path);
        return -1;
    }
    const uint8_t *at = get(buf + 8, &version, 4);
    at                = get(at, &value, 4);
    h->n_embd         = (uint32_t)value;
    at                = get(at, &k_max, 4);
    at                = get(at, &value, 4);
    h->flags          = (uint32_t)value;
    at                = get(at, &h->n_records, 8);
    if (version != VERSION || k_max != S1_K_MAX) {
        fprintf(stderr, "judgly: %s has version %d and k_max %d, expected %u and %d\n", path,
                (int)version, (int)k_max, VERSION, S1_K_MAX);
        return -1;
    }
    bytes_to_hex(at, h->gguf_sha256);
    bytes_to_hex(at + 32, h->template_sha256);
    at += 64;
    for (int k = 0; k < S1_K_MAX; k++) {
        at         = get(at, &value, 4);
        h->slot[k] = (int32_t)(uint32_t)value;
    }
    return 0;
}

static void read_record(struct s1_feat_record *r, float *h, const uint8_t *at, size_t n_embd)
{
    uint64_t v;
    at           = get(at, &r->id_hash, 8);
    at           = get(at, &v, 4);
    r->task_id   = (uint32_t)v;
    at           = get(at, &v, 2);
    r->family_id = (uint16_t)v;
    r->type      = *at++;
    r->split     = *at++;
    r->K         = *at++;
    r->label     = *at++;
    r->rotation  = *at++;
    memcpy(r->perm, at, S1_K_MAX);
    at = get_floats(at + S1_K_MAX, r->target, S1_K_MAX);
    at = get_floats(at, r->z, S1_K_MAX);
    at = get_floats(at, r->zc, S1_K_MAX);
    at = get_floats(at, &r->slot_mass, 1);
    get_floats(at, h, n_embd);
}

int s1_feat_load(struct s1_feat *f, const char *path)
{
    memset(f, 0, sizeof *f);
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "judgly: cannot open feature file %s\n", path);
        return -1;
    }
    int      rc  = read_header(&f->header, file, path);
    size_t   n   = (size_t)f->header.n_records;
    size_t   dim = f->header.n_embd;
    size_t   bytes = RECORD_BYTES + 4 * dim;
    uint8_t *buf   = malloc(bytes);
    if (rc == 0) {
        f->rec = malloc((n ? n : 1) * sizeof *f->rec);
        f->h   = malloc((n ? n : 1) * dim * sizeof *f->h);
        if (!buf || !f->rec || !f->h) {
            fprintf(stderr, "judgly: out of memory loading %zu records from %s\n", n, path);
            rc = -1;
        }
    }
    for (size_t i = 0; rc == 0 && i < n; i++) {
        if (fread(buf, 1, bytes, file) != bytes) {
            fprintf(stderr, "judgly: %s ends after %zu of %zu records\n", path, i, n);
            rc = -1;
        } else {
            read_record(&f->rec[i], f->h + i * dim, buf, dim);
        }
    }
    free(buf);
    fclose(file);
    if (rc != 0) {
        s1_feat_free(f);
    }
    return rc;
}

void s1_feat_free(struct s1_feat *f)
{
    free(f->rec);
    free(f->h);
    memset(f, 0, sizeof *f);
}

void s1_feat_drop_letters(struct s1_feat *f)
{
    for (uint64_t i = 0; i < f->header.n_records; i++) {
        memset(f->rec[i].z, 0, sizeof f->rec[i].z);
        memset(f->rec[i].zc, 0, sizeof f->rec[i].zc);
    }
}
