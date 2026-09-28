/* The 26 slot rows of the output matrix, read straight from the GGUF file and
 * dequantised on the CPU.
 *
 * The rows are what head H2 starts from, and dot(w[k], h) against libllama's own logit
 * (test T2) is the cheapest proof that the vector, the rows and the dequantisation are
 * all the right ones. */
#include "s1.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ggml.h"
#include "gguf.h"

/* The soft-cap lives under "<architecture>.final_logit_softcapping"; 0 when absent. */
static float read_softcap(const struct gguf_context *g)
{
    int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch < 0) {
        return 0.0f;
    }
    char key[128];
    int  len = snprintf(key, sizeof key, "%s.final_logit_softcapping", gguf_get_val_str(g, arch));
    if (len < 0 || (size_t)len >= sizeof key) {
        return 0.0f;
    }
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? gguf_get_val_f32(g, id) : 0.0f;
}

/* Reads and dequantises one row per slot. `base` is the tensor's byte offset in the file. */
static int read_rows(struct s1_rows *r, int fd, enum ggml_type type, size_t base,
                     const int32_t slot[S1_K_MAX])
{
    const struct ggml_type_traits *traits    = ggml_get_type_traits(type);
    size_t                         row_bytes = ggml_row_size(type, r->n_embd);
    if (type != GGML_TYPE_F32 && !traits->to_float) {
        fprintf(stderr, "judgly: no dequantisation for tensor type %s\n", traits->type_name);
        return -1;
    }
    void *raw = malloc(row_bytes);
    if (!raw) {
        fprintf(stderr, "judgly: out of memory\n");
        return -1;
    }
    int rc = 0;
    for (int k = 0; k < S1_K_MAX && rc == 0; k++) {
        float  *w   = r->w + (size_t)k * (size_t)r->n_embd;
        off_t   at  = (off_t)(base + (size_t)slot[k] * row_bytes);
        ssize_t got = pread(fd, raw, row_bytes, at);
        if (got != (ssize_t)row_bytes) {
            fprintf(stderr, "judgly: short read of row %d: %zd of %zu bytes\n", slot[k], got,
                    row_bytes);
            rc = -1;
        } else if (type == GGML_TYPE_F32) {
            memcpy(w, raw, row_bytes);
        } else {
            traits->to_float(raw, w, r->n_embd);
        }
    }
    free(raw);
    return rc;
}

int s1_rows_load(struct s1_rows *r, const char *gguf_path, int n_embd, int n_vocab,
                 const int32_t slot[S1_K_MAX])
{
    struct gguf_init_params gp = { .no_alloc = true, .ctx = NULL };
    struct gguf_context    *g  = gguf_init_from_file(gguf_path, gp);
    if (!g) {
        fprintf(stderr, "judgly: cannot read GGUF metadata of %s\n", gguf_path);
        return -1;
    }
    int     rc     = -1;
    int     fd     = -1;
    int64_t tensor = gguf_find_tensor(g, "output.weight");
    r->tied        = tensor < 0;
    if (r->tied) {
        tensor = gguf_find_tensor(g, "token_embd.weight"); /* tied embeddings */
    }
    if (tensor < 0) {
        fprintf(stderr, "judgly: %s has neither output.weight nor token_embd.weight\n", gguf_path);
        goto done;
    }
    enum ggml_type type     = gguf_get_tensor_type(g, tensor);
    size_t         expected = ggml_row_size(type, n_embd) * (size_t)n_vocab;
    if (gguf_get_tensor_size(g, tensor) != expected) {
        fprintf(stderr, "judgly: output matrix is %zu bytes, expected %zu for %d rows of %d\n",
                gguf_get_tensor_size(g, tensor), expected, n_vocab, n_embd);
        goto done;
    }
    r->n_embd  = n_embd;
    r->softcap = read_softcap(g);
    r->w       = malloc((size_t)S1_K_MAX * (size_t)n_embd * sizeof *r->w);
    fd         = open(gguf_path, O_RDONLY);
    if (!r->w || fd < 0) {
        fprintf(stderr, "judgly: cannot %s\n", r->w ? "open the GGUF file" : "allocate the rows");
        goto done;
    }
    rc = read_rows(r, fd, type, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tensor), slot);
    fprintf(stderr, "judgly: slot rows from %s (%s), soft-cap %g\n",
            r->tied ? "token_embd.weight" : "output.weight", ggml_type_name(type),
            (double)r->softcap);

done:
    if (fd >= 0) {
        close(fd);
    }
    gguf_free(g);
    if (rc != 0) {
        s1_rows_free(r);
    }
    return rc;
}

void s1_rows_free(struct s1_rows *r)
{
    free(r->w);
    r->w = NULL;
}

double s1_rows_dot(const struct s1_rows *r, int k, const float *h)
{
    const float *w   = r->w + (size_t)k * (size_t)r->n_embd;
    double       sum = 0.0;
    for (int i = 0; i < r->n_embd; i++) {
        sum += (double)w[i] * (double)h[i];
    }
    return sum;
}

double s1_rows_logit(const struct s1_rows *r, int k, const float *h)
{
    double raw = s1_rows_dot(r, k, h);
    double cap = (double)r->softcap;
    return cap > 0.0 ? cap * tanh(raw / cap) : raw;
}
