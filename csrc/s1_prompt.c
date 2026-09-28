/* Template file, prompt packing and the slot table.
 *
 * The prompt is defined as a concatenation of token arrays, one per piece, because
 * tokenising the joined string can merge characters across a piece boundary. */
#include "s1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yyjson.h"

#define S1_SEP "\n\n" /* between the state and the question body */

static char *copy_json_string(yyjson_val *obj, const char *key, const char *path)
{
    const char *s = yyjson_get_str(yyjson_obj_get(obj, key));
    if (!s) {
        fprintf(stderr, "judgly: template %s has no string \"%s\"\n", path, key);
        return NULL;
    }
    char *copy = strdup(s);
    if (!copy) {
        fprintf(stderr, "judgly: out of memory\n");
    }
    return copy;
}

int s1_template_load(struct s1_template *t, const char *path)
{
    yyjson_read_err err;
    yyjson_doc     *doc = yyjson_read_file(path, 0, NULL, &err);
    if (!doc) {
        fprintf(stderr, "judgly: cannot read template %s: %s\n", path, err.msg);
        return -1;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    t->open  = copy_json_string(root, "open", path);
    t->close = copy_json_string(root, "close", path);
    yyjson_doc_free(doc);
    if (!t->open || !t->close) {
        s1_template_free(t);
        return -1;
    }
    return 0;
}

void s1_template_free(struct s1_template *t)
{
    free(t->open);
    free(t->close);
    t->open  = NULL;
    t->close = NULL;
}

/* Tokenises one piece at tok[*n] and advances *n. */
static int add_piece(const struct s1_engine *e, const char *text, size_t len, bool parse_special,
                     int32_t *tok, int cap, int *n)
{
    int got = s1_tokenize(e, text, (int)len, parse_special, tok + *n, cap - *n);
    if (got < 0) {
        return -1;
    }
    *n += got;
    return 0;
}

int s1_prompt_prefix(const struct s1_engine *e, const struct s1_template *t, const char *state,
                     int32_t *tok, int cap)
{
    size_t len = strlen(state);
    while (len > 0 && strchr(" \t\r\n", state[len - 1])) {
        len--; /* trailing whitespace is not part of the state */
    }
    int n = 0;
    if (add_piece(e, t->open, strlen(t->open), true, tok, cap, &n) != 0 ||
        add_piece(e, state, len, false, tok, cap, &n) != 0 ||
        add_piece(e, S1_SEP, strlen(S1_SEP), false, tok, cap, &n) != 0) {
        return -1;
    }
    return n;
}

static char *put(char *at, const char *s, size_t len)
{
    memcpy(at, s, len);
    return at + len;
}

/* "QUESTION\n<instructions>\nOPTIONS\nA. <description>\nB. <description>", with newlines
 * inside a description replaced by spaces. The caller frees the result. */
static char *question_body(const struct s1_question *q, size_t *len)
{
    static const char head[] = "QUESTION\n";
    static const char mid[]  = "\nOPTIONS";

    size_t cap = strlen(head) + strlen(q->instructions) + strlen(mid);
    for (int k = 0; k < q->K; k++) {
        cap += strlen("\nA. ") + strlen(q->option[k]);
    }
    char *body = malloc(cap);
    if (!body) {
        fprintf(stderr, "judgly: out of memory\n");
        return NULL;
    }
    char *at = put(body, head, strlen(head));
    at       = put(at, q->instructions, strlen(q->instructions));
    at       = put(at, mid, strlen(mid));
    for (int k = 0; k < q->K; k++) {
        const char line[] = { '\n', (char)('A' + k), '.', ' ' };
        at                = put(at, line, sizeof line);
        char *desc        = at;
        at                = put(at, q->option[k], strlen(q->option[k]));
        for (; desc < at; desc++) {
            if (*desc == '\n' || *desc == '\r') {
                *desc = ' ';
            }
        }
    }
    *len = cap;
    return body;
}

int s1_prompt_suffix(const struct s1_engine *e, const struct s1_template *t,
                     const struct s1_question *q, int32_t *tok, int cap)
{
    if (q->K < 2 || q->K > S1_K_MAX) {
        fprintf(stderr, "judgly: question has %d options, must be 2 to %d\n", q->K, S1_K_MAX);
        return -1;
    }
    size_t len  = 0;
    char  *body = question_body(q, &len);
    if (!body) {
        return -1;
    }
    int n  = 0;
    int rc = add_piece(e, body, len, false, tok, cap, &n);
    free(body);
    if (rc != 0 || add_piece(e, t->close, strlen(t->close), true, tok, cap, &n) != 0) {
        return -1;
    }
    return n;
}

int s1_slots_init(const struct s1_engine *e, bool leading_space, int32_t slot[S1_K_MAX])
{
    for (int k = 0; k < S1_K_MAX; k++) {
        const char text[] = { ' ', (char)('A' + k) };
        const char *from  = leading_space ? text : text + 1;
        int         len   = leading_space ? 2 : 1;
        int32_t     tok[4];
        int         n = s1_tokenize(e, from, len, false, tok, 4);
        if (n != 1) {
            fprintf(stderr, "judgly: letter '%.*s' is %d tokens, need exactly 1\n", len, from, n);
            return -1;
        }
        slot[k] = tok[0];
    }
    return 0;
}

void s1_slot_logprobs(const float *z, const int32_t *slot, int K, double *logp)
{
    double zmax = -INFINITY;
    for (int k = 0; k < K; k++) {
        if (z[slot[k]] > zmax) {
            zmax = z[slot[k]];
        }
    }
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += exp(z[slot[k]] - zmax);
    }
    double lse = zmax + log(sum);
    for (int k = 0; k < K; k++) {
        logp[k] = z[slot[k]] - lse;
    }
}

double s1_slot_mass(const float *z, int n_vocab, const int32_t *slot, int K)
{
    double zmax = -INFINITY;
    for (int v = 0; v < n_vocab; v++) {
        if (z[v] > zmax) {
            zmax = z[v];
        }
    }
    double all = 0.0;
    for (int v = 0; v < n_vocab; v++) {
        all += exp(z[v] - zmax);
    }
    double slots = 0.0;
    for (int k = 0; k < K; k++) {
        slots += exp(z[slot[k]] - zmax);
    }
    return slots / all;
}
