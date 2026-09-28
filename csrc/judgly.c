/* judgly.c: the C API of libjudgly (judgly.h), a JSON layer over the engine (s1.h).
 * Requests are checked here with messages for the caller, then parsed and answered by the
 * engine's own request parser and decide path, so that a response carries the same
 * probabilities as the s1-run tool for the same model, template, head and options. */
#include "judgly.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "s1.h"
#include "yyjson.h"

#define HEADS_MAX 32
#define ERROR_MAX 512

#define DEFAULT_MAX_REQUEST_BYTES (4 << 20) /* 4 MiB */
#define DEFAULT_MAX_QUESTIONS     256
#define STATE_BYTES_PER_TOKEN     16 /* the state is cut to this many bytes per token first */

/* ---- version ---------------------------------------------------------------------------- */

static char *version_json; /* built once, kept for the life of the process */

const char *judgly_native_version(void)
{
    if (version_json)
        return version_json;

    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc)
        return NULL;
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "judgly", JUDGLY_VERSION);
    yyjson_mut_obj_add_str(doc, root, "llama_cpp_commit", JUDGLY_LLAMA_COMMIT);
    yyjson_mut_obj_add_str(doc, root, "ggml_version", ggml_version());

    yyjson_mut_val *backends = yyjson_mut_obj_add_arr(doc, root, "backends");
    for (size_t i = 0; i < ggml_backend_reg_count(); i++)
        yyjson_mut_arr_add_str(doc, backends, ggml_backend_reg_name(ggml_backend_reg_get(i)));

    yyjson_mut_val *devices = yyjson_mut_obj_add_arr(doc, root, "devices");
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        yyjson_mut_val    *d   = yyjson_mut_arr_add_obj(doc, devices);
        yyjson_mut_obj_add_str(doc, d, "name", ggml_backend_dev_name(dev));
        yyjson_mut_obj_add_str(doc, d, "description", ggml_backend_dev_description(dev));
    }

    version_json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    return version_json;
}

/* ---- handle ----------------------------------------------------------------------------- */

struct head_entry {
    char          *format; /* the key in "heads": a format name or "*" */
    struct s1_head head;
    char           sha256[S1_SHA256_HEX];
};

struct judgly_handle {
    struct s1_engine     *engine;
    struct s1_template    tpl;
    int32_t               slot[S1_K_MAX];
    char                  model_sha256[S1_SHA256_HEX];
    char                  template_sha256[S1_SHA256_HEX];
    struct head_entry     heads[HEADS_MAX];
    int                   n_heads;
    struct s1_decide_opts opts;
    int                   max_state_tokens;
    int                   n_ctx;
    size_t                max_request_bytes;
    size_t                max_questions;
    bool                  broken; /* the engine failed mid-request; its batch state is unknown */
};

#define ENGINE_BATCH 4096 /* S1_N_BATCH of s1_engine.c: most tokens in one decode call */
#define SUFFIX_CAP   4096 /* SUFFIX_CAP of s1_decide.c: tokens of one question */

static void quiet_log(enum ggml_log_level level, const char *text, void *user)
{
    (void)level;
    (void)text;
    (void)user;
}

static bool is_hex64(const char *s)
{
    if (!s || strlen(s) != 64)
        return false;
    for (const char *c = s; *c; c++)
        if (!((*c >= '0' && *c <= '9') || (*c >= 'a' && *c <= 'f')))
            return false;
    return true;
}

/* An optional integer field in [lo, hi]; *out keeps its default when the field is absent. */
static int opt_int(yyjson_val *obj, const char *key, int lo, int hi, int *out, char *why)
{
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v)
        return 0;
    if (!yyjson_is_int(v) || yyjson_get_sint(v) < lo || yyjson_get_sint(v) > hi) {
        snprintf(why, ERROR_MAX, "config: \"%s\" must be an integer from %d to %d", key, lo, hi);
        return -1;
    }
    *out = (int)yyjson_get_sint(v);
    return 0;
}

static int opt_bool(yyjson_val *obj, const char *key, bool *out, char *why)
{
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v)
        return 0;
    if (!yyjson_is_bool(v)) {
        snprintf(why, ERROR_MAX, "config: \"%s\" must be true or false", key);
        return -1;
    }
    *out = yyjson_get_bool(v);
    return 0;
}

static int check_keys(yyjson_val *obj, const char *const *known, const char *what, char *why)
{
    yyjson_obj_iter it = yyjson_obj_iter_with(obj);
    for (yyjson_val *key; (key = yyjson_obj_iter_next(&it));) {
        bool found = false;
        for (int i = 0; known[i] && !found; i++)
            found = strcmp(yyjson_get_str(key), known[i]) == 0;
        if (!found) {
            snprintf(why, ERROR_MAX, "%s: unknown field \"%s\"", what, yyjson_get_str(key));
            return -1;
        }
    }
    return 0;
}

static int load_heads(judgly_handle *h, yyjson_val *heads, char *why)
{
    if (!heads || yyjson_is_null(heads))
        return 0;
    if (!yyjson_is_obj(heads) || yyjson_obj_size(heads) > HEADS_MAX) {
        snprintf(why, ERROR_MAX, "config: \"heads\" must be an object of at most %d format: path "
                                 "entries", HEADS_MAX);
        return -1;
    }
    yyjson_obj_iter it = yyjson_obj_iter_with(heads);
    for (yyjson_val *key; (key = yyjson_obj_iter_next(&it));) {
        const char        *path = yyjson_get_str(yyjson_obj_iter_get_val(key));
        struct head_entry *e    = &h->heads[h->n_heads];
        if (!path) {
            snprintf(why, ERROR_MAX, "config: head \"%s\" must be a file path", yyjson_get_str(key));
            return -1;
        }
        e->format = strdup(yyjson_get_str(key));
        h->n_heads++;
        if (!e->format || s1_head_load(&e->head, path) != 0 || s1_sha256_file(path, e->sha256) != 0) {
            snprintf(why, ERROR_MAX, "cannot load head \"%s\" from %s", yyjson_get_str(key), path);
            return -1;
        }
        if (s1_head_check(&e->head, h->model_sha256, h->template_sha256, h->slot) != 0) {
            snprintf(why, ERROR_MAX, "head \"%s\" (%s) was trained for a different model, template "
                                     "or slot table", yyjson_get_str(key), path);
            return -1;
        }
        /* H2 reads n_embd floats of each readout's hidden vector: a head of another width would
         * read the wrong numbers, or past the end of the buffer. */
        if (e->head.h2 && e->head.n_embd != s1_engine_h_stride(h->engine)) {
            snprintf(why, ERROR_MAX, "head \"%s\" (%s) has hidden size %d; the model has %d",
                     yyjson_get_str(key), path, e->head.n_embd, s1_engine_h_stride(h->engine));
            return -1;
        }
    }
    return 0;
}

static judgly_handle *open_handle(const char *config_json, char *why)
{
    static const char *const known[] = { "model", "template", "heads", "model_sha256", "n_ctx",
                                         "n_seq", "n_gpu_layers", "rotations", "max_rotations",
                                         "content_free", "max_state_tokens", "plain_slots",
                                         "verbose", "max_request_bytes", "max_questions", NULL };
    yyjson_doc *doc = config_json ? yyjson_read(config_json, strlen(config_json), 0) : NULL;
    yyjson_val *cfg = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(cfg)) {
        snprintf(why, ERROR_MAX, "config: not a JSON object");
        yyjson_doc_free(doc);
        return NULL;
    }
    judgly_handle *h = calloc(1, sizeof *h);
    if (!h) {
        snprintf(why, ERROR_MAX, "out of memory");
        yyjson_doc_free(doc);
        return NULL;
    }
    const char *model    = yyjson_get_str(yyjson_obj_get(cfg, "model"));
    const char *tpl_path = yyjson_get_str(yyjson_obj_get(cfg, "template"));
    const char *sha      = yyjson_get_str(yyjson_obj_get(cfg, "model_sha256"));
    int         n_ctx = 32768, n_seq = 65, n_gpu = 999, max_rot = 0, max_state = -1;
    int         max_bytes = DEFAULT_MAX_REQUEST_BYTES, max_q = DEFAULT_MAX_QUESTIONS;
    bool        rotations = true, content_free = true, plain = false, verbose = false;
    int         rc = check_keys(cfg, known, "config", why);
    if (rc == 0 && (!model || !tpl_path)) {
        snprintf(why, ERROR_MAX, "config: \"model\" and \"template\" must be file paths");
        rc = -1;
    }
    if (rc == 0 && yyjson_obj_get(cfg, "model_sha256") && !is_hex64(sha)) {
        snprintf(why, ERROR_MAX, "config: \"model_sha256\" must be 64 lower-case hex digits");
        rc = -1;
    }
    rc = rc ? rc : opt_int(cfg, "n_ctx", 4096, 1 << 20, &n_ctx, why);
    rc = rc ? rc : opt_int(cfg, "n_seq", 2, 256, &n_seq, why);
    rc = rc ? rc : opt_int(cfg, "n_gpu_layers", 0, 999, &n_gpu, why);
    rc = rc ? rc : opt_int(cfg, "max_rotations", 0, S1_K_MAX, &max_rot, why);
    rc = rc ? rc : opt_int(cfg, "max_state_tokens", 1, 1 << 20, &max_state, why);
    rc = rc ? rc : opt_int(cfg, "max_request_bytes", 64, 1 << 30, &max_bytes, why);
    rc = rc ? rc : opt_int(cfg, "max_questions", 1, 1 << 16, &max_q, why);
    rc = rc ? rc : opt_bool(cfg, "rotations", &rotations, why);
    rc = rc ? rc : opt_bool(cfg, "content_free", &content_free, why);
    rc = rc ? rc : opt_bool(cfg, "plain_slots", &plain, why);
    rc = rc ? rc : opt_bool(cfg, "verbose", &verbose, why);
    if (rc == 0) {
        llama_log_set(verbose ? NULL : quiet_log, NULL);
        h->opts             = (struct s1_decide_opts){ rotations, content_free, max_rot };
        h->max_state_tokens = max_state > 0 ? max_state : n_ctx / 2;
        h->n_ctx            = n_ctx;
        h->max_request_bytes = (size_t)max_bytes;
        h->max_questions     = (size_t)max_q;
        if (sha) {
            memcpy(h->model_sha256, sha, S1_SHA256_HEX);
        } else if (s1_sha256_file(model, h->model_sha256) != 0) {
            snprintf(why, ERROR_MAX, "cannot read the model file %s", model);
            rc = -1;
        }
    }
    if (rc == 0 && s1_sha256_file(tpl_path, h->template_sha256) != 0) {
        snprintf(why, ERROR_MAX, "cannot read the template file %s", tpl_path);
        rc = -1;
    }
    if (rc == 0 && s1_template_load(&h->tpl, tpl_path) != 0) {
        snprintf(why, ERROR_MAX, "cannot parse the template file %s", tpl_path);
        rc = -1;
    }
    struct s1_engine_params ep = { model, n_ctx, n_seq, n_gpu, false, NULL, 0 };
    if (rc == 0 && s1_engine_init(&h->engine, &ep) != 0) {
        snprintf(why, ERROR_MAX, "cannot load the model %s", model);
        rc = -1;
    }
    if (rc == 0 && s1_slots_init(h->engine, !plain, h->slot) != 0) {
        snprintf(why, ERROR_MAX, "the model does not have one token per option letter");
        rc = -1;
    }
    rc = rc ? rc : load_heads(h, yyjson_obj_get(cfg, "heads"), why);
    yyjson_doc_free(doc);
    if (rc != 0) {
        judgly_close(h);
        return NULL;
    }
    return h;
}

judgly_handle *judgly_open(const char *config_json, char **error)
{
    char           why[ERROR_MAX] = "";
    judgly_handle *h              = open_handle(config_json, why);
    if (error)
        *error = h ? NULL : strdup(why);
    return h;
}

void judgly_close(judgly_handle *h)
{
    if (!h)
        return;
    for (int i = 0; i < h->n_heads; i++) {
        s1_head_free(&h->heads[i].head);
        free(h->heads[i].format);
    }
    s1_template_free(&h->tpl);
    s1_engine_free(h->engine);
    free(h);
}

void judgly_free_string(char *s) { free(s); }

/* ---- requests --------------------------------------------------------------------------- */

/* Every string of the document, keys included, has no embedded NUL: C code downstream reads them
 * with strlen and strcmp, so "state\u0000x" would otherwise pass as "state". The values of an
 * immutable yyjson document are one flat array. */
static int check_nul(yyjson_doc *doc, char *why)
{
    yyjson_val *v = yyjson_doc_get_root(doc);
    size_t      n = yyjson_doc_get_val_count(doc);
    for (size_t i = 0; i < n; i++, v++) {
        if (yyjson_is_str(v) && strlen(yyjson_get_str(v)) != yyjson_get_len(v)) {
            snprintf(why, ERROR_MAX, "request: a string contains a NUL character (\\u0000)");
            return -1;
        }
    }
    return 0;
}

/* No key appears twice in obj (yyjson keeps duplicates; the last one would silently win). */
static int check_unique(yyjson_val *obj, const char *what, char *why)
{
    yyjson_obj_iter it = yyjson_obj_iter_with(obj);
    for (yyjson_val *a; (a = yyjson_obj_iter_next(&it));) {
        yyjson_obj_iter rest = it; /* the keys after a */
        for (yyjson_val *b; (b = yyjson_obj_iter_next(&rest));) {
            if (yyjson_equals_strn(b, yyjson_get_str(a), yyjson_get_len(a))) {
                snprintf(why, ERROR_MAX, "%s: duplicate key \"%.100s\"", what, yyjson_get_str(a));
                return -1;
            }
        }
    }
    return 0;
}

/* The checks of s1_request_parse and s1_ask_parse, with messages for the caller, plus the
 * judgly fields: "schema", "format", and no unknown fields. Collects each question's format. */
static int check_request(const judgly_handle *h, yyjson_val *root, const char **format, char *why)
{
    static const char *const known_root[] = { "schema", "state", "questions", NULL };
    static const char *const known_q[]    = { "type", "format", "instructions", "options",
                                              "levels", NULL };
    if (!yyjson_is_obj(root)) {
        snprintf(why, ERROR_MAX, "request: not a JSON object");
        return -1;
    }
    if (check_unique(root, "request", why) != 0 || check_keys(root, known_root, "request", why) != 0)
        return -1;
    yyjson_val *schema = yyjson_obj_get(root, "schema");
    if (schema && !(yyjson_is_int(schema) && yyjson_get_sint(schema) == 1)) {
        snprintf(why, ERROR_MAX, "request: \"schema\" must be 1");
        return -1;
    }
    yyjson_val *questions = yyjson_obj_get(root, "questions");
    if (!yyjson_is_str(yyjson_obj_get(root, "state"))) {
        snprintf(why, ERROR_MAX, "request: \"state\" must be a string");
        return -1;
    }
    if (!yyjson_is_obj(questions) || yyjson_obj_size(questions) == 0) {
        snprintf(why, ERROR_MAX, "request: \"questions\" must be a non-empty object");
        return -1;
    }
    if (yyjson_obj_size(questions) > h->max_questions) {
        snprintf(why, ERROR_MAX, "request: %zu questions, more than max_questions %zu",
                 yyjson_obj_size(questions), h->max_questions);
        return -1;
    }
    if (check_unique(questions, "request: \"questions\"", why) != 0)
        return -1;
    int             i  = 0;
    yyjson_obj_iter it = yyjson_obj_iter_with(questions);
    for (yyjson_val *key; (key = yyjson_obj_iter_next(&it)); i++) {
        const char *name = yyjson_get_str(key);
        yyjson_val *q    = yyjson_obj_iter_get_val(key);
        char        what[ERROR_MAX / 2];
        snprintf(what, sizeof what, "question \"%.100s\"", name);
        if (!yyjson_is_obj(q)) {
            snprintf(why, ERROR_MAX, "%s: must be an object", what);
            return -1;
        }
        if (check_unique(q, what, why) != 0 || check_keys(q, known_q, what, why) != 0)
            return -1;
        const char *type = yyjson_get_str(yyjson_obj_get(q, "type"));
        yyjson_val *fmt  = yyjson_obj_get(q, "format");
        if (!type || !yyjson_is_str(yyjson_obj_get(q, "instructions"))) {
            snprintf(why, ERROR_MAX, "%s: needs string \"type\" and \"instructions\"", what);
            return -1;
        }
        if (fmt && !yyjson_is_str(fmt) && !yyjson_is_null(fmt)) {
            snprintf(why, ERROR_MAX, "%s: \"format\" must be a string", what);
            return -1;
        }
        format[i] = yyjson_get_str(fmt);
        if (strcmp(type, "choice") == 0) {
            yyjson_val *options = yyjson_obj_get(q, "options");
            size_t      K       = yyjson_obj_size(options);
            if (!yyjson_is_obj(options) || K < 2 || K > S1_K_MAX) {
                snprintf(why, ERROR_MAX, "%s: \"options\" must be an object with 2 to 26 entries",
                         what);
                return -1;
            }
            char owhat[ERROR_MAX / 2];
            snprintf(owhat, sizeof owhat, "%s: \"options\"", what);
            if (check_unique(options, owhat, why) != 0)
                return -1;
            yyjson_obj_iter oi = yyjson_obj_iter_with(options);
            for (yyjson_val *k; (k = yyjson_obj_iter_next(&oi));) {
                if (!yyjson_is_str(yyjson_obj_iter_get_val(k))) {
                    snprintf(why, ERROR_MAX, "%s: every option needs a string description", what);
                    return -1;
                }
            }
            if (yyjson_obj_get(q, "levels")) {
                snprintf(why, ERROR_MAX, "%s: \"levels\" belongs to score questions", what);
                return -1;
            }
        } else if (strcmp(type, "bool") == 0) {
            if (yyjson_obj_get(q, "options") || yyjson_obj_get(q, "levels")) {
                snprintf(why, ERROR_MAX, "%s: a bool question has no \"options\" or \"levels\"",
                         what);
                return -1;
            }
        } else if (strcmp(type, "score") == 0) {
            yyjson_val *levels = yyjson_obj_get(q, "levels");
            int64_t     K      = yyjson_is_int(levels) ? yyjson_get_sint(levels) : 0;
            if (K < 2 || K > 9) {
                snprintf(why, ERROR_MAX, "%s: \"levels\" must be an integer from 2 to 9", what);
                return -1;
            }
            if (yyjson_obj_get(q, "options")) {
                snprintf(why, ERROR_MAX, "%s: \"options\" belongs to choice questions", what);
                return -1;
            }
        } else {
            snprintf(why, ERROR_MAX, "%s: \"type\" must be choice, bool or score", what);
            return -1;
        }
    }
    return 0;
}

/* The state cut to its first max_state_tokens tokens, as text (*cut, malloc'd), or *cut NULL
 * when it fits. */
static int truncate_state(const judgly_handle *h, const char *state, char **cut, char *why)
{
    *cut = NULL;
    /* A token is at most a few bytes in practice; reading more than STATE_BYTES_PER_TOKEN bytes
     * per kept token only costs tokeniser time, so the text is cut there first, back to the start
     * of a UTF-8 character. */
    size_t full  = strlen(state);
    size_t limit = (size_t)h->max_state_tokens * STATE_BYTES_PER_TOKEN;
    size_t bytes = full;
    if (bytes > limit) {
        bytes = limit;
        while (bytes > 0 && ((unsigned char)state[bytes] & 0xC0) == 0x80)
            bytes--;
    }
    if (bytes > (size_t)INT_MAX - 16) {
        snprintf(why, ERROR_MAX, "the state is too long");
        return -1;
    }
    if (bytes < full) {
        *cut = malloc(bytes + 1);
        if (!*cut) {
            snprintf(why, ERROR_MAX, "out of memory");
            return -1;
        }
        memcpy(*cut, state, bytes);
        (*cut)[bytes] = '\0';
        state         = *cut;
    }
    int      len = (int)bytes;
    int      cap = len + 16; /* a token is at least a byte */
    int32_t *tok = malloc((size_t)cap * sizeof *tok);
    int      n   = tok ? s1_tokenize(h->engine, state, len, false, tok, cap) : -1;
    if (n < 0) {
        snprintf(why, ERROR_MAX, "cannot tokenise the state");
        free(tok);
        free(*cut);
        *cut = NULL;
        return -1;
    }
    /* Detokenising and tokenising again may merge tokens differently; shrink until it fits. */
    for (int keep = h->max_state_tokens; n > h->max_state_tokens && keep > 0; keep -= 8) {
        size_t text_cap = (size_t)keep * 64 + 16;
        free(*cut);
        *cut = malloc(text_cap);
        if (!*cut || s1_detokenize(h->engine, (struct s1_span){ tok, keep }, *cut,
                                   (int)text_cap) < 0) {
            break;
        }
        n = s1_tokenize(h->engine, *cut, (int)strlen(*cut), false, tok, cap);
        if (n < 0)
            break;
    }
    free(tok);
    if (*cut && (n < 0 || n > h->max_state_tokens)) {
        snprintf(why, ERROR_MAX, "cannot truncate the state to %d tokens", h->max_state_tokens);
        free(*cut);
        *cut = NULL;
        return -1;
    }
    return 0;
}

/* Cache cells a request needs, checked before the engine runs so that an oversized request is
 * refused instead of failing inside llama_decode: the prefix (template, state, separator) plus
 * the question branches decoded together, at most one engine batch of them. Rotations are
 * counted as s1_decide.c plans them. */
static int check_budget(const judgly_handle *h, const char *state, const struct s1_ask *ask,
                        int n_ask, char *why)
{
    size_t   chars  = strlen(h->tpl.open) + strlen(state);
    if (chars > (size_t)INT_MAX - 16) {
        snprintf(why, ERROR_MAX, "the state is too long");
        return -1;
    }
    int      cap    = (int)chars + 16;
    int32_t *tok    = malloc((size_t)(cap > SUFFIX_CAP ? cap : SUFFIX_CAP) * sizeof *tok);
    int      prefix = tok ? s1_prompt_prefix(h->engine, &h->tpl, state, tok, cap) : -1;
    long     branch = 0;
    for (int i = 0; i < n_ask && prefix >= 0; i++) {
        const struct s1_ask *a = &ask[i];
        int n_rot = s1_n_rotations(a->type, a->K, h->opts);
        for (int j = 0; j < n_rot; j++) {
            int         rot = s1_rotation_of(j, n_rot, a->K);
            const char *shown[S1_K_MAX];
            for (int k = 0; k < a->K; k++)
                shown[k] = a->option[(k + rot) % a->K];
            struct s1_question q = { a->instructions, shown, a->K };
            int                n = s1_prompt_suffix(h->engine, &h->tpl, &q, tok, SUFFIX_CAP);
            if (n < 0) {
                snprintf(why, ERROR_MAX, "question \"%.100s\" is longer than %d tokens", a->name,
                         SUFFIX_CAP);
                free(tok);
                return -1;
            }
            branch += n;
        }
    }
    free(tok);
    if (prefix < 0) {
        snprintf(why, ERROR_MAX, "cannot tokenise the state");
        return -1;
    }
    long need = prefix + (branch < ENGINE_BATCH ? branch : ENGINE_BATCH);
    if (need > h->n_ctx) {
        snprintf(why, ERROR_MAX, "the request needs up to %ld cache cells (state %d tokens, "
                 "questions %ld), more than n_ctx %d; lower max_state_tokens or raise n_ctx",
                 need, prefix, branch, h->n_ctx);
        return -1;
    }
    return 0;
}

static const struct head_entry *head_for(const judgly_handle *h, const char *format)
{
    const struct head_entry *star = NULL;
    for (int i = 0; i < h->n_heads; i++) {
        if (format && strcmp(h->heads[i].format, format) == 0)
            return &h->heads[i];
        if (strcmp(h->heads[i].format, "*") == 0)
            star = &h->heads[i];
    }
    return star;
}

/* One answer object. The mean over rotations is formed exactly as s1_combine forms it. */
static yyjson_mut_val *answer(yyjson_mut_doc *doc, const struct s1_ask *a, int ask_index,
                              const char *format, const struct head_entry *he,
                              const struct s1_readouts *r)
{
    const struct s1_head *head = he ? &he->head : NULL;
    double                p[S1_K_MAX] = { 0 };
    double                slot_mass   = 0.0;
    int                   n_rot       = 0;
    for (int j = 0; j < r->n; j++) {
        const struct s1_readout *item = &r->item[j];
        if (item->ask != ask_index)
            continue;
        double q[S1_K_MAX];
        s1_readout_probs(item, head, a->type, a->K, q);
        for (int s = 0; s < a->K; s++)
            p[(s + item->rotation) % a->K] += q[s];
        slot_mass += (double)item->slot_mass;
        n_rot++;
    }
    slot_mass /= n_rot;
    int top = 0;
    for (int k = 0; k < a->K; k++) {
        p[k] /= n_rot;
        top = p[k] > p[top] ? k : top;
    }
    /* The top option's probability under each rotation: its range. */
    double lo = INFINITY, hi = -INFINITY;
    for (int j = 0; j < r->n; j++) {
        const struct s1_readout *item = &r->item[j];
        if (item->ask != ask_index)
            continue;
        double q[S1_K_MAX];
        s1_readout_probs(item, head, a->type, a->K, q);
        double v = q[(top - item->rotation % a->K + a->K) % a->K];
        lo       = fmin(lo, v);
        hi       = fmax(hi, v);
    }

    yyjson_mut_val *obj = yyjson_mut_obj(doc);
    if (a->type == S1_BOOL) {
        yyjson_mut_obj_add_str(doc, obj, "type", "bool");
        yyjson_mut_obj_add_real(doc, obj, "p_true", p[0]);
        yyjson_mut_obj_add_bool(doc, obj, "top", top == 0);
    } else if (a->type == S1_CHOICE) {
        yyjson_mut_obj_add_str(doc, obj, "type", "choice");
        yyjson_mut_val *probs = yyjson_mut_obj_add_obj(doc, obj, "probs");
        for (int k = 0; k < a->K; k++)
            yyjson_mut_obj_add_real(doc, probs, a->key[k], p[k]);
        yyjson_mut_obj_add_str(doc, obj, "top", a->key[top]);
    } else {
        yyjson_mut_obj_add_str(doc, obj, "type", "score");
        yyjson_mut_val *probs = yyjson_mut_obj_add_arr(doc, obj, "probs");
        double          mean  = 0.0;
        for (int k = 0; k < a->K; k++) {
            yyjson_mut_arr_add_real(doc, probs, p[k]);
            mean += (k + 1) * p[k];
        }
        yyjson_mut_obj_add_real(doc, obj, "mean", mean);
        yyjson_mut_obj_add_int(doc, obj, "top", top + 1);
    }
    yyjson_mut_obj_add_real(doc, obj, "slot_mass", slot_mass);
    yyjson_mut_obj_add_real(doc, obj, "rotation_spread", n_rot > 1 ? hi - lo : 0.0);
    yyjson_mut_obj_add_int(doc, obj, "n_rotations", n_rot);
    if (format)
        yyjson_mut_obj_add_str(doc, obj, "format", format);
    else
        yyjson_mut_obj_add_null(doc, obj, "format");
    if (he) {
        yyjson_mut_obj_add_str(doc, obj, "head", he->sha256);
        yyjson_mut_obj_add_str(doc, obj, "head_format", he->format);
    } else {
        yyjson_mut_obj_add_null(doc, obj, "head");
        yyjson_mut_obj_add_null(doc, obj, "head_format");
    }
    return obj;
}

static char *error_json(const char *why)
{
    yyjson_mut_doc *doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    char           *text = NULL;
    if (root) {
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_str(doc, root, "error", why);
        text = yyjson_mut_write(doc, 0, NULL);
    }
    yyjson_mut_doc_free(doc);
    return text;
}

char *judgly_decide(judgly_handle *h, const char *request_json)
{
    char why[ERROR_MAX] = "";
    if (!h)
        return error_json("no engine handle");
    if (h->broken)
        return error_json("an earlier request failed inside the engine; close this handle and "
                          "open a new one");
    if (!request_json)
        return error_json("request: missing");

    size_t len = strlen(request_json);
    if (len > h->max_request_bytes) {
        snprintf(why, ERROR_MAX, "request: %zu bytes, more than max_request_bytes %zu", len,
                 h->max_request_bytes);
        return error_json(why);
    }
    yyjson_read_err err;
    yyjson_doc     *check = yyjson_read_opts((char *)(uintptr_t)request_json, len, 0, NULL, &err);
    if (!check) {
        snprintf(why, ERROR_MAX, "request: not valid JSON at byte %zu: %s", err.pos, err.msg);
        return error_json(why);
    }
    if (check_nul(check, why) != 0) {
        yyjson_doc_free(check);
        return error_json(why);
    }
    yyjson_val  *root      = yyjson_doc_get_root(check);
    size_t       n_q       = yyjson_obj_size(yyjson_obj_get(root, "questions"));
    const char **format    = calloc(n_q ? n_q : 1, sizeof *format);
    int          rc        = format ? check_request(h, root, format, why) : -1;
    char        *buf       = NULL;
    char        *cut       = NULL;
    char        *text      = NULL;
    bool         parsed    = false;
    struct s1_request   req;
    struct s1_readouts  r      = { 0 };
    struct s1_timing    timing = { 0 };
    if (!format)
        snprintf(why, ERROR_MAX, "out of memory");

    /* The engine's parser reads from a writable buffer. */
    if (rc == 0 && (buf = malloc(len + 1)) == NULL) {
        snprintf(why, ERROR_MAX, "out of memory");
        rc = -1;
    }
    if (rc == 0) {
        size_t used = 0;
        memcpy(buf, request_json, len + 1);
        rc = s1_request_parse(&req, buf, len, &used);
        parsed = rc == 0;
        if (rc != 0)
            snprintf(why, ERROR_MAX, "request: rejected by the engine's parser");
    }
    if (rc == 0)
        rc = truncate_state(h, req.state, &cut, why);
    if (rc == 0)
        rc = check_budget(h, cut ? cut : req.state, req.ask, req.n_ask, why);
    if (rc == 0 && s1_read(h->engine, &h->tpl, h->slot, cut ? cut : req.state, req.ask, req.n_ask,
                           h->opts, &r, &timing) != 0) {
        snprintf(why, ERROR_MAX, "the engine failed on this request (details on stderr); this "
                                 "handle can no longer be used");
        h->broken = true;
        rc = -1;
    }
    if (rc == 0) {
        yyjson_mut_doc *doc  = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *out  = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, out);
        yyjson_mut_obj_add_int(doc, out, "schema", 1);
        yyjson_mut_val *answers = yyjson_mut_obj_add_obj(doc, out, "answers");
        for (int i = 0; i < req.n_ask; i++) {
            yyjson_mut_obj_add_val(doc, answers, req.ask[i].name,
                                   answer(doc, &req.ask[i], i, format[i],
                                          head_for(h, format[i]), &r));
        }
        yyjson_mut_obj_add_str(doc, out, "model_sha256", h->model_sha256);
        yyjson_mut_obj_add_str(doc, out, "template_sha256", h->template_sha256);
        yyjson_mut_obj_add_bool(doc, out, "truncated", cut != NULL);
        yyjson_mut_val *tokens = yyjson_mut_obj_add_obj(doc, out, "tokens");
        yyjson_mut_obj_add_int(doc, tokens, "state", timing.n_state_tokens);
        yyjson_mut_obj_add_int(doc, tokens, "questions", timing.n_question_tokens);
        yyjson_mut_val *ms = yyjson_mut_obj_add_obj(doc, out, "timing_ms");
        yyjson_mut_obj_add_real(doc, ms, "state", timing.state_ms);
        yyjson_mut_obj_add_real(doc, ms, "questions", timing.questions_ms);
        text = yyjson_mut_write(doc, 0, NULL);
        yyjson_mut_doc_free(doc);
    }
    s1_readouts_free(&r);
    if (parsed)
        s1_request_free(&req);
    free(cut);
    free(buf);
    free(format);
    yyjson_doc_free(check);
    return rc == 0 ? text : error_json(why);
}

/* ---- test hook -------------------------------------------------------------------------- */

double judgly_test_head_loss(const double *x, double *g, int n_param, int h2, int n_embd, int n,
                             const float *h, const float *z, const float *zc, const float *target,
                             const uint8_t *K, double lambda)
{
    struct s1_records data = { n, n_embd, h, z, zc, target, K, NULL };
    struct s1_fit     fit  = { &data, h2 != 0, true, lambda, false };
    return s1_head_loss(x, g, n_param, &fit);
}

int judgly_test_n_rotations(int type, int K, int rotations, int max_rotations)
{
    struct s1_decide_opts opts = { rotations != 0, false, max_rotations };
    return s1_n_rotations((enum s1_type)type, K, opts);
}

double judgly_test_theta_clamp(double theta)
{
    return s1_theta_clamp(theta);
}

double judgly_test_h2_start_theta(double theta)
{
    double x[S1_HEAD_D] = { 0.0 };
    x[S1_HEAD_THETA]    = theta;
    s1_head_h2_start(x);
    return x[S1_HEAD_THETA];
}

double judgly_test_head_prob_sd(const double *x, int h2, int n_embd, int n, const float *h,
                                const float *z, const float *zc, const uint8_t *K)
{
    struct s1_records data = { n, n_embd, h, z, zc, NULL, K, NULL };
    return s1_head_prob_sd(x, h2 != 0, &data);
}

int judgly_test_head_load(const char *path)
{
    struct s1_head head;
    if (s1_head_load(&head, path) != 0)
        return -1;
    s1_head_free(&head);
    return 0;
}
