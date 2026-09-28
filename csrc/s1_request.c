/* The JSON contract: requests in, responses out. */
#include "s1.h"

#include <stdlib.h>
#include <string.h>

#include "yyjson.h"

#define S1_LEVELS_MAX 9

static const char *const LEVEL[S1_LEVELS_MAX] = { "1", "2", "3", "4", "5", "6", "7", "8", "9" };

static int bad(const char *name, const char *why)
{
    fprintf(stderr, "judgly: request: question \"%s\": %s\n", name, why);
    return -1;
}

static int parse_choice(struct s1_ask *a, yyjson_val *q)
{
    yyjson_val *options = yyjson_obj_get(q, "options");
    size_t      K       = yyjson_obj_size(options);
    if (!yyjson_is_obj(options) || K < 2 || K > S1_K_MAX) {
        return bad(a->name, "\"options\" must be an object with 2 to 26 entries");
    }
    yyjson_obj_iter it = yyjson_obj_iter_with(options);
    for (yyjson_val *key; (key = yyjson_obj_iter_next(&it));) {
        const char *text = yyjson_get_str(yyjson_obj_iter_get_val(key));
        if (!text) {
            return bad(a->name, "every option needs a string description");
        }
        a->key[a->K]      = yyjson_get_str(key);
        a->option[a->K++] = text;
    }
    return 0;
}

int s1_ask_parse(struct s1_ask *a, const char *name, yyjson_val *q)
{
    const char *type = yyjson_get_str(yyjson_obj_get(q, "type"));
    a->name          = name;
    a->instructions  = yyjson_get_str(yyjson_obj_get(q, "instructions"));
    if (!type || !a->instructions) {
        return bad(name, "needs string \"type\" and \"instructions\"");
    }
    if (strcmp(type, "choice") == 0) {
        a->type = S1_CHOICE;
        return parse_choice(a, q);
    }
    if (strcmp(type, "bool") == 0) {
        a->type      = S1_BOOL;
        a->option[0] = "True";
        a->option[1] = "False";
        a->K         = 2;
        return 0;
    }
    if (strcmp(type, "score") == 0) {
        yyjson_val *levels = yyjson_obj_get(q, "levels");
        int64_t     K      = yyjson_is_int(levels) ? yyjson_get_sint(levels) : 0;
        if (K < 2 || K > S1_LEVELS_MAX) {
            return bad(name, "\"levels\" must be an integer from 2 to 9");
        }
        a->type = S1_SCORE;
        a->K    = (int)K;
        for (int k = 0; k < a->K; k++) {
            a->option[k] = LEVEL[k];
        }
        return 0;
    }
    return bad(name, "\"type\" must be choice, bool or score");
}

int s1_ask_option_index(const struct s1_ask *a, const char *answer)
{
    if (a->type == S1_BOOL) {
        return strcmp(answer, "true") == 0 ? 0 : strcmp(answer, "false") == 0 ? 1 : -1;
    }
    for (int k = 0; k < a->K; k++) {
        if (strcmp(a->type == S1_CHOICE ? a->key[k] : a->option[k], answer) == 0) {
            return k;
        }
    }
    return -1;
}

int s1_example_parse(struct s1_example *x, const char *text, long where)
{
    static const char *const split_name[] = { "train", "validation", "test", "heldout" };
    memset(x, 0, sizeof *x);
    x->doc = yyjson_read(text, strlen(text), 0);
    if (!x->doc) {
        fprintf(stderr, "judgly: example %ld is not valid JSON\n", where);
        return -1;
    }
    yyjson_val *ex    = yyjson_doc_get_root(x->doc);
    yyjson_val *label = yyjson_obj_get(ex, "label");
    const char *split = yyjson_get_str(yyjson_obj_get(ex, "split"));
    x->id             = yyjson_get_str(yyjson_obj_get(ex, "id"));
    x->task           = yyjson_get_str(yyjson_obj_get(ex, "task"));
    x->family         = yyjson_get_str(yyjson_obj_get(ex, "family"));
    x->state          = yyjson_get_str(yyjson_obj_get(ex, "state"));
    x->split          = -1;
    for (int i = 0; split && i < 4; i++) {
        x->split = strcmp(split, split_name[i]) == 0 ? i : x->split;
    }
    if (!x->id || !x->task || !x->family || !x->state || x->split < 0 ||
        s1_ask_parse(&x->ask, x->id, ex) != 0) {
        fprintf(stderr, "judgly: example %ld needs id, task, family, split, state and a question\n",
                where);
        return -1;
    }
    if (yyjson_is_int(label)) { /* a score label is its level number */
        int64_t level = yyjson_get_sint(label);
        x->label      = level >= 1 && level <= x->ask.K ? (int)level - 1 : -1;
    } else {
        x->label = yyjson_is_str(label) ? s1_ask_option_index(&x->ask, yyjson_get_str(label)) : -1;
    }
    if (x->label < 0) {
        fprintf(stderr, "judgly: example %ld (%s): the label names no option\n", where, x->id);
        return -1;
    }
    return 0;
}

void s1_example_free(struct s1_example *x)
{
    yyjson_doc_free(x->doc);
    memset(x, 0, sizeof *x);
}

int s1_request_parse(struct s1_request *r, char *buf, size_t len, size_t *used)
{
    yyjson_read_err err;
    memset(r, 0, sizeof *r);
    r->doc = yyjson_read_opts(buf, len, YYJSON_READ_STOP_WHEN_DONE, NULL, &err);
    if (!r->doc) {
        fprintf(stderr, "judgly: request is not valid JSON at byte %zu: %s\n", err.pos, err.msg);
        return -1;
    }
    *used = yyjson_doc_get_read_size(r->doc);

    yyjson_val *root      = yyjson_doc_get_root(r->doc);
    yyjson_val *questions = yyjson_obj_get(root, "questions");
    r->state              = yyjson_get_str(yyjson_obj_get(root, "state"));
    if (!r->state || !yyjson_is_obj(questions) || yyjson_obj_size(questions) == 0) {
        fprintf(stderr, "judgly: request needs a string \"state\" and a non-empty \"questions\" object\n");
        goto fail;
    }
    r->ask = calloc(yyjson_obj_size(questions), sizeof *r->ask);
    if (!r->ask) {
        fprintf(stderr, "judgly: out of memory\n");
        goto fail;
    }
    yyjson_obj_iter it = yyjson_obj_iter_with(questions);
    for (yyjson_val *key; (key = yyjson_obj_iter_next(&it));) {
        if (s1_ask_parse(&r->ask[r->n_ask++], yyjson_get_str(key), yyjson_obj_iter_get_val(key)) != 0) {
            goto fail;
        }
    }
    return 0;

fail:
    s1_request_free(r);
    return -1;
}

void s1_request_free(struct s1_request *r)
{
    free(r->ask);
    yyjson_doc_free(r->doc);
    memset(r, 0, sizeof *r);
}

/* One entry of "answers", shaped by the question type. */
static yyjson_mut_val *answer(yyjson_mut_doc *doc, const struct s1_ask *a, const struct s1_reply *r)
{
    yyjson_mut_val *obj = yyjson_mut_obj(doc);
    if (a->type == S1_BOOL) {
        yyjson_mut_obj_add_real(doc, obj, "p_true", r->p[0]);
    } else if (a->type == S1_CHOICE) {
        yyjson_mut_val *probs = yyjson_mut_obj_add_obj(doc, obj, "probs");
        int             top   = 0;
        for (int k = 0; k < a->K; k++) {
            yyjson_mut_obj_add_real(doc, probs, a->key[k], r->p[k]);
            top = r->p[k] > r->p[top] ? k : top;
        }
        yyjson_mut_obj_add_str(doc, obj, "top", a->key[top]);
    } else {
        yyjson_mut_val *probs = yyjson_mut_obj_add_arr(doc, obj, "probs");
        double          mean  = 0.0;
        for (int k = 0; k < a->K; k++) {
            yyjson_mut_arr_add_real(doc, probs, r->p[k]);
            mean += (k + 1) * r->p[k];
        }
        yyjson_mut_obj_add_real(doc, obj, "mean", mean);
    }
    yyjson_mut_obj_add_real(doc, obj, "slot_mass", r->slot_mass);
    return obj;
}

int s1_response_write(FILE *out, const struct s1_request *r, const struct s1_reply *reply,
                      const struct s1_timing *timing, const char *model_sha256,
                      const char *head_sha256)
{
    yyjson_mut_doc *doc     = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root    = yyjson_mut_obj(doc);
    yyjson_mut_val *answers = yyjson_mut_obj_add_obj(doc, root, "answers");
    yyjson_mut_doc_set_root(doc, root);
    for (int i = 0; i < r->n_ask; i++) {
        yyjson_mut_obj_add_val(doc, answers, r->ask[i].name, answer(doc, &r->ask[i], &reply[i]));
    }
    yyjson_mut_val *ms = yyjson_mut_obj_add_obj(doc, root, "timing_ms");
    yyjson_mut_obj_add_real(doc, ms, "state", timing->state_ms);
    yyjson_mut_obj_add_real(doc, ms, "questions", timing->questions_ms);
    yyjson_mut_val *tokens = yyjson_mut_obj_add_obj(doc, root, "tokens");
    yyjson_mut_obj_add_int(doc, tokens, "state", timing->n_state_tokens);
    yyjson_mut_obj_add_int(doc, tokens, "questions", timing->n_question_tokens);
    yyjson_mut_obj_add_str(doc, root, "model", model_sha256);
    if (head_sha256) {
        yyjson_mut_obj_add_str(doc, root, "head", head_sha256);
    } else {
        yyjson_mut_obj_add_null(doc, root, "head");
    }

    char *text = yyjson_mut_write(doc, 0, NULL);
    int   rc   = text && fprintf(out, "%s\n", text) > 0 && fflush(out) == 0 ? 0 : -1;
    if (rc != 0) {
        fprintf(stderr, "judgly: cannot write the response\n");
    }
    free(text);
    yyjson_mut_doc_free(doc);
    return rc;
}
