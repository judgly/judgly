/* s1-run: typed questions about a state in, a probability per allowed answer out.
 * Request and response are the JSON of csrc/s1_request.c. Every JSON document on stdin is
 * answered with one line on stdout; the log goes to stderr. No text is ever generated. */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s1.h"

#define N_CTX 32768
#define N_SEQ 65 /* 64 branches per call, and the state */

/* All of stdin as one NUL-terminated buffer, with the padding yyjson asks for. */
static char *read_stdin(size_t *len)
{
    size_t cap = 1 << 16;
    char  *buf = malloc(cap);
    *len       = 0;
    while (buf) {
        *len += fread(buf + *len, 1, cap - *len - 8, stdin);
        if (*len < cap - 8) {
            break;
        }
        cap *= 2;
        char *grown = realloc(buf, cap);
        if (!grown) {
            free(buf);
        }
        buf = grown;
    }
    if (!buf || ferror(stdin)) {
        fprintf(stderr, "s1-run: cannot read stdin\n");
        free(buf);
        return NULL;
    }
    memset(buf + *len, 0, 8);
    return buf;
}

static int answer_all(struct s1_engine *e, const struct s1_template *tpl, const int32_t *slot,
                      struct s1_decide_opts opts, const struct s1_head *head,
                      const char *model_sha256, const char *head_sha256, char *buf, size_t len)
{
    int n = 0;
    for (size_t at = 0;; n++) {
        while (at < len && isspace((unsigned char)buf[at])) {
            at++;
        }
        if (at == len) {
            break;
        }
        struct s1_request req;
        struct s1_timing  timing;
        size_t            used  = 0;
        struct s1_reply  *reply = NULL;
        int               rc    = s1_request_parse(&req, buf + at, len - at, &used);
        if (rc == 0) {
            reply = malloc((size_t)req.n_ask * sizeof *reply);
            rc    = reply ? s1_decide(e, tpl, slot, req.state, req.ask, req.n_ask, opts, head,
                                      reply, &timing)
                          : -1;
        }
        if (rc == 0) {
            rc = s1_response_write(stdout, &req, reply, &timing, model_sha256, head_sha256);
        }
        free(reply);
        s1_request_free(&req);
        if (rc != 0) {
            fprintf(stderr, "s1-run: request %d failed\n", n + 1);
            return -1;
        }
        at += used;
    }
    if (n == 0) {
        fprintf(stderr, "s1-run: no request on stdin\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const char *const options[] = { "--model", "--template", "--head", "--max-rotations", NULL };
    static const char *const flags[]   = { "--rotations", "--content-free", "--plain-slots", NULL };
    if (s1_args_check(argc, argv, options, flags) != 0) {
        return 2;
    }
    const char *model    = s1_arg_value(argc, argv, "--model");
    const char *tpl_path = s1_arg_value(argc, argv, "--template");
    if (!model) {
        model = getenv("S1_MODEL");
    }
    if (!model || !tpl_path) {
        fprintf(stderr, "usage: s1-run --model FILE.gguf --template FILE.tpl "
                        "[--head HEAD.bin]\n"
                        "         [--rotations] [--max-rotations N] [--content-free] [--plain-slots] < request.json\n"
                        "       --model defaults to $S1_MODEL. --plain-slots reads the letters "
                        "without a leading space.\n");
        return 2;
    }
    struct s1_decide_opts opts = { s1_arg_flag(argc, argv, "--rotations"),
                                   s1_arg_flag(argc, argv, "--content-free"),
                                   (int)strtol(s1_arg_value(argc, argv, "--max-rotations")
                                                   ? s1_arg_value(argc, argv, "--max-rotations") : "0",
                                               NULL, 10) };
    struct s1_engine_params ep = { model, N_CTX, N_SEQ, 999, false, NULL, 0 };

    const char        *head_path = s1_arg_value(argc, argv, "--head");
    struct s1_engine  *engine    = NULL;
    struct s1_template tpl       = { 0 };
    struct s1_head     head      = { 0 };
    int32_t            slot[S1_K_MAX];
    char               sha[S1_SHA256_HEX];
    char               tpl_sha[S1_SHA256_HEX];
    char               head_sha[S1_SHA256_HEX];
    size_t             len    = 0;
    char              *buf    = read_stdin(&len);
    int                status = 1;
    if (buf && s1_sha256_file(model, sha) == 0 && s1_sha256_file(tpl_path, tpl_sha) == 0 &&
        s1_engine_init(&engine, &ep) == 0 && s1_template_load(&tpl, tpl_path) == 0 &&
        s1_slots_init(engine, !s1_arg_flag(argc, argv, "--plain-slots"), slot) == 0 &&
        (!head_path || (s1_head_load(&head, head_path) == 0 &&
                        s1_sha256_file(head_path, head_sha) == 0 &&
                        s1_head_check(&head, sha, tpl_sha, slot) == 0)) &&
        answer_all(engine, &tpl, slot, opts, head_path ? &head : NULL, sha,
                   head_path ? head_sha : NULL, buf, len) == 0) {
        status = 0;
    }
    s1_head_free(&head);
    s1_template_free(&tpl);
    s1_engine_free(engine);
    free(buf);
    return status;
}
