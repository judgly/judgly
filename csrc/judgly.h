/* judgly.h: the C API of libjudgly, loaded by the Python package through ctypes.
 *
 * JSON in, JSON out. A handle holds one model, one prompt template and the heads that turn its
 * answer-letter logits into probabilities. A handle is not re-entrant: calls on one handle must
 * not overlap (the Python Engine serialises them with a lock). Separate handles are independent
 * but each holds its own copy of the model in memory.
 *
 * Config (judgly_open), a JSON object:
 *   "model"            path of a single-file GGUF model (required)
 *   "template"         path of the prompt template file (required)
 *   "heads"            {format: head file}, "*" the fallback for any format; absent or {} for
 *                      no head (H0: raw letter probabilities). A head file is an H1 or H2 head
 *                      (applied to each rotation's letter logits) or a temperature head (one
 *                      temperature per question type, applied to the probabilities averaged
 *                      over the rotations); the file says which
 *   "model_sha256"     the model file's SHA-256 if the caller has verified it already; the file
 *                      is hashed when absent
 *   "n_ctx"            cache cells shared by the state and the question branches, at least
 *                      4096, the engine's batch size (32768)
 *   "n_seq"            sequences per decode call, the state included (65)
 *   "n_gpu_layers"     999 puts every layer on the GPU, 0 keeps all on the CPU (999)
 *   "rotations"        average over cyclic rotations of the options (true; always on for bool,
 *                      never for score, whose levels keep their natural order)
 *   "max_rotations"    at most this many rotations per question, 0 for all K (0)
 *   "content_free"     subtract the letter logits obtained against the state "N/A" (true)
 *   "max_state_tokens" longer states are cut to their first this many tokens and the response
 *                      says "truncated": true (n_ctx / 2)
 *   "plain_slots"      read the letters without a leading space (false)
 *   "verbose"          let llama.cpp and the engine log to stderr (false)
 *   "max_request_bytes" longer requests are refused before parsing (4 MiB = 4194304)
 *   "max_questions"    requests with more questions are refused (256)
 *
 * Request (judgly_decide), a JSON object:
 *   {"schema": 1, "state": "...", "questions": {id: question, ...}}
 *   question: {"type": "choice", "instructions": "...", "options": {key: description, ...}}
 *           | {"type": "bool", "instructions": "..."}
 *           | {"type": "score", "instructions": "...", "levels": 2..9}
 *   each with an optional "format" naming the head to use.
 *   Refused with an error: a key that appears twice in the request, in "questions", in a
 *   question or in "options"; any string or key containing a NUL character (\u0000). A state
 *   longer than max_state_tokens * 16 bytes is cut to that many bytes (at a UTF-8 character
 *   boundary) before it is tokenised and cut to max_state_tokens tokens.
 *
 * Response: {"schema": 1, "answers": {id: answer}, "model_sha256", "template_sha256",
 *   "truncated", "tokens": {"state", "questions"}, "timing_ms": {"state", "questions"}}
 *   answer: "type"; choice: "probs" {key: p}, "top" key; bool: "p_true", "top" true|false;
 *   score: "probs" [p per level], "mean", "top" level; all: "slot_mass", "rotation_spread" (the
 *   range, max minus min, of the top option's probability across the rotations asked; 0 with
 *   one rotation; with a temperature head, the range before the temperature, since the
 *   rotations are read without a head), "n_rotations", "format", "head" (the head file's
 *   SHA-256 or null),
 *   "head_format" (the heads entry used: the format itself, "*", or null).
 *   On failure: {"error": "..."}.
 */
#ifndef JUDGLY_H
#define JUDGLY_H

#include <stdint.h>

#if defined(_WIN32)
#define JUDGLY_API __declspec(dllexport)
#else
#define JUDGLY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct judgly_handle judgly_handle;

/* Build information as a JSON object: judgly version, the llama.cpp commit and ggml version
 * compiled in, and the ggml backends and devices available. The string is owned by the library
 * and stays valid until the process exits. Returns NULL if it cannot be built. */
JUDGLY_API const char *judgly_native_version(void);

/* Loads the model, template and heads named in config_json. Returns NULL on failure and, when
 * error is not NULL, sets *error to a message the caller releases with judgly_free_string. */
JUDGLY_API judgly_handle *judgly_open(const char *config_json, char **error);

/* Answers one request. Always returns a JSON string, the answers or {"error": ...}, which the
 * caller releases with judgly_free_string. Returns NULL only when memory runs out. */
JUDGLY_API char *judgly_decide(judgly_handle *h, const char *request_json);

JUDGLY_API void judgly_free_string(char *s);    /* accepts NULL */
JUDGLY_API void judgly_close(judgly_handle *h); /* accepts NULL */

/* Internal, for the test suite only (not a stable API): the mean cross-entropy plus penalty of
 * head parameters x on n records, with its gradient in g; the loss the head trainer minimises.
 * Arrays as in struct s1_records of the engine (h: n x n_embd, z/zc/target: n x 26, K: n). */
JUDGLY_API double judgly_test_head_loss(const double *x, double *g, int n_param, int h2,
                                        int n_embd, int n, const float *h, const float *z,
                                        const float *zc, const float *target, const uint8_t *K,
                                        double lambda);

/* Internal, for the test suite only: the engine's rotation rule (type as enum s1_type), the
 * trainer's temperature bound and H2 starting point on theta, and the spread of a head's
 * probabilities across records (s1_head_prob_sd). */
JUDGLY_API int    judgly_test_n_rotations(int type, int K, int rotations, int max_rotations);
JUDGLY_API double judgly_test_theta_clamp(double theta);
JUDGLY_API double judgly_test_h2_start_theta(double theta);
JUDGLY_API int    judgly_test_head_load(const char *path); /* 0 if the head file parses */
JUDGLY_API double judgly_test_head_prob_sd(const double *x, int h2, int n_embd, int n,
                                           const float *h, const float *z, const float *zc,
                                           const uint8_t *K);

#ifdef __cplusplus
}
#endif

#endif
