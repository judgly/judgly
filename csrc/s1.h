/* s1: decisions read from one position of a frozen language model.
 * Functions that can fail return 0 on success and
 * -1 after printing the reason to stderr, unless their comment says otherwise. */
#ifndef S1_H
#define S1_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define S1_K_MAX      26 /* one slot per option letter, A to Z */
#define S1_SHA256_HEX 65 /* 64 hex digits and the NUL */

/* A run of token ids. Never owns its memory. */
struct s1_span {
    const int32_t *tok;
    int            n;
};

/* What the backbone returns at one decision position. Caller-allocated. */
struct s1_out {
    float *h; /* n_embd floats taken after the final norm, then n_embd floats per captured layer
                 (s1_engine_h_stride in all) */
    float *z; /* n_vocab logits, as libllama returns them */
};

/* ---- engine: the model, its cache, and the two GPU calls ---- */

struct s1_engine;

struct s1_engine_params {
    const char *model_path;
    int         n_ctx;        /* cache cells shared by all sequences */
    int         n_seq;        /* sequences per decode call, the state included; 2 to 256 */
    int         n_gpu_layers; /* 999 puts every layer on the GPU, 0 keeps all on the CPU */
    bool        swa_pruned;   /* let sliding-window layers drop entries outside the window;
                                 the default keeps the whole state */
    const int  *layers;       /* up to S1_LAYERS_MAX layers whose output is captured at every
                                 decision position and appended to h; NULL for none */
    int         n_layers;
};

#define S1_LAYERS_MAX 8
#define S1_LAYER_NORM (-1) /* in `layers`: the final-norm output itself, copied as the model
                              returns it (the capture check); any other id is the output of
                              that block ("l_out"), divided by its root mean square */

/* True when the model has sliding-window attention layers. */
bool s1_engine_has_swa(const struct s1_engine *e);

/* Loads the model and creates the context. On success *e is owned by the caller
 * and released with s1_engine_free. */
int  s1_engine_init(struct s1_engine **e, const struct s1_engine_params *p);
void s1_engine_free(struct s1_engine *e); /* accepts NULL */

int s1_engine_n_embd(const struct s1_engine *e);
int s1_engine_n_vocab(const struct s1_engine *e);
int s1_engine_n_seq(const struct s1_engine *e);
int s1_engine_n_ctx(const struct s1_engine *e); /* cache cells shared by all sequences */
int s1_engine_h_stride(const struct s1_engine *e); /* n_embd * (1 + captured layers) */

/* Text to token ids, never adding BOS or EOS. With parse_special false, the text of a
 * special marker becomes ordinary tokens. Returns the token count, or -1 if it
 * exceeds cap or tokenisation fails. */
int s1_tokenize(const struct s1_engine *e, const char *text, int len, bool parse_special,
                int32_t *tok, int cap);

/* Token ids to text, special tokens rendered. Writes a terminating NUL. Returns the
 * byte count without the NUL, or -1 if it does not fit in cap. */
int s1_detokenize(const struct s1_engine *e, struct s1_span span, char *text, int cap);

/* True for tokens that mark structure instead of text, such as chat turn markers. */
bool s1_token_is_control(const struct s1_engine *e, int32_t tok);

/* Call 1. Clears the cache and processes the prefix as sequence 0. */
int s1_engine_state(struct s1_engine *e, struct s1_span prefix);

/* Call 2. Runs each suffix as its own branch of the current state, in as many groups
 * as the sequence and batch limits require. Writes out[j] for suffix[j]. The state
 * survives and may be branched again. */
int s1_engine_branches(struct s1_engine *e, const struct s1_span *suffix, int n,
                       struct s1_out *out);

/* Feature-extraction mode with shared states: n_prefix states in one call, then n_suffix
 * branches, branch j continuing prefix owner[j], in a second call. Every rotation of an
 * example reads its state once. Needs n_prefix + n_suffix sequences and one batch of
 * prefix tokens; the caller sizes the block. Discards any state held from s1_engine_state. */
int s1_engine_shared(struct s1_engine *e, const struct s1_span *prefix, int n_prefix,
                     const struct s1_span *suffix, const int *owner, int n_suffix,
                     struct s1_out *out);

/* Feature-extraction mode. Runs n whole sequences (prefix and suffix already joined)
 * that share nothing, packing as many as fit into each call. Writes out[j] for the last
 * token of seq[j]. Discards any state held from s1_engine_state. */
int s1_engine_packed(struct s1_engine *e, const struct s1_span *seq, int n,
                     struct s1_out *out);

/* ---- prompt: template file, prompt packing, slot table ---- */

/* The literal text a model's chat template puts around the state and the question. */
struct s1_template {
    char *open;  /* start of turn, system message, start of the user turn, "STATE\n" */
    char *close; /* end of the user turn, start of the assistant turn, "Answer:" */
};

int  s1_template_load(struct s1_template *t, const char *path);
void s1_template_free(struct s1_template *t); /* accepts a zeroed struct */

/* One question as the model sees it. option[k] is the description shown in slot k. */
struct s1_question {
    const char        *instructions;
    const char *const *option;
    int                K; /* 2 to S1_K_MAX */
};

/* OPEN + state + SEP, each piece tokenised on its own. Returns the token count or -1. */
int s1_prompt_prefix(const struct s1_engine *e, const struct s1_template *t,
                     const char *state, int32_t *tok, int cap);

/* Question body + CLOSE. The last token is the decision position. Returns the token
 * count or -1. */
int s1_prompt_suffix(const struct s1_engine *e, const struct s1_template *t,
                     const struct s1_question *q, int32_t *tok, int cap);

/* Token ids of the letters A to Z, with or without a leading space. Fails if any
 * letter is not exactly one token. */
int s1_slots_init(const struct s1_engine *e, bool leading_space, int32_t slot[S1_K_MAX]);

/* Log of the softmax restricted to the first K slots. */
void s1_slot_logprobs(const float *z, const int32_t *slot, int K, double *logp);

/* Probability the full-vocabulary softmax puts on the first K slots. Near 1 when the
 * model agrees that an option letter comes next. */
double s1_slot_mass(const float *z, int n_vocab, const int32_t *slot, int K);

/* ---- rows: the slot rows of the output matrix, from the GGUF file ---- */

struct s1_rows {
    float *w;       /* S1_K_MAX rows of n_embd floats, dequantised */
    int    n_embd;
    float  softcap; /* final logit soft-cap from the metadata; 0 when the model has none */
    bool   tied;    /* rows came from token_embd.weight: the model has no output.weight */
};

/* Reads the rows for the given slot token ids. Single-file GGUF models only. n_embd and
 * n_vocab come from the engine and are checked against the tensor's size. */
int  s1_rows_load(struct s1_rows *r, const char *gguf_path, int n_embd, int n_vocab,
                  const int32_t slot[S1_K_MAX]);
void s1_rows_free(struct s1_rows *r); /* accepts a zeroed struct */

/* dot(w[k], h): the uncapped slot logit, the quantity head H2 works on. */
double s1_rows_dot(const struct s1_rows *r, int k, const float *h);

/* The slot logit as the model reports it: the dot product through the soft-cap, if any.
 * Must equal libllama's logit for the slot token (test T2). */
double s1_rows_logit(const struct s1_rows *r, int k, const float *h);

/* ---- decide: questions to probabilities, with the label-free corrections ---- */

enum s1_type { S1_CHOICE, S1_BOOL, S1_SCORE };
enum s1_split { S1_TRAIN, S1_VALIDATION, S1_TEST, S1_HELDOUT };

/* One question of a request. For bool the options are "True" and "False"; for score they are
 * the level numbers. Strings are borrowed from the request. */
struct s1_ask {
    const char  *name;
    enum s1_type type;
    const char  *instructions;
    const char  *key[S1_K_MAX];    /* choice only: the caller's option keys */
    const char  *option[S1_K_MAX]; /* the text shown for each option, in the caller's order */
    int          K;
};

struct s1_reply {
    double p[S1_K_MAX];   /* probability per option, in the caller's order; sums to 1 */
    double slot_mass;     /* mean over rotations */
    int    n_rotation;    /* 1, or K when every cyclic rotation of the options was asked */
    int    top[S1_K_MAX]; /* the option ranked first under each rotation used, in order */
};

struct s1_decide_opts {
    bool rotations;     /* average over cyclic rotations of the options; always on for bool,
                           never for score */
    bool content_free;  /* subtract the slot logits obtained against the state "N/A" */
    int  max_rotations; /* at most this many rotations per question, evenly spaced over the
                           K orders; 0 means all K */
};

/* How many orders of a question's options are read: K for bool and, with opts.rotations, for
 * choice; 1 for score, whose levels always keep their natural order (level 1 in slot A, level
 * 2 in slot B, ...): an ordinal scale shown cyclically shifted (A = 3, B = 4, ...) is read
 * against its order, and the head's per-slot bias then cannot correct the model's preference
 * for some levels.
 * At most opts.max_rotations when that is positive. Extraction, evaluation and serving all
 * use this one rule. */
int s1_n_rotations(enum s1_type type, int K, struct s1_decide_opts opts);

/* The j-th of n rotations of a K-option question: all K when n == K, otherwise evenly spaced
 * so that the same rotations are chosen for the same K everywhere. Under rotation r, slot s
 * shows option (s + r) mod K. */
int s1_rotation_of(int j, int n, int K);

struct s1_timing {
    double state_ms;
    double questions_ms;
    int    n_state_tokens;
    int    n_question_tokens;
};

/* What the backbone gave for one rotation of one question. Under rotation r, slot s shows
 * option (s + r) mod K. */
struct s1_readout {
    int          ask;
    int          rotation;
    float        z[S1_K_MAX];  /* slot logits against the real state, as libllama returns them */
    float        zc[S1_K_MAX]; /* slot logits against the state "N/A"; zero unless content_free */
    float        slot_mass;
    const float *h;            /* n_embd floats at the decision position, then one n_embd block per
                                  captured layer (s1_engine_h_stride); owned by the set */
};

struct s1_readouts {
    struct s1_readout *item;
    int                n;
    float             *h_block; /* n x s1_engine_h_stride, backing every item's h */
};

struct s1_head; /* the trained head, declared with the head section below; NULL means H0 */

/* Runs the backbone: the state once, then every rotation of every question as a branch; with
 * content_free the same branches again on the state "N/A". On success the caller releases
 * `out` with s1_readouts_free. */
int  s1_read(struct s1_engine *e, const struct s1_template *t, const int32_t slot[S1_K_MAX],
             const char *state, const struct s1_ask *ask, int n_ask, struct s1_decide_opts opts,
             struct s1_readouts *out, struct s1_timing *timing);
void s1_readouts_free(struct s1_readouts *r); /* accepts a zeroed struct */

/* Feature-extraction form of s1_read: n examples that share nothing, ask[i] about state[i].
 * Each state runs once and its rotations branch from it, many examples per engine call. The
 * content-free branches all continue one state "N/A". */
int s1_read_packed(struct s1_engine *e, const struct s1_template *t,
                   const int32_t slot[S1_K_MAX], const char *const *state,
                   const struct s1_ask *ask, int n, struct s1_decide_opts opts,
                   struct s1_readouts *out, struct s1_timing *timing);

/* Probabilities over the K slots of one readout: softmax(z - zc) with no head, otherwise
 * the head's own use of z, zc and h. */
void s1_readout_probs(const struct s1_readout *item, const struct s1_head *head,
                      enum s1_type type, int K, double *p);

/* Readouts to probabilities. Per rotation: with no head, softmax(z - zc); with a head, the
 * head's own use of z, zc and h. Then slots back to options, and the mean over rotations.
 * Writes reply[i] for ask[i]. */
void s1_combine(const struct s1_ask *ask, int n_ask, const struct s1_readouts *r,
                const struct s1_head *head, struct s1_reply *reply);

/* s1_read followed by s1_combine. */
int s1_decide(struct s1_engine *e, const struct s1_template *t, const int32_t slot[S1_K_MAX],
              const char *state, const struct s1_ask *ask, int n_ask, struct s1_decide_opts opts,
              const struct s1_head *head, struct s1_reply *reply, struct s1_timing *timing);

/* ---- request: the JSON contract ---- */

struct yyjson_doc;

struct s1_request {
    struct yyjson_doc *doc; /* owns every string below */
    const char        *state;
    struct s1_ask     *ask;
    int                n_ask;
};

/* Fills one question from a JSON object with "type", "instructions" and "options" or
 * "levels". A request question and a training example both have this
 * shape. Strings are borrowed from the document. */
struct yyjson_val;
int s1_ask_parse(struct s1_ask *a, const char *name, struct yyjson_val *q);

/* The option index an answer names: an option key for choice, "true" or "false" for bool,
 * a level number for score. Returns -1 if the answer names no option. */
int s1_ask_option_index(const struct s1_ask *a, const char *answer);

/* One labelled training example. The strings belong to `doc`. */
struct s1_example {
    struct yyjson_doc *doc;
    const char        *id;
    const char        *task;
    const char        *family;
    const char        *state;
    int                split; /* enum s1_split */
    int                label; /* index of the correct option */
    struct s1_ask      ask;
};

/* Parses one line of an examples file. `where` names the line in error messages. */
int  s1_example_parse(struct s1_example *x, const char *text, long where);
void s1_example_free(struct s1_example *x); /* accepts a zeroed struct */

/* Parses one JSON document from the front of buf and sets *used to the bytes consumed, so
 * that a stream of requests can be read one after another. */
int  s1_request_parse(struct s1_request *r, char *buf, size_t len, size_t *used);
void s1_request_free(struct s1_request *r); /* accepts a zeroed struct */

/* Writes the response for one request as a single line of JSON. model_sha256 and head_sha256
 * are hex strings; head_sha256 is NULL when no head is loaded. */
int s1_response_write(FILE *out, const struct s1_request *r, const struct s1_reply *reply,
                      const struct s1_timing *timing, const char *model_sha256,
                      const char *head_sha256);

/* ---- metrics: the evaluation metrics. Pure. ---- */

/* One item's prediction: a probability per option, in the caller's option order. */
struct s1_pred {
    double   p[S1_K_MAX];
    int      K;
    int      label;     /* index of the correct option */
    int      type;      /* enum s1_type */
    double   slot_mass; /* mean over the rotations used */
    uint64_t id_hash;
    uint32_t task_id;
    uint16_t family_id;
};

enum s1_metric {
    S1_M_ACCURACY,      /* fraction of items whose top option is correct */
    S1_M_LOG_LOSS,      /* mean of -log p[label]; the primary metric */
    S1_M_BRIER,         /* mean of sum_k (p[k] - t[k])^2 */
    S1_M_ECE,           /* ten equal-width bins on the top probability */
    S1_M_SLOT_MASS,     /* mean */
    S1_M_SLOT_MASS_P05, /* 5th percentile */
    S1_M_RPS,           /* ranked probability score, score items only; NaN when there are none */
    S1_N_METRICS
};

extern const char *const S1_METRIC_NAME[S1_N_METRICS];

/* The metrics of pred[index[0]], ..., pred[index[n-1]]; index NULL means all n in order. */
void s1_metrics(const struct s1_pred *pred, const int *index, int n, double m[S1_N_METRICS]);

/* 95% percentile intervals from `resamples` bootstrap resamples of the items. */
int s1_metrics_bootstrap(const struct s1_pred *pred, int n, int resamples, uint64_t seed,
                         double lo[S1_N_METRICS], double hi[S1_N_METRICS]);

#define S1_N_BINS 10

struct s1_bin {
    int    n;
    double confidence; /* mean top probability */
    double accuracy;
    double wilson_lo;  /* 95% Wilson interval of the accuracy */
    double wilson_hi;
};

/* The reliability table: items binned by their top probability, [0, 0.1) to [0.9, 1]. */
void s1_reliability(const struct s1_pred *pred, int n, struct s1_bin bin[S1_N_BINS]);

/* Accuracy among the items whose top probability exceeds the threshold, and the fraction of
 * items that qualify. accuracy is NaN when none qualifies. */
void s1_risk_coverage(const struct s1_pred *pred, int n, double threshold, double *accuracy,
                      double *coverage);

/* ---- args: command-line arguments of the tools ---- */

/* Fails, naming the argument, unless every argument is one of `options` (each followed by a
 * value) or one of `flags`. Both lists end with NULL. */
int         s1_args_check(int argc, char **argv, const char *const *options,
                          const char *const *flags);
const char *s1_arg_value(int argc, char **argv, const char *name); /* NULL when absent */
bool        s1_arg_flag(int argc, char **argv, const char *name);

/* ---- sha256: identity of model, template, feature and head files ---- */

/* SHA-256 of a whole file as lower-case hex. */
int s1_sha256_file(const char *path, char hex[S1_SHA256_HEX]);

/* ---- feat: the feature file ----
 *
 * Little-endian, written field by field, no padding: a 96-byte header, 26 int32 slot token
 * ids, then fixed-size records, each followed by its n_embd floats of h. One record per
 * rotation of one example. */

#define S1_FEAT_HAS_ZC 2u /* header flag bit 1: records carry content-free logits */

struct s1_feat_header {
    uint32_t n_embd;
    uint32_t flags;
    uint64_t n_records;
    char     gguf_sha256[S1_SHA256_HEX];
    char     template_sha256[S1_SHA256_HEX];
    int32_t  slot[S1_K_MAX];
};

struct s1_feat_record {
    uint64_t id_hash;   /* s1_hash64 of the example id; shared by the rotations of one item */
    uint32_t task_id;   /* s1_hash64 of the task name, low 32 bits */
    uint16_t family_id; /* s1_hash64 of the family name, low 16 bits */
    uint8_t  type;      /* enum s1_type */
    uint8_t  split;     /* enum s1_split */
    uint8_t  K;
    uint8_t  label;     /* slot of the correct option under this rotation */
    uint8_t  rotation;
    uint8_t  perm[S1_K_MAX]; /* perm[slot] = original option index */
    float    target[S1_K_MAX];
    float    z[S1_K_MAX];
    float    zc[S1_K_MAX];
    float    slot_mass;
};

/* FNV-1a, 64 bits: stable ids from names, the same in every file and on every machine. */
uint64_t s1_hash64(const char *text);

struct s1_feat_writer {
    FILE    *file;
    uint32_t n_embd;
    uint64_t n_records;
};

int s1_feat_create(struct s1_feat_writer *w, const char *path, const struct s1_feat_header *h);
int s1_feat_append(struct s1_feat_writer *w, const struct s1_feat_record *rec, const float *h);
int s1_feat_close(struct s1_feat_writer *w); /* writes the record count into the header */

/* A whole feature file in memory. */
struct s1_feat {
    struct s1_feat_header  header;
    struct s1_feat_record *rec; /* header.n_records */
    float                 *h;   /* header.n_records x header.n_embd */
};

int  s1_feat_load(struct s1_feat *f, const char *path);
void s1_feat_free(struct s1_feat *f); /* accepts a zeroed struct */

/* Sets every record's slot logits (z and zc) to zero, so that a head reads h alone: the probe
 * of an intermediate layer, as opposed to its correction of the final-layer letter scores. */
void s1_feat_drop_letters(struct s1_feat *f);

/* ---- lbfgs: the optimiser. Pure: arrays in, arrays out. ---- */

/* Returns f(x) and writes its gradient to g. */
typedef double (*s1_objective)(const double *x, double *g, int n, void *ctx);

/* Called after every accepted step; a non-zero return stops the optimiser early. */
typedef int (*s1_progress)(const double *x, int iter, double f, void *ctx);

struct s1_lbfgs_opts {
    int         max_iter; /* 500 by default */
    double      g_tol;    /* stop when the infinity norm of the gradient falls below this */
    s1_progress progress; /* may be NULL */
    void       *progress_ctx;
};

struct s1_lbfgs_result {
    int    n_iter;
    double f;
    double g_inf;
    bool   converged; /* g_tol reached, as opposed to max_iter, early stop or a failed search */
};

/* Minimises f from the starting point in x, in place. Full-batch L-BFGS with 10 correction
 * pairs and Armijo backtracking. Returns -1 only if memory cannot be allocated. */
int s1_lbfgs(s1_objective f, void *ctx, double *x, int n, const struct s1_lbfgs_opts *opts,
             struct s1_lbfgs_result *result);

/* ---- head: H1 and H2. Pure. ----
 *
 *   u[k] = a * (z[k] + dot(d[k], h)) - c * zc[k] + b[k],   a = exp(theta),   p = softmax(u)
 *
 * H1 has d = 0. Parameters live in one array: theta, c, b[S1_K_MAX], then for H2
 * d[S1_K_MAX][n_embd]. All zeros is the identity head: it reproduces H0. */

#define S1_HEAD_THETA 0
#define S1_HEAD_C     1
#define S1_HEAD_B     2
#define S1_HEAD_D     (2 + S1_K_MAX)
#define S1_HEAD_N_EMBD_MAX 65536 /* largest hidden size a head file may declare */

/* Records of one question type as parallel arrays, the form the trainer reads. */
struct s1_records {
    int            n;
    int            n_embd;
    const float   *h;      /* n x n_embd: the vector at the decision position */
    const float   *z;      /* n x S1_K_MAX: uncapped slot logits */
    const float   *zc;     /* n x S1_K_MAX: content-free slot logits, or all zero */
    const float   *target; /* n x S1_K_MAX: target distribution over slots */
    const uint8_t *K;      /* n: slots in use */
    const float   *weight; /* n: weight of each record in the mean loss, or NULL for all 1 */
};

struct s1_fit {
    const struct s1_records *data;
    bool                     h2;        /* train the row corrections d */
    bool                     fit_c;     /* false keeps c at its starting value: no zc in the data */
    double                   lambda;    /* penalty lambda * sum(d * d) */
    bool                     fix_theta; /* keep theta at its starting value (a bounded fit) */
};

/* Guardrails of the fit. The temperature exp(-theta) of a fitted head is held within
 * [S1_TEMP_MIN, S1_TEMP_MAX]: a larger one means the head has all but discarded the slot
 * logits (a -> 0), which also freezes H2, whose row gradients are multiplied by a. H2 starts
 * from H1 with its temperature capped at S1_H2_START_TEMP. A fitted type whose probabilities
 * hardly vary across the validation records (largest standard deviation of any option's
 * probability below S1_MIN_PROB_SD), or whose validation loss is not below the raw readout's,
 * is replaced by the identity head (H0). */
#define S1_TEMP_MIN      0.05
#define S1_TEMP_MAX      100.0
#define S1_H2_START_TEMP 20.0
#define S1_MIN_PROB_SD   1e-3

/* theta moved into the bounds, so that exp(-theta) lies in [S1_TEMP_MIN, S1_TEMP_MAX]. */
double s1_theta_clamp(double theta);

/* Turns a fitted H1 parameter array into H2's starting point: the temperature is capped at
 * S1_H2_START_TEMP, everything else is kept. */
void s1_head_h2_start(double *x);

/* The largest, over option positions k, standard deviation of p[k] across the records (those
 * with K > k): 0 for a head whose output does not depend on its input. */
double s1_head_prob_sd(const double *x, bool h2, const struct s1_records *data);

/* A trained head: one parameter array per question type, valid for one GGUF file, one
 * template and one slot table only. */
struct s1_head {
    bool    h2;
    int     n_embd;
    double *x[3]; /* indexed by enum s1_type; s1_head_n_param(h2, n_embd) doubles each */
    char    gguf_sha256[S1_SHA256_HEX];
    char    template_sha256[S1_SHA256_HEX];
    int32_t slot[S1_K_MAX];
};

int s1_head_n_param(bool h2, int n_embd);

/* Probabilities of one record under head x. h may be NULL for H1. */
void s1_head_apply(const double *x, bool h2, int n_embd, const float *z, const float *zc,
                   const float *h, int K, double *p);

/* The s1_objective of a fit: mean cross-entropy plus the penalty. ctx is a struct s1_fit. */
double s1_head_loss(const double *x, double *g, int n, void *ctx);

/* Mean cross-entropy of head x on the records (weighted when data->weight is set), without the
 * penalty: the validation loss. */
double s1_head_mean_loss(const double *x, bool h2, const struct s1_records *data);

/* ---- headfile: a trained head on disk ----
 *
 * Little-endian: magic, version, head type, n_embd, the two SHA-256 values, the 26 slot ids,
 * then per question type the parameter array as float64. */

/* Allocates identity parameters (all zero, which reproduces H0) for every question type. */
int  s1_head_init(struct s1_head *head, bool h2, int n_embd);
void s1_head_free(struct s1_head *head); /* accepts a zeroed struct */
int  s1_head_save(const struct s1_head *head, const char *path);
int  s1_head_load(struct s1_head *head, const char *path);

/* Fails, saying why, unless the head was trained for exactly this model file, template file
 * and slot table. */
int s1_head_check(const struct s1_head *head, const char *gguf_sha256,
                  const char *template_sha256, const int32_t slot[S1_K_MAX]);

#endif
