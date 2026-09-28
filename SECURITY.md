# Security

## Reporting a vulnerability

Please report security problems privately through GitHub Security Advisories: on
[github.com/judgly/judgly](https://github.com/judgly/judgly), open the **Security** tab and
choose **Report a vulnerability**. Do not open a public issue for them. Reports are
acknowledged as time allows; judgly is a small hobby project maintained by one person.

Only the latest release is supported.

## What judgly trusts and what it does not

- **The state and the questions are untrusted text.** judgly places them in a prompt and reads
  the model's answer. Text in the state can steer the answer, for example by instructing the
  model to pick an option ("prompt injection"). judgly does not detect or prevent this. Do not
  use an answer as the only safeguard for a decision that an attacker who controls the text
  would want to change.
- **The heads are calibrated on benign public data.** The reported calibration says nothing
  about text written to mislead the model; on such text the probabilities can be confidently
  wrong.
- **Model files, packs and heads are code-adjacent inputs.** A GGUF file, a pack directory, a
  template or a head file is parsed by native code (llama.cpp and libjudgly). Load only files
  you trust. `Engine.load` checks the model file's size and SHA-256 against the pack, and each
  head file against the SHA-256 in `pack.json`; a pack you did not write can name any file.
- **Some paths skip verification.** `Engine(config)` loads whatever model, template and heads
  the configuration names, without a pack, and a configuration that gives `model_sha256` is
  trusted to have that hash: the file is not hashed. Use these only with files you have
  verified yourself.
- **Input limits.** A request larger than `max_request_bytes` (default 4 MiB) or with more than
  `max_questions` questions (default 256) is refused before it is processed; a state longer
  than `max_state_tokens` (default 16,384 tokens) is truncated and the response says so; a
  question whose instructions and options take more than 4,096 tokens is refused, and so is a
  request that does not fit the context. Each `Engine` answers one request at a
  time, so a service that exposes judgly should still limit request rates and sizes itself.
