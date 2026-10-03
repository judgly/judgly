# Decisions after freezing (the frozen files are unchanged)

- 2026-09-29 16:54: Verified from the sources that Ollama v0.35.0's /v1/systemone builds
  {"context", "schema"} prompts (decision/systemone.go), while Tev1 was trained on
  {"state", "question", "options": [{"label", "key", "description"}]} (Tev1-4B-experimental
  model card). A second "native prompt" run for Tev1 was considered and declined by the person:
  every model is tested only through the systemone endpoint, and the mismatch is reported as a
  limitation of Tev1 as served by Ollama. Decided before any Tev1 result was looked at.

- 2026-10-03 16:02: A fourth external model, `clef:27b` (Ollama ID 2bb11a61d1fb, the registry's
  current ID for the tag; 27.0B, Qwen3.5 architecture, base model given as "Qwen3.8 27B",
  Q4_K_M, context 16,384 tokens as served, Apache 2.0), is added to the comparison by the
  person's decision, folded into the existing record: same test items, same request mapping and
  same frozen runner and scorer (unchanged; the scorer treats a model it does not list as having
  no context limit), and the same equal-calibration control on the same 2,000 training items.
  Written before clef answered any item.
  clef needs Ollama 0.35.1. Under 0.35.1, `/v1/systemone` refuses nimble:9b, tev1:4b and
  tev1:0.8b (HTTP 400 "does not support decision": 0.35.1 checks a `decision` capability that
  their manifests, pulled under 0.35.0, do not declare; 200 of 200 tev1:0.8b spot-check
  requests refused). Their tags have since moved on the registry (tev1:0.8b 8d11b3146b7f,
  tev1:4b 9b5bb969e46c, nimble:9b 9b953de7a533), so re-pulling would not give the recorded
  weights and was not done. The three models' committed answers (Ollama 0.35.0) are kept; clef
  is answered under 0.35.1. Whether 0.35.0 and 0.35.1 serve the same weights identically could
  not be checked; this is stated as a limitation.

- 2026-10-04 06:43 (after Clef's results were scored): Clef's provenance, looked up in
  the Ollama library page (https://ollama.com/library/clef), Cloudflare's model card
  (https://huggingface.co/Cloudflare/clef, with its LICENSE), Cloudflare's announcement
  (https://blog.cloudflare.com/clef-decision-models/, 2026-10-01) and its Workers AI page
  (https://developers.cloudflare.com/workers-ai/models/clef/).
  Clef is Cloudflare's: a 27B decision model post-trained from Qwen/Qwen3.8-27B (Qwen team,
  Alibaba Cloud), released under Apache 2.0 (the LICENSE in Cloudflare's repository is the
  Apache 2.0 text with "Copyright 2026 Alibaba Cloud", as is the licence layer of the Ollama
  manifest). Cloudflare describes its training data only as internal synthetic datasets; no
  dataset is named. Its model card reports an internal run of the Decision Index 0.2.1 suite,
  which includes Humicroedit and Amazon ESCI: Humicroedit is the whole `humour` family of
  judgly's general confirm tier (500 items) and Amazon ESCI the whole `relevance` family of the
  general final tier. The suite also includes BANKING77 and MMLU, two sources of the control's
  training sample (18 and 19 items). Whether Cloudflare's evaluation items are the same as
  judgly's, or whether any of these benchmarks informed Clef's training or model selection, is
  not stated. Nothing in the record was changed; the overlap is reported as a caveat.
  Cloudflare's release is safetensors with a separate joint schema head and a 64K context; Ollama
  serves a Q4_K_M GGUF with a 16,384-token context (Cloudflare's own default `max_length` is also
  16,384). Whether Ollama's package scores exactly as Cloudflare's release does was not checked.

- 2026-10-04: The current registry tags of the first three models were pulled under Ollama 0.35.1
  (`nimble:9b` `9b953de7a533`, `tev1:4b` `9b5bb969e46c`, `tev1:0.8b` `8d11b3146b7f`). They declare
  the `decision` capability and each answered a one-question test request. They are other
  weights than the record's and were not used for any result; the recorded versions are no
  longer on this machine.
