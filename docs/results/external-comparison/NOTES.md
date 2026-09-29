# Decisions after freezing (the frozen files are unchanged)

- 2026-09-29 16:54: Verified from the sources that Ollama v0.35.0's /v1/systemone builds
  {"context", "schema"} prompts (decision/systemone.go), while Tev1 was trained on
  {"state", "question", "options": [{"label", "key", "description"}]} (Tev1-4B-experimental
  model card). A second "native prompt" run for Tev1 was considered and declined by the person:
  every model is tested only through the systemone endpoint, and the mismatch is reported as a
  limitation of Tev1 as served by Ollama. Decided before any Tev1 result was looked at.
