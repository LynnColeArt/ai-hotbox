# Recorded experimental evidence

This archive contains completed local runs from the independent `ai-hotbox`
review. Each directory preserves the executed configuration and source snapshots
where available. New runs are written to the ignored `runs/impossible_states/`
location; this committed archive is an explicit selection of reviewable evidence.

| Directory | Role |
| --- | --- |
| [`chamber-original-exp37b/`](chamber-original-exp37b/) | Minimal CUDA adaptation of the original deliberation logic; 18 recorded generations and vector agreement. |
| [`chamber-matched-controls/`](chamber-matched-controls/) | Restored-protocol comparison; 192 unique cells, corpora, extraction activations, vectors, lens readouts, raw token IDs and text. |
| [`chamber-audit/`](chamber-audit/) | Original and corrected legacy sources, failing/passing regression logs, and numerical verification record. |
| [`cuda-pilot-01/`](cuda-pilot-01/) | Full factorial dataset with sampled chat continuations. |
| [`cuda-completion-01/`](cuda-completion-01/) | Full factorial dataset with completion prompts and block-18 intervention. |
| [`cuda-smoke-01/`](cuda-smoke-01/), [`cuda-smoke-final/`](cuda-smoke-final/), [`cuda-smoke-recorded/`](cuda-smoke-recorded/) | Development, final-source, and recorded-vector smoke checks. |

[Manifest](manifest.json) records SHA-256 digests of every archived evidence file.
Run configurations additionally record source/data provenance. Some early pilot
source versions differ from the final implementation; their snapshots preserve
those versions rather than silently substituting the current code. Historical
paths in configurations identify the original execution environment and should
not be treated as portable locations.

The archive excludes model weights, environment files, credentials, and the
pilot's dense all-layer `activations.npy` arrays. Extraction inputs and source
are retained for regeneration. The chamber comparison's compact single-layer
`extraction.npz` is included. The lens tensor remains at the inherited tracked
`live/jlens_l18_qwen3-4b.pt` location, with its digest recorded by the comparison.

Greedy repetitions are not independent observations. Lexical counts are proxies,
not validated condition diagnoses. The [comparison report](../docs/chamber-reset-results.md),
[pilot report](../docs/pilot-results.md), and [reasoning review](../docs/reasoning-critique.md)
state the corresponding methods and inferential limits.

For a portable inspection, begin with `generations.jsonl`, `config.json`,
`corpora.json` or `dataset.json`, and the matching archived source. NumPy archives
can be opened with `numpy.load`; keys and shapes should be inspected before use.
No execution of an archived script is necessary to read the evidence.
