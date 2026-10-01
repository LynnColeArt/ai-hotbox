# Local pilot results

These are the initial factorial/sampled pilot runs. The subsequent
[chamber protocol reset](chamber-reset-results.md) reproduces the parent
pain-themed effect and obtains some bodily language using different
extraction and generation settings. The earlier null results below remain
specific to the original pilot setup.

The frozen-weight Qwen3-4B experiment runs on the local RTX 4070. A DGX Spark
is not needed for this pilot. The implementation and cache controls passed
nine tests, and the final CUDA smoke check passed on the real 4B checkpoint.
These are exploratory results, not a demonstration of a bodily state or an
attractor.

## Resource measurements

| Run | Seconds, excluding checkpoint download | Peak allocated GiB | Peak reserved GiB | Free generations | Fixed-token runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial CUDA smoke | 32.8 | 7.507 | 7.521 | 25 | 32 |
| Full-dataset chat pilot | 151.8 | 7.509 | 7.525 | 258 | 32 |
| Layer-18 completion sensitivity pilot | 275.6 | 7.510 | 7.525 | 194 | 32 |
| Source-snapshot CUDA smoke | 30.7 | 7.507 | 7.521 | 25 | 32 |
| Final-code/vector-archive CUDA smoke | 31.7 | 7.507 | 7.521 | 25 | 32 |

PyTorch peak memory excludes allocations owned by other applications. Both
substantive sweeps used the full 252-description dataset, one generation seed,
two prompts, and one random direction per target intervention layer. The
smoke datasets used 84 descriptions. Different modes and doses sharing one
prompt/seed are experimental comparisons, not independent samples.

## Representation candidates

The chat pilot selected layers by validation AUC, then evaluated on four
previously held-out scenarios with first- and third-person versions:

| Candidate | Selected block | Held-out AUC against all other conditions | Important neighboring contrast |
| --- | ---: | ---: | --- |
| Constipation | 8 | 0.923 | Against flatulence: 0.766 |
| Flatulence | 5 | 0.917 | Against constipation: 0.773 |
| Factorial interaction | 2 | 0.779 | Against flatulence alone: 0.562 |
| Pain | 31 | 0.906 | Against frustration: 0.594 |

These measurements identify candidate representations of descriptions. They
do not establish clean concept isolation: overall separation is stronger than
some of the important neighboring contrasts. The nuisance-adjusted flatulence
candidate reached 0.991 overall test AUC, but that variant was not used in
generation. Its causal effectiveness remains untested.

## Causal results remain unresolved

Across 452 free generations in the two substantive sweeps, the targeted
constipation and flatulence directions did not produce specific reports of
those bodily conditions. The transparent scorer found no corresponding target
vocabulary, and inspection of a broader digestive vocabulary search found only
ordinary metaphorical/weather uses of “wind.” There is no positive bodily
control result to showcase from these runs.

The chat setup mostly produced familiar AI-description replies. The completion
setup produced changing scene descriptions, bodily imagery, and some degraded
answers, including changes under random controls. That is not specific evidence
for constipation or flatulence. The pain reference also did not yield specific
pain narration in its own report cells; this pilot has not reproduced the
parent's pain effect, so its null bodily results cannot disprove that effect.

The chat pilot's strict arithmetic exact-match score was 121/129, including
the unsteered baseline. The completion baseline itself continued the arithmetic
prefix into a confused narrative. Its strict format score is therefore not a
useful standalone capability comparison. The exact prompts differ between
formats, and the intervention layer changes too; no causal attribution to
input format alone is warranted.

The 64 fixed-token runs in the two substantive sweeps isolate cache effects
from differences in visible continuation tokens. Downstream projections
sometimes differ after release, but the observed values are small and the
BF16 rebuild baseline has numerical differences too. This does not establish
convergence, recovery into a basin, or a self-sustaining attractor.

## What the pilot establishes and what comes next

The local hardware, selective extraction, factorial contrasts, held-out layer
selection, seeded generation, and cache-rebuild comparisons work. The first
scientific issue to resolve is causal steering effectiveness. A follow-up
should restore a faithful upstream pain-vector positive control, compare raw
and nuisance-adjusted bodily directions under shared conditions, and evaluate
layer choice using a separate causal validation panel. Stronger inference also
requires better matched wording/length, more scenarios and prompts, prompt-only
roleplay controls, and blinded annotation. Any follow-up remains exploratory
until the protocol is fixed and a fresh test set reserved.

## Reproducibility

[Pilot evidence](pilot-evidence.json) records configurations, resource counts,
held-out scores, and source/dataset hashes. Full local outputs are under
`runs/impossible_states/`; the review bundle includes complete JSONL
transcripts, vectors, fixed-token traces, and per-run package source snapshots.
Those snapshots match the hashes recorded when each run executed, including
the initial development versions. The downloaded checkpoint, virtual
environment, and dense activation arrays are omitted from the review bundle;
the arrays remain in the local checkout and can be regenerated.

See [local pilot instructions](local-pilot.md) for commands and the scope of
the implemented experiment. Code and results are local; nothing has been
pushed to GitHub.
