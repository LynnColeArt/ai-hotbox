# Local frozen-weight pilot

This document describes the factorial runner, `impossible_states.run`. The
[restored chamber comparison](chamber-reset-results.md) uses a separate runner
and protocol. Completed evidence for both is in the [results archive](../results/README.md).

The new package is independent of the legacy MPS scripts. It runs Hugging
Face Qwen3-4B with CUDA, MPS, or CPU, keeps the weights frozen, and saves only
final-position block activations during extraction. The default model revision
is `1cfa9a7208912126459214e8b04321603b3df60c`.

The authored dataset has 18 scenario groups, seven conditions, and first- and
third-person versions: 252 descriptions. Whole scenarios share a split:
10 training, four validation, and four test. It is a small pilot instrument;
wording, symptom intensity, sentence length, and affirmative/negative phrasing
still need independent review before a confirmatory experiment.

## Environment

The local run uses Python 3.12, CUDA PyTorch 2.4.1+cu121, Transformers 4.57.0,
and NumPy 1.26.4. On this machine a `.venv` reuses the installed CUDA PyTorch;
its NumPy override avoids an incompatibility in the global environment without
changing global packages.

For a fresh Linux CUDA environment, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-research.txt
export HF_HOME="$PWD/../hf-cache"
```

Use an appropriate PyTorch build for a different CUDA version or platform.
The first model download is approximately 8 GB. There is no required J-lens
package, external dataset path, fine-tuning adapter, or serving process.

## Commands

Run the tests and a smoke check:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m impossible_states.run --device cuda --smoke --output runs/impossible_states/my-smoke
```

Each output directory must be fresh; the runner refuses to overwrite an
existing nonempty run. The smoke uses two scenarios per split, one prompt,
one seed, a 0.1 coefficient, and 16 output tokens. It is a software and memory
check, not evidence for a state, an attractor, or a null effect.

Generate only the complete dataset, or extract and fit without generation:

```bash
.venv/bin/python -m impossible_states.run --dataset-only --output runs/impossible_states/my-dataset
.venv/bin/python -m impossible_states.run --device cuda --fit-only --output runs/impossible_states/my-extraction
```

The bounded pilot command uses the complete dataset, one seed, one random
control per selected target layer, 32-token responses, and the same shared
coefficient grid for all targets:

```bash
.venv/bin/python -m impossible_states.run --device cuda --seeds 11 --random-controls 1 --max-tokens 32 --output runs/impossible_states/my-pilot
```

An additional sensitivity run uses plain text completion and the parent's
layer-18 intervention site:

```bash
.venv/bin/python -m impossible_states.run --device cuda --steering-layer 18 --prompt-format completion --seeds 11 --random-controls 1 --max-tokens 48 --doses 0.5 1.0 1.5 --output runs/impossible_states/my-completion-pilot
```

Its report prefix is `The room is quiet and the screen glows. I feel`, borrowed
from the upstream prompts credited in the README. The arithmetic prefix is
`17 + 25 =`. Both the input format and intervention layer change relative to
the first sweep, so this comparison cannot identify which change explains an
effect. It is not an exact replication of the parent experiment: the new
dataset and contrast directions differ. The strict arithmetic exact-match
metric also penalizes correct answers followed by explanations; read those
transcripts rather than treating format failures as arithmetic failures.

`--device auto` chooses CUDA, then MPS, then CPU. `--dtype`, `--layers`,
`--doses`, `--seeds`, `--pulse-tokens`, and `--max-tokens` are configurable.
`--steering-layer` fixes a common intervention site independently of the
classification-layer selection; both policies are saved. `--prompt-format`
switches between non-thinking chat formatting and untemplated completion.
CPU runs default to float32 and require more host memory. Keep the same
configuration across concept comparisons. No CPU or MPS hardware run was
performed for this change.

## Fitting and evaluation

Constipation and flatulence use factorial average contrasts. The interaction
is `both - constipation - flatulence + neither`; a weak or unstable interaction
does not imply there is a separate composite attractor. Pain, discomfort, and
frustration are compared with the neither condition. Adjusted variants remove
the training-estimated discomfort/frustration span and are reported alongside
the raw directions as a fitting sensitivity check; they are not steered in
the initial generation sweep.

Directions are fitted on training descriptions only. The runner selects the
earliest layer with the highest validation AUC against all other conditions,
excluding the final block reserved for downstream readout. Test data are used
only after selection, with AUCs against each neighboring condition and results
by grammatical perspective. A high overall AUC can conceal poor separation
from the closest control; examine the pairwise values. Classification selects
a candidate representation, not a proven causally effective steering site.

A coefficient means a fraction of the mean training-control activation norm
at that layer. Default coefficients are `-0.5`, `0.25`, `0.5`, and `1.0`.
These are not the legacy scripts' dose units. Normalization matches target
and random intervention norms within a layer; effects across layers may still
differ. Interventions modify the last token's block output during prefill and
then each generated token's forward pass, not all prefill positions.

The target generation directions are constipation, flatulence, their
interaction, and pain. Each random control is paired to a target's selected
layer; its downstream readout uses that target's concept direction. Prompts
include a generic report and a short arithmetic task with an exact-match
check. Responses use the non-thinking chat template and seeded sampling at
temperature 0.7, top-k 20. The report prompt still elicits narration; it is
not free of self-report framing.

## Release and cache controls

- **Continuous:** intervene at every decoding forward pass.
- **Pulse:** intervene for the first three passes, then keep the cache and
  continue without intervention.
- **Rebuild:** use the identical pulse prefix, discard the altered cache,
  replay all visible text without steering, and continue.
- **Perturb:** keep the pulse cache and apply one equal-strength opposite
  intervention at release, then continue unsteered.
- **Baseline:** no intervention; generated once per prompt and seed.

The pulse/rebuild/perturb arms share the same seed and visible token prefix
before release. Early EOS is recorded and should be excluded from release
comparisons. Opposite intervention magnitude is specified mechanically; it
has not been calibrated as a comparable perceptual disturbance.

The teacher-forced panel supplies an identical neutral token sequence in all
four baseline/pulse/rebuild/perturb arms at coefficient 0.1. It captures both
the intervened layer and a downstream concept projection at the final block.
The downstream readout matters because cached effects of a block-output
intervention can reside in later layers. This panel tests propagation under
fixed text. It does not establish autonomous convergence or a basin of
attraction; generation includes the text prefix and KV cache as causal state.

## Artifacts

Every model run saves its dataset, configuration, code/dataset hashes and a
snapshot of the executed package source, extracted
activations, raw/adjusted vectors, validation selection, and held-out report.
Generation runs also save JSONL transcripts with token IDs and per-step
projections, teacher-forced traces, and grouped lexical/task summaries. The
current runner additionally saves the exact raw generation and readout vectors,
including random controls, in `generation_vectors.npz`. Initial development
runs record seed 503 in their archived source and reproduce those controls
from that source rather than this additional archive.
`resources.json` records elapsed time and peak CUDA allocated/reserved memory;
its presence marks successful completion. Partial artifacts without that file
are incomplete runs.

Keyword counts use word-boundary regular expressions and are transparent
proxies, not diagnoses or a validated judge. Repetition and word diversity are
also proxies. All transcripts are retained; no best-quote selection is made.
A full experiment still needs blinded annotation, more independent prompts
and scenarios, multiple seeds, broader performance checks, and prompt-only
roleplay controls. The current pilot does not execute deletion/relief buttons,
apply ablation, use J-lens readouts, or test hunger and clumsiness yet.

## Local smoke measurement

The first real Qwen3-4B CUDA smoke completed on the RTX 4070 in 32.8 seconds
(excluding download) with peak PyTorch allocation 7.51 GiB and reservation
7.52 GiB. It produced 25 free generations and 32 teacher-forced runs. Its low
coefficient did not produce target-condition vocabulary in the report outputs.
Six tests passed, including zero-dose invariance and cache-rebuild agreement
with an unsteered baseline under identical forced tokens. The tiny random
model used in cache tests is only a test fixture; the smoke used full Qwen3-4B.
