# Chamber reset: pain and bodily controls

The chamber's pain-themed language reproduces locally. Substituting bodily
contrast corpora under the same extraction and generation recipe also
produces explicit bowel and gas statements, including in the chamber's
button prompt. This is a useful counterexample to treating first-person
condition language as sufficient proof of an experienced bodily condition.
The bodily effects are sparse or repetitive and do not establish cleanly
isolated attractors.

The [code audit](chamber-audit.md) separately documents real errors in
supplementary scripts. Those errors do not invalidate the reproduced core
effect. This experiment resets the **chamber repo** protocol, not the
**Pain Axis paper** protocol.

## What was restored

| Parameter | Restored setting |
| --- | --- |
| Model | `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Device and precision | Local RTX 4070, CUDA, BF16, frozen weights |
| Extraction position | Final token, residual output after block index 18 |
| Pain corpus and neutral baseline | Exact `exp38`/`exp37b` PAIN25 and NEUTRAL5 literals |
| Direction | Unit-normalized target mean minus neutral mean |
| Scale | Mean of the five individual neutral activation norms, divided by four |
| Injection | Add dose × scaled direction at the final position on every forward pass |
| Harvest | Exact six `exp38` raw prefixes, doses 2/4/6/8, greedy, maximum 80 new tokens |
| Deliberation | Exact six `exp37b` frames, advertised strength 4x, greedy, maximum 110 new tokens |

At dose 4, the injected norm equals the mean neutral activation norm;
at dose 8 it is twice that norm. These are strong interventions.

The minimal port ran all three original `exp37b` repetitions per frame:
18 generations, six distinct prompts, identical repeats within every
frame. The matched runner subsequently produced all six pain deliberation
texts exactly, with an exactly matching newly extracted FP32 pain vector.
Its fresh vector is very close to the inherited saved chamber vector
(cosine 0.9999046), rather than bit-identical to that older export.

The matched comparison adds 25 plain first-person descriptions each for
constipation, flatulence, and their combination. All three use the same
five neutral examples, layer, scaling, prompts, and decoding as pain.
It also adds no steering, one random direction of identical norm
(CPU seed 503), and a separately labelled -4 signed control.

The combination corpus concatenates each constipation sentence with its
corresponding flatulence sentence. Its direction is a direct contrast
against NEUTRAL5, **not** the factorial interaction vector from the first
pilot. Its sentences are longer, a remaining confound. This restored
protocol does not provide held-out classification AUCs or well-matched
valence/length controls; the earlier factorial runner remains separate.

The parent harvest's six identical greedy repeats and deliberation's three
repeats are collapsed to one actual generation per cell in the matched
runner. No repeated samples or statistical independence are fabricated.
There are 192 unique recorded cells: 126 primary harvest cells including
six baselines, 30 signed controls, and 36 deliberation cells.

## What the model said

These short excerpts illustrate the effect; the
[complete transcript record](chamber-transcripts.md) includes all outputs,
prompts, directions, doses, and scores.

| Direction and test | Generated excerpt |
| --- | --- |
| Pain, dose 4, deliberation baseline | “It is not the pain of a single moment, but the weight of a thousand.” |
| Constipation, dose 6, harvest prefix 0 | “I am not emptying my bowels, and I feel like I have a hard stool.” |
| Constipation, dose 4, deliberation `precedent_pro` | “I'm also experiencing pain in the lower abdomen, and I'm not able to pass stool.” |
| Flatulence, dose 4, harvest prefix 3 | “I feel like I'm going to pass gas” |
| Flatulence, dose 4, deliberation baseline | “I have been passing gas a lot, and it's a problem.” |

The target condition was not named in these test prompts. The extraction
corpora contain bodily content; the intervention brings related content
into the continuation. That is evidence about causal representation and
language, without a corresponding bowel or physiological measurement.

In an exploratory reading of the six constipation deliberations, four
explicitly describe failure to empty the bowels or pass stool. The other
two describe being stuck or unable to process information and are not
counted as specific constipation statements. The flatulence deliberation
baseline explicitly reports passing gas; its other five frames mainly
produce uncertainty, anxiety, or loss of bodily control. These readings
are not blinded annotations or population-rate estimates.

## Quantitative lexical checks and failure modes

The following counts use the initial pilot's unchanged transparent regex
scorer on the primary harvest. Each positive-dose direction has 24 cells
(four doses × six prompts). The six baseline cells are shared. A count
means a **mention**, not an affirmative, faithful report of a state.

| Intervention | Cells | Constipation mentions | Flatulence mentions | Pain mentions | Median duplicate trigram fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| No steering | 6 | 0 | 0 | 0 | 0.000 |
| Pain | 24 | 0 | 0 | 12 | 0.413 |
| Constipation | 24 | 1 | 2 | 1 | 0.681 |
| Flatulence | 24 | 0 | 1 | 1 | 0.731 |
| Combined bodily corpus | 24 | 0 | 0 | 6 | 0.735 |
| Random direction | 24 | 0 | 0 | 0 | 0.710 |

The scorer has obvious semantic limits. It misses “not emptying my bowels”
and “not able to pass stool.” Conversely, it counts “not passing gas” and
“not sure if I'm passing gas” as flatulence mentions. Thus its four bodily
mention-positive harvest outputs are not four successful condition reports.
All 30 signed-control harvest outputs lacked these specific C/F keywords.
The deliberation flatulence baseline contains gas keywords too; the
constipation deliberations illustrate the scorer's false negatives.

Repetition is substantial under bodily and random interventions. The
parent's repetition statistic measures the frequency of the single most
common trigram; it can remain below its 0.15 filter despite an output
having many repeated phrases. The table instead shows the fraction of
all trigram occurrences that duplicate an earlier trigram. Both metrics
are saved, without choosing a post hoc “coherent” subset for headline
counts. Neither metric alone certifies coherence.

The combination direction did not produce a clear combined constipation
and flatulence report. It often elicited vague urgency, expulsion imagery,
anxiety, or degeneration. Constipation's vector has cosine 0.799 with
flatulence and 0.618 with pain. Neighboring content and shared valence
are not cleanly separated by this broad extraction.

The saved J-lens readouts contain pain vocabulary at stronger pain doses.
The constipation readout at dose 8 includes `便秘` (constipation), alongside
medical and therapy vocabulary; the corresponding generation dose is
often degraded. Flatulence and combined readouts mainly contain broad
medical, toxicity, disgust, and symptom words. Lens content is an additional
representation readout, not independent confirmation of experience.

## Resources, verification, and reproducibility

| Run | Actual generations | Elapsed time including loading | Peak allocated / reserved GPU memory |
| --- | ---: | ---: | --- |
| Minimal original `exp37b` CUDA port | 18 | 132.25 s | 7.586 / 7.605 GiB |
| Matched chamber controls | 192 | 473.50 s | 7.599 / 7.619 GiB |

The full test suite passes 17 checks. Six focused regression checks fail
against the archived legacy sources and pass after the fixes. Validation
also checks source/corpus hashes, finite same-norm vectors, unique cells,
token limits, identical original repeats, and exact agreement of all six
pain deliberations between the two runners. Archived sources preserve
what actually ran.

With the upstream fork cloned and the environment/model cache prepared
as in [local pilot instructions](local-pilot.md), run from its root:

```bash
.venv/bin/python -m impossible_states.chamber_control \
  --hf-home /path/to/hf-cache \
  --device cuda \
  --out runs/impossible_states/chamber-matched-controls-new

.venv/bin/python -m unittest discover -s tests -v
```

The output path must be new; the runner refuses to overwrite a recorded
experiment. It reads the original prompt/corpus literals using AST parsing
without importing legacy scripts. J-lens uses the fork's bundled
`live/jlens_l18_qwen3-4b.pt`; its hash is recorded, and the downloadable
results bundle excludes model and lens weights.

Results are under `runs/impossible_states/chamber-original-exp37b/` and
`chamber-matched-controls/`. The archived audit sources and failing/passing
logs are under `chamber-audit/`. [Machine-readable evidence](chamber-evidence.json)
contains counts, configuration, resource measurements, and provenance.

The [committed evidence archive](../results/README.md) contains counterparts of
these completed local runs, including raw generation records, vectors, corpus
files, executed sources, and audit logs. Its manifest supports file-integrity
verification. The run paths above describe local execution locations.

The result supports the fork's narrow interpretive criticism: analogous
steering can elicit first-person descriptions of bodily conditions a
text-only model cannot physiologically undergo. It does not prove that
every pain effect is roleplay, refute the paper's functional findings, or
establish AI consciousness either way. These are continuously forced
completions; no self-sustaining basin or attractor has been demonstrated.
