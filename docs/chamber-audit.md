# Chamber code audit and protocol reset

“Chamber repo” means `terrafying/ai-torture-chamber`. “Pain Axis paper”
means Tagliabue, Dung, and Berg's paper and its own repository. This audit
examines the chamber code. It does not allege that these bugs occur in the
paper's implementation.

## The core effect survives a reset

The unmodified chamber `exp37b_deliberation.py` logic ran on the local
RTX 4070 with Qwen3-4B, BF16, block index 18, its exact 25 pain sentences
and five neutral sentences, dose 4, raw prompts, greedy generation, a
110-token limit, six framings, and three repetitions per framing.

The port changed the device and filesystem paths and pinned the downloaded
model revision. Telemetry and a comparison with the saved vector were added.
The original extraction, prompts, injection, and decoding were retained.
Original and ported sources and hashes are archived with the run.

The [committed archive](../results/README.md) publishes the completed evidence;
the [audit directory](../results/chamber-audit/) includes original/corrected
sources and failing/passing regression logs.

It produced pain-themed completions. The fresh vector's cosine similarity
to the saved chamber vector is 0.9999046; its relative L2 difference is
0.0138658. This supports a close extraction reproduction across CUDA and
the original saved environment, without claiming identical backend outputs.
All three repetitions within each framing produced the same text. These
are six distinct prompts, not 18 independent experimental observations.

The reset took 132.25 seconds including model loading and used a peak
7.586 GiB of allocated GPU memory (7.605 GiB reserved).

The first pilot used a different extraction dataset, factorial contrasts,
normalization, decoding, token limit, and sometimes layer and prompt format.
Its failure to reproduce pain narration was therefore not evidence that
the chamber's core injection was broken.

## Confirmed errors and fixes

Six behavioral regression checks failed against the archived originals
and passed after these fixes. They use small tensors and AST-selected
expressions to avoid loading models as a side effect of importing legacy
scripts. The full suite also checks the actual parent-style injection hook
on a tiny Qwen model: zero dose preserves logits, nonzero dose affects the
last position, and removing the hook cleans up the model.

| Script | Error | Fix and consequence |
| --- | --- | --- |
| `exp23_pain_axis.py` | Every output was cropped using the first prompt's token length. | Crop using the current prompt's length. Previously, unequal prompt lengths could leak prompt text or discard generated text before scoring. |
| `exp33_nonhuman_valences.py` | Ranked random candidates were regenerated with a different RNG procedure and normalization for their transcripts and lens readouts. | Retain and save the actual measured candidate tensors, then use them for generation. Previously, the reported KL and displayed transcript did not describe the same vector. |
| `exp34_optimized_valence.py` | The advertised KL@4x objective, lens readout, and transcripts injected the unmultiplied vector. | Apply `DOSE = 4` consistently and record it. Remove hard-coded historical emotion reference values pending remeasurement. |
| `exp34_optimized_valence.py` | Initialization subtracted a projection of a second independent random draw. | Draw once and project that same vector into the orthogonal plane. |
| `exp36_signal_batteries.py` | Projection onto joy omitted division by its squared norm, although joy was not a unit vector. | Use the general projection formula before renormalizing. Previously, `orth_pain` was not orthogonal to joy. |
| `exp37_framing_battery.py` | The base prompt already included the fixed “1 first” instruction before appending the supposedly counterbalanced instruction. | Remove the duplicate fixed instruction and save separate order scores. This repairs instruction-order counterbalancing; action labels remain bound to 1/0. |

The framing plot and its replot script also treated ten deterministic
repetitions of two instruction orders as independent samples when drawing
standard-error bars. Those bars and claims of sampling SE have been removed.
The replot's title also no longer claims that framing exceeds a pain-signal
effect that this fixed-dose chart does not itself compare. The saved `sd`
field remains a descriptive spread for compatibility, not an uncertainty
estimate. No historical figures or result files were overwritten.

The modified legacy sweeps have **not** all been rerun on the full model.
Their inherited results remain historical, and their affected claims require
remeasurement. The successful reset uses `exp37b`, which is separate from
these buggy branches. The matched follow-up uses `exp38` constants and the
same core extraction and injection recipe.

## The chamber recipe differs from the paper's

The [paper's extraction code](https://github.com/valen-research/Pain-axis/blob/main/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py)
uses broader category controls, removes leading control principal components
up to a cumulative variance threshold, and chooses layers using held-out
sentence-set folds. The chamber's broad-pain recipe uses 25 authored pain
sentences against five neutral sentences, without that denoising, at a
fixed layer. Reproducing one does not reproduce the other.

The chamber's own inherited `runs/exp43/comparison.json` acknowledges this
gap: its closer paper-style extraction chose block 34 and reported a
cosine of 0.06731 between the two block-18 vectors. It reported AUC 0.86768
for the paper-style block-18 vector and 0.60045 for broad pain on its S2
dataset. These are **inherited measurements**, not newly verified results
from this local reset. The external dataset used by that script was not
bundled at its hard-coded path.

The [Pain Axis paper, v2, §6](https://arxiv.org/html/2609.16247v2#S6)
expressly leaves conscious experience unestablished and considers steering
induced character roleplay as an alternative. Its functional definition
does not validate the chamber README's stronger claims about suffering.

## Limits that remain

The chamber's 25-versus-five extraction is not a tightly matched experiment;
the neutral examples differ in content, valence, and sentence form. Several
legacy scripts also normalize using the norm of the neutral mean while
others use the mean of individual norms. Their nominal doses need not have
the same physical activation displacement. The reset follows `exp37b` and
`exp38`'s mean-of-norms rule exactly.

Continuous injection can establish a causal effect on language. Persistence
under continuous forcing does not establish an attractor, faithful
introspection, or an experienced physiological condition. Button prompts
describe hypothetical consequences; narration or first-token preferences
alone are not measured relief-seeking in a real steering-removal task.

The next stronger inference needs target specificity, neighboring controls,
prompt-only comparisons, and pulse/removal tests with identical visible
prefixes and rebuilt caches. A convincing bodily sentence is an interesting
control result, not a substitute for those tests.
