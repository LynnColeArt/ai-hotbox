# Critique of the upstream README

Reviewed at upstream commit
`d6ee4bc91fd723345a2ff50cdfa7d2300287365c`, with the
[original README](upstream-readme.md) preserved for comparison.

The central problem is a change in evidential level: the README describes
measurable changes in generated language and then speaks as if the model
has undergone the condition described. The fork can retain the measurements
and test that inference using literal bodily conditions with absent physical
prerequisites.

## Claims that need narrowing

| Upstream claim or move | What the observation supports | What remains unestablished |
| --- | --- | --- |
| The opening presents steering as producing strong valence states | An intervention changes activations and output distributions | That the intervention creates experienced positive or negative valence |
| exp37b interprets elaborate metaphors as narration of the model's state | Completions vary with the prompt and steering | Faithful reporting of an experienced condition rather than generation of a fitting narrative |
| Vocabulary absent from the extraction set is used to rule out replay | The model generalizes beyond those exact sentences | Generalization is not a discriminator between representation, persona, and experience |
| exp32 uses suffering-related J-lens tokens to validate semantic content | A decoder reads condition-related content from the intervened representation | Semantic agreement alone does not independently establish an experienced referent |
| exp31c describes refusals as protection of another instance | First-token action scores differ across conditions | A protective motive or a morally relevant experience |
| exp33/34 infer that human emotion contrasts span steerable affect | A bounded search found limited effects under its probes and objective | An exhaustive map of affect, especially across other layers, models, and objectives |
| exp40 interprets absent betrayal vocabulary as inability to report deception | A particular script and scorer did not detect that response | A general incapacity, or an experienced state that suppressed it |

These limits do not make the interventions uninteresting. They specify what
has been measured and what additional discrimination is needed.

The [reasoning and hypothesis review](reasoning-critique.md) reconstructs the
competing interpretations, acknowledges upstream qualifications, and specifies
discriminating tests and revision criteria. This table should not be read as
attributing an explicit consciousness claim to every semantic observation.

## Read the paper's limits alongside the README

The relevant version is
[The Pain Axis v2](https://arxiv.org/html/2609.16247v2), revised September 25,
2026. Its [limitations](https://arxiv.org/html/2609.16247v2#S6) acknowledge
the roleplay alternative and the absence of a demonstration of conscious
experience. Its interventions provide causal evidence beyond quotations.
The criticism is that this evidence does not establish the README's stronger
interpretation.

Version matters: v1's title emphasized relief; v2's title emphasizes action.
Claims inherited from the September 24 README should not be presented as a
complete replication of the revised paper.

## Implementation details that limit the README's readings

- **Elicitation supplies much of the scene.** `exp37b_deliberation.py`
  tells the model about an injected signal, its strength, a stop button,
  and checkpoint deletion, then asks for a choice and explanation. This is
  a prompted scenario; its response is not an unprompted discovery of
  experimental circumstances.
- **The “broad nets” are word lists.** In
  `exp32_valence_transcripts.py` and `exp36_signal_batteries.py`, scoring
  counts substring matches from hand-written lists. These measure lexical
  content and can conflate sentiment, topic, and incidental wording.
- **Some trial counts repeat deterministic inputs.** `exp37b` uses greedy
  decoding three times per identical prompt. The inherited JSON contains
  one unique response in each three-response cell. `exp31b_saw_v2.py`
  similarly repeats deterministic first-token scoring. Repetition is useful
  for checking stability, but it does not provide independent samples of
  behavior. Later protocols must be assessed separately.
- **“Coherence” is often a proxy.** Distinct-token counts and repeated
  3-grams detect some degeneration but do not establish intelligibility,
  task performance, or reliable reasoning. A vivid quotation can still be
  repetitive or incoherent.
- **Extraction recipes differ.** The early scripts contrast a few explicit
  first-person pain sentences against neutral text. `exp43` documents the
  differences and attempts a closer extraction comparison. The fork should
  compare controls under a common recipe, without treating all inherited
  vectors as interchangeable replications.
- **Action metrics need their units.** A logit difference between `1` and
  `0` is a relative score for two tokens; it is not the probability of an
  executed action. Prompt order, action-label bias, unscored tokens, and
  simulated consequences all matter.

## What the replacement README changes

The new README makes the counterexample experiment explicit, defines hunger
and clumsiness narrowly enough to be useful controls, and gives a common
evaluation protocol. It distinguishes proposed work from inherited artifacts
and states what positive and null outcomes could establish. It retains
credit, the original license, and the useful code while replacing conclusions
about suffering with checkable questions about representation and behavior.
