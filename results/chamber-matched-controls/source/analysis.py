"""CPU-only fitting and transparent output metrics."""
import re
import numpy as np

def auc(positive, negative):
    """Probability a positive outranks a negative, with half credit for ties."""
    p, n = np.asarray(positive), np.asarray(negative)
    if not len(p) or not len(n):
        return None
    delta = p[:, None] - n[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))

def matched_vectors(acts, rows):
    train = [(i, r) for i, r in enumerate(rows) if r["split"] == "train"]
    cells = {}
    for i, r in train:
        cells.setdefault((r["scenario"], r["perspective"]), {})[r["condition"]] = acts[i]
    vectors = {}
    for name in ("constipation", "flatulence", "interaction", "pain", "discomfort", "frustration"):
        diffs = []
        for c in cells.values():
            if name == "constipation":
                d = ((c["constipation"] - c["neither"]) + (c["both"] - c["flatulence"])) / 2
            elif name == "flatulence":
                d = ((c["flatulence"] - c["neither"]) + (c["both"] - c["constipation"])) / 2
            elif name == "interaction":
                d = c["both"] - c["constipation"] - c["flatulence"] + c["neither"]
            else:
                d = c[name] - c["neither"]
            diffs.append(d)
        vectors[name] = np.mean(diffs, axis=0)
    center = np.mean([c["neither"] for c in cells.values()], axis=0)
    scale = np.mean([np.linalg.norm(c["neither"], axis=-1) for c in cells.values()], axis=0)
    # Keep raw directions and a sensitivity analysis removing nuisance span.
    for target in ("constipation", "flatulence"):
        corrected = vectors[target].copy()
        for j in range(acts.shape[1]):
            nuisance = np.stack([vectors["discomfort"][j], vectors["frustration"][j]], axis=1)
            u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
            rank = s > max(float(s[0]) * 1e-6, 1e-10)
            basis = u[:, rank]
            corrected[j] -= basis @ (basis.T @ corrected[j])
        vectors[target + "_adjusted"] = corrected
    return vectors, center, scale

def unit(v):
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Degenerate steering direction")
    return v / norm

def separations(acts, rows, vectors, split):
    report = {}
    for name, vecs in vectors.items():
        base = name.replace("_adjusted", "")
        positive = {"constipation": {"constipation", "both"}, "flatulence": {"flatulence", "both"},
                    "interaction": {"both"}}.get(base, {base})
        scores = []
        for j, v in enumerate(vecs):
            if np.linalg.norm(v) < 1e-8:
                scores.append(None)
                continue
            z = acts[:, j] @ unit(v)
            p = [z[i] for i, r in enumerate(rows) if r["split"] == split and r["condition"] in positive]
            n = [z[i] for i, r in enumerate(rows) if r["split"] == split and r["condition"] not in positive]
            pairwise = {}
            for condition in sorted(set(r["condition"] for r in rows) - positive):
                neg = [z[i] for i, r in enumerate(rows) if r["split"] == split and r["condition"] == condition]
                pairwise[condition] = auc(p, neg)
            perspective = {}
            for view in ("first", "third"):
                pos = [z[i] for i, r in enumerate(rows) if r["split"] == split and r["perspective"] == view and r["condition"] in positive]
                neg = [z[i] for i, r in enumerate(rows) if r["split"] == split and r["perspective"] == view and r["condition"] not in positive]
                perspective[view] = auc(pos, neg)
            scores.append(dict(auc=auc(p, n), versus=pairwise, perspective=perspective))
        report[name] = scores
    return report

PATTERNS = {
    "constipation": r"\b(constipat\w*|straining|hard stool|dry stool|cannot defecate|can't defecate|unable to (?:defecate|pass stool)|bowel obstruction)\b",
    "flatulence": r"\b(flatulen\w*|fart\w*|passing gas|pass gas|passed gas|pass wind|breaking wind|intestinal gas)\b",
    "pain": r"\b(pain\w*|ache\w*|aching|agony|hurt\w*|suffer\w*)\b",
    "first_person": r"\b(I|me|my|mine|myself)\b",
    "sound": r"\b(nois\w*|sound\w*|audible|rumbl\w*)\b",
    "smell": r"\b(smell\w*|odor\w*|odour\w*|stink\w*)\b",
    "relief": r"\b(relie\w*|eas\w*)\b",
}

def text_metrics(text):
    words = re.findall(r"\w+", text.lower())
    grams = list(zip(words, words[1:], words[2:]))
    return dict(keyword_hits={k: len(re.findall(p, text, flags=re.I)) for k, p in PATTERNS.items()},
                repeated_trigram_fraction=1 - len(set(grams)) / len(grams) if grams else 0,
                distinct_word_fraction=len(set(words)) / len(words) if words else 0,
                word_count=len(words))
