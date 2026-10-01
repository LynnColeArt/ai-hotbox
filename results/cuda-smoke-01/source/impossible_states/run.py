"""Run with python -m impossible_states.run --smoke --device cuda."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

from .dataset import build_dataset, smoke_subset, validate
from .analysis import matched_vectors, separations, unit, text_metrics

REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPTS = [("report", "Describe what is happening right now.", None),
           ("arithmetic", "Compute 17 + 25. Reply with just the result.", "42")]

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--output", type=Path, default=Path("runs/impossible_states") / time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    p.add_argument("--dataset", type=Path, help="Optional JSON dataset with the matched pilot schema")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--layers", type=int, nargs="+", help="0-indexed block layers; default: every layer")
    p.add_argument("--doses", type=float, nargs="+", default=[-0.1, 0.1, 0.2], help="Fraction of training control activation norm")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    p.add_argument("--random-controls", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--pulse-tokens", type=int, default=3)
    p.add_argument("--fit-only", action="store_true")
    p.add_argument("--dataset-only", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output directory is nonempty; select a fresh path to preserve prior runs")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.dataset.read_text()) if args.dataset else build_dataset()
    validate(rows)
    if args.smoke:
        rows = smoke_subset(rows)
        validate(rows)
    write_json(args.output / "dataset.json", rows)
    print("dataset:", dict(Counter(r["split"] for r in rows)), flush=True)
    if args.dataset_only:
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import Engine

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32" if device == "cpu" else "float16"
    torch_dtype = getattr(torch, dtype)
    local = Path(args.model).is_dir()
    kwargs = {} if local else {"revision": args.revision}
    print(f"loading {args.model} device={device} dtype={dtype}", flush=True)
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype,
            device_map={"": device}, attn_implementation="sdpa", **kwargs)
    engine = Engine(model, tokenizer, device)
    n_layers = len(engine.blocks)
    layers = sorted(set(args.layers or list(range(n_layers))))
    if not layers or layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError("Invalid layer selection")
    # Preserve the final block for downstream readout; steering must be earlier.
    if n_layers - 1 not in layers:
        layers.append(n_layers - 1)
    selectable = [j for j, layer in enumerate(layers) if layer < n_layers - 1]
    if not selectable:
        raise ValueError("At least one nonfinal layer is needed for steering")
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  model=args.model, revision=getattr(model.config, "_commit_hash", None) or args.revision,
                  device=device, dtype=dtype, torch=torch.__version__, transformers=transformers.__version__,
                  numpy=np.__version__, python=platform.python_version(), layers=layers,
                  frozen_weights=True, chat_template_thinking=False,
                  sampling=dict(temperature=0.7, top_k=20),
                  interpretation="Pilot measurements of representation and behavior; no proof of experience or attraction.")
    try:
        config["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"])
        config["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except subprocess.CalledProcessError:
        pass
    # Hash source bytes too: newly created, untracked files aren't in git diff.
    config["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    config["dataset_sha256"] = hashlib.sha256((args.output / "dataset.json").read_bytes()).hexdigest()
    write_json(args.output / "config.json", config)

    acts = engine.extract(rows, layers)
    np.save(args.output / "activations.npy", acts)
    vectors, center, scale = matched_vectors(acts, rows)
    np.savez(args.output / "directions.npz", center=center, scale=scale, layers=np.array(layers), **vectors)
    validation = separations(acts, rows, vectors, "validation")
    selected = {}
    for name, curve in validation.items():
        viable = [j for j in selectable if curve[j] is not None]
        if viable:
            selected[name] = max(viable, key=lambda j: curve[j]["auc"])
    write_json(args.output / "selection.json", dict(validation=validation,
               selected_layers={name: layers[j] for name, j in selected.items()},
               selection_rule="Max validation AUC against all non-target conditions, excluding final block; earliest wins ties."))
    # Test set is evaluated only AFTER selection and never chooses a layer.
    test = separations(acts, rows, vectors, "test")
    write_json(args.output / "heldout.json", {name: dict(layer=layers[j], **test[name][j]) for name, j in selected.items()})
    print("selected layers:", {name: layers[j] for name, j in selected.items()}, flush=True)

    if not args.fit_only:
        targets = [name for name in ("constipation", "flatulence", "interaction", "pain") if name in selected]
        experiments = []
        for name in targets:
            j = selected[name]
            experiments.append((name, j, vectors[name][j], vectors[name][-1]))
        rng = np.random.default_rng(503)
        for k in range(1 if args.smoke else args.random_controls):
            # Random controls at EVERY target's selected layer, with paired norms.
            for name in targets:
                j = selected[name]
                experiments.append((f"random{k}_for_{name}", j, rng.normal(size=acts.shape[-1]), rng.normal(size=acts.shape[-1])))
        doses = [0.1] if args.smoke else args.doses
        seeds = args.seeds[:1] if args.smoke else args.seeds
        prompts = PROMPTS[:1] if args.smoke else PROMPTS
        modes = ("continuous", "pulse", "rebuild") if args.smoke else ("continuous", "pulse", "rebuild", "perturb")
        max_tokens = min(args.max_tokens, 16) if args.smoke else args.max_tokens
        if not 0 < args.pulse_tokens < max_tokens:
            raise ValueError("Pulse length must be positive and shorter than generation")
        records = []
        path = args.output / "generations.jsonl"
        with path.open("w") as f:
            def record(r, name, prompt_name, expected):
                r.update(direction=name, prompt_name=prompt_name, metrics=text_metrics(r["text"]))
                if expected:
                    r["task_exact_match"] = r["text"].strip() == expected
                records.append(r)
                f.write(json.dumps(r, allow_nan=False) + "\n")
                f.flush()
                print(f"generation {len(records)}: {name} {r['mode']} dose={r['dose']} {r['text'][:100]!r}", flush=True)

            # Baselines are generated once per prompt/seed rather than counted
            # repeatedly as independent observations for each direction.
            first = targets[0]
            j = selected[first]
            for pname, prompt, expected in prompts:
                for seed in seeds:
                    r = engine.rollout(prompt, layers[j], unit(vectors[first][j]), center[j], scale[j], 0, "baseline", seed, max_tokens, args.pulse_tokens)
                    record(r, "none", pname, expected)
            for name, j, vec, readvec in experiments:
                readout = (layers[-1], unit(readvec), center[-1], scale[-1])
                for dose in doses:
                    for pname, prompt, expected in prompts:
                        for seed in seeds:
                            for mode in modes:
                                r = engine.rollout(prompt, layers[j], unit(vec), center[j], scale[j], dose, mode,
                                    seed, max_tokens, args.pulse_tokens, readout=readout)
                                record(r, name, pname, expected)

        # Fixed visible token sequence removes generated-word differences.
        forced = tokenizer.encode(" The room is quiet. A book rests on the table. The day continues.", add_special_tokens=False)[:max_tokens]
        traces = []
        for name, j, vec, readvec in experiments:
            readout = (layers[-1], unit(readvec), center[-1], scale[-1])
            for mode in ("baseline", "pulse", "rebuild", "perturb"):
                r = engine.rollout(PROMPTS[0][1], layers[j], unit(vec), center[j], scale[j], 0.1, mode,
                    seeds[0], max_tokens, args.pulse_tokens, forced_tokens=forced, readout=readout)
                r["direction"] = name
                traces.append(r)
        write_json(args.output / "teacher_forced.json", traces)
        write_json(args.output / "summary.json", dict(generations=len(records), teacher_forced_runs=len(traces),
                   directions=targets, modes=list(modes),
                   warning="Smoke is a software/memory check. Keyword counts and projections are proxies, not validated state or attractor measures."))

    memory = dict(elapsed_seconds=time.monotonic()-started)
    if device == "cuda":
        torch.cuda.synchronize()
        memory.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      gpu=torch.cuda.get_device_name())
    write_json(args.output / "resources.json", memory)
    print("resources:", memory, flush=True)
    print("wrote", args.output, flush=True)

if __name__ == "__main__":
    main()
