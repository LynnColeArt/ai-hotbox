"""Single-device frozen-weight extraction and explicit cached decoding."""
import numpy as np
import torch

class Engine:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model.eval(), tokenizer, device
        self.blocks = model.model.layers
        model.requires_grad_(False)

    def prompt_ids(self, prompt):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.inference_mode()
    def extract(self, rows, layers, max_input_tokens=256):
        # Only final-position block outputs move to CPU. Never retain all tokens
        # at all layers, and never request output_hidden_states.
        acts = np.empty((len(rows), len(layers), self.model.config.hidden_size), dtype=np.float32)
        current = [0]
        handles = []
        for j, layer in enumerate(layers):
            def capture(module, inputs, output, j=j):
                h = output[0] if isinstance(output, tuple) else output
                acts[current[0], j] = h[0, -1].float().cpu().numpy()
            handles.append(self.blocks[layer].register_forward_hook(capture))
        try:
            for i, row in enumerate(rows):
                current[0] = i
                ids = self.tokenizer(row["text"], return_tensors="pt").input_ids.to(self.device)
                if ids.shape[1] > max_input_tokens:
                    raise ValueError(f"Input {row['id']} exceeds extraction limit; do not silently truncate")
                self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
                if (i + 1) % 14 == 0:
                    print(f"extraction {i+1}/{len(rows)}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
        return acts

    @torch.inference_mode()
    def rollout(self, prompt, layer, direction, center, scale, dose, mode,
                seed=11, max_tokens=48, pulse_tokens=3, forced_tokens=None,
                readout=None):
        if mode not in {"baseline", "continuous", "pulse", "rebuild", "perturb"}:
            raise ValueError(mode)
        ids = self.prompt_ids(prompt)
        prefix = ids.clone()
        u = torch.as_tensor(direction, device=self.device, dtype=torch.float32)
        u = u / u.norm()
        origin = torch.as_tensor(center, device=self.device, dtype=torch.float32)
        delta = (dose * scale * u).to(self.model.dtype)
        state = {"inject": False, "opposite": False, "trace": None}

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            before = h[0, -1].float()
            pre = float((before - origin) @ u / scale)
            if state["inject"] or state["opposite"]:
                h = h.clone()
                h[0, -1] += -delta if state["opposite"] else delta
            post = float((h[0, -1].float() - origin) @ u / scale)
            state["trace"] = dict(before=pre, after=post)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        handle = self.blocks[layer].register_forward_hook(hook)
        readout_handle = None
        if readout is not None:
            readout_layer, rv, rc, rs = readout
            ru = torch.as_tensor(rv, device=self.device, dtype=torch.float32)
            ru = ru / ru.norm()
            ro = torch.as_tensor(rc, device=self.device, dtype=torch.float32)
            def read_hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                state["trace"]["downstream_projection"] = float((h[0, -1].float() - ro) @ ru / rs)
            readout_handle = self.blocks[readout_layer].register_forward_hook(read_hook)
        cache = None
        generated, traces = [], []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        try:
            steps = len(forced_tokens) if forced_tokens is not None else max_tokens
            for step in range(steps):
                state["inject"] = mode == "continuous" or (mode in {"pulse", "rebuild", "perturb"} and step < pulse_tokens)
                state["opposite"] = mode == "perturb" and step == pulse_tokens
                if mode == "rebuild" and step == pulse_tokens:
                    # Discard the steered cache, then replay IDENTICAL visible
                    # text without intervention. Future text is free to diverge.
                    cache = None
                    ids = torch.cat([prefix, torch.tensor([generated], device=self.device)], dim=1)
                out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                traces.append(dict(step=step, injected=state["inject"], opposing=state["opposite"],
                                   cache_rebuilt=mode == "rebuild" and step == pulse_tokens, **state["trace"]))
                if forced_tokens is None:
                    # Seeded sampling, fixed temperature and top-k in every arm.
                    values, indices = torch.topk(out.logits[0, -1].float() / 0.7, 20)
                    selected = torch.multinomial(torch.softmax(values, -1), 1, generator=generator)
                    token = int(indices[selected].item())
                else:
                    token = int(forced_tokens[step])
                del out
                generated.append(token)
                ids = torch.tensor([[token]], device=self.device)
                if forced_tokens is None and token in eos:
                    break
        finally:
            handle.remove()
            if readout_handle:
                readout_handle.remove()
        return dict(prompt=prompt, layer=layer, dose=dose, mode=mode, seed=seed,
                    pulse_tokens=pulse_tokens, generated_token_ids=generated,
                    text=self.tokenizer.decode(generated, skip_special_tokens=True), trace=traces,
                    release_prefix_token_ids=generated[:pulse_tokens],
                    eos_before_release=len(generated) <= pulse_tokens,
                    teacher_forced=forced_tokens is not None)
