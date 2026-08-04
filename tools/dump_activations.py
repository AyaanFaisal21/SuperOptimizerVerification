"""phase 3: capture real activations from the trained TransformerOp checkpoint.

CLAIM.md T2 pre-registers that "realistic" means activations sampled from a trained
model, not torch.randn. this produces that fixture set.

which tensors, and why these:

  resid_pre_ln   input to LayerNorm -- the tensor whose variance actually gets
                 computed. the natural fixture for layernorm_variance, and the one
                 that matters: post-LayerNorm output is normalised by construction,
                 so feeding it back into a variance computation tests nothing.
  post_ln        output of LayerNorm. zero-mean, unit-variance by construction --
                 included precisely as the well-conditioned control.
  post_gelu      output of the MLP's GELU. bounded below by ~-0.17 and mostly
                 positive, so it carries a large nonzero mean. this is the predicted
                 high-value case: E[x^2] - mu^2 cancels when mu^2 approaches E[x^2],
                 which zero-mean randn never produces.
  attn_scores    pre-softmax scores. the natural fixture for softmax_online.
  mlp_out        MLP projection output, back in the residual stream's scale.

each site records shape and distribution statistics alongside the tensor, because
the mean/std ratio is the quantity that governs cancellation and a reader needs it
to interpret any layernorm result.

    python tools/dump_activations.py --ckpt ~/TransformerOp/checkpoints/gpt.pt \
                                     --repo ~/TransformerOp --out fixtures/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROWS = 512          # rows kept per site; keeps the committed fixture small
LAYERS = (0, 5)     # first and last block -- early and late residual statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="TransformerOp checkout")
    ap.add_argument("--out", default="fixtures")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from model.gpt import GPT, GPTConfig
    from model.tokenizer import CharTokenizer

    text = (Path(args.repo) / "data" / "shakespeare.txt").read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    val = ids[int(0.9 * len(ids)):]          # held-out split, not what it trained on

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()                             # dropout off: fixtures must be the
                                             # deterministic forward pass
    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_m, inp, out):
            t = (inp[0] if name.startswith("resid_pre") else out).detach()
            captured[name] = t.reshape(-1, t.shape[-1]).float()
        return hook

    handles = []
    for i in LAYERS:
        blk = model.blocks[i]
        handles += [
            blk.ln1.register_forward_hook(grab(f"resid_pre_ln_L{i}")),
            blk.ln1.register_forward_hook(grab(f"post_ln_L{i}")),
            blk.ffwd.net[1].register_forward_hook(grab(f"post_gelu_L{i}")),
            blk.ffwd.net[2].register_forward_hook(grab(f"mlp_out_L{i}")),
        ]

    g = torch.Generator().manual_seed(0)
    off = torch.randint(len(val) - cfg.block_size, (args.batch,), generator=g)
    x = torch.stack([val[o:o + cfg.block_size] for o in off])

    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    # attention scores are computed inside forward and never surface as a module
    # output, so they are recomputed here from the same captured input rather than
    # hooked -- same weights, same tensor, just reconstructed.
    with torch.no_grad():
        blk = model.blocks[LAYERS[0]]
        h = captured[f"resid_pre_ln_L{LAYERS[0]}"]
        B, T, C = x.shape[0], x.shape[1], cfg.n_embd
        hn = blk.ln1(h.reshape(B, T, C))
        q, k, _ = blk.attn.qkv(hn).split(C, dim=2)
        hs = C // cfg.n_head
        q = q.view(B, T, cfg.n_head, hs).transpose(1, 2)
        k = k.view(B, T, cfg.n_head, hs).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (hs ** -0.5)
        captured["attn_scores"] = scores.reshape(-1, T).float()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixtures, meta = {}, {}
    for name, t in sorted(captured.items()):
        # .clone(), not .contiguous(): slicing rows of a row-major tensor is already
        # contiguous, so .contiguous() is a no-op and torch.save writes the whole
        # underlying storage -- 56 MB instead of 11.
        t = t[:ROWS].clone()
        fixtures[name] = t

        # LayerNorm reduces PER ROW, so per-row statistics are what govern
        # cancellation in E[x^2] - mu^2. the global mean is the wrong number: a
        # tensor can be globally zero-mean while individual rows are badly biased,
        # and it is the rows that get normalised.
        row_mu = t.mean(dim=-1)
        row_ex2 = t.pow(2).mean(dim=-1)
        row_cancel = (row_mu.pow(2) / row_ex2.clamp_min(1e-30))
        m, s = t.mean().item(), t.std().item()
        meta[name] = {
            "shape": list(t.shape),
            "mean": m,
            "std": s,
            "min": t.min().item(),
            "max": t.max().item(),
            # mu^2 / E[x^2] per row: 0 means the one-pass variance is safe,
            # approaching 1 means catastrophic cancellation.
            "cancel_global": (m * m) / (t.pow(2).mean().item() or 1.0),
            "cancel_row_mean": row_cancel.mean().item(),
            "cancel_row_p99": row_cancel.quantile(0.99).item(),
            "cancel_row_max": row_cancel.max().item(),
        }

    torch.save(fixtures, out / "activations.pt")
    (out / "activations.json").write_text(json.dumps(meta, indent=2))

    print(f"{'site':<22}{'shape':>16}{'std':>8}"
          f"{'cancel: global':>15}{'row mean':>10}{'row p99':>10}{'row max':>10}")
    for name, m in meta.items():
        print(f"{name:<22}{str(tuple(m['shape'])):>16}{m['std']:>8.3f}"
              f"{m['cancel_global']:>15.4f}{m['cancel_row_mean']:>10.4f}"
              f"{m['cancel_row_p99']:>10.4f}{m['cancel_row_max']:>10.4f}")
    size = (out / "activations.pt").stat().st_size
    print(f"\nwrote {out}/activations.pt ({size/1e6:.1f} MB) and activations.json")
    print("cancel = mu^2/E[x^2]; ->1 is catastrophic for the one-pass variance form")


if __name__ == "__main__":
    sys.exit(main())
