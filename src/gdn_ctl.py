"""Training-free intervention on Qwen3.5 GatedDeltaNet state updates.

Patches the chunk kernel call inside every Qwen3_5GatedDeltaNet so that, during prefill, per-token
write strength (beta) and log-decay (g) can be rescaled by a relevance mask m_t in [0, 1]:

    beta'_t = beta_t * (s + (1 - s) * m_t)        s = write scale for irrelevant tokens
    g'_t    = g_t    * (d + (1 - d) * m_t)        d = decay scale for irrelevant tokens (0 => no forgetting)

Tokens outside the context span always keep m_t = 1.
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as M


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cfg = None          # dict describing the intervention, None = off
        self.layer = None        # current GDN layer index (set by pre-hook)
        self.ctx = None          # bool [T] context tokens
        self.qidx = None         # long [Lq] question-before token indices
        self.mask = None         # float [T] or [T, H] precomputed relevance mask (oracle / pass-1)
        self.capture = None      # None or dict to fill with per-layer relevance scores (diagnostics)
        self.cap_layers = None
        self.stats = {}
        self.seed = 0

    @property
    def active(self):
        return self.cfg is not None or self.capture is not None


CTL = Controller()
_orig_chunk = M.torch_chunk_gated_delta_rule


def qk_relevance(q, k, qidx, ctx, agg="max"):
    """q, k: [T, H, D] (raw). Returns r: [H, Tc] cosine(question q, context k) aggregated over question tokens."""
    qn = F.normalize(q[qidx].float(), dim=-1)          # [Lq, H, D]
    kn = F.normalize(k[ctx].float(), dim=-1)           # [Tc, H, D]
    sim = torch.einsum("qhd,thd->hqt", qn, kn)          # [H, Lq, Tc]
    return sim.amax(1) if agg == "max" else sim.mean(1)


def topk_mask(r, frac, win):
    """r: [..., Tc] scores -> {0,1} mask keeping top `frac` per row, then causal forward dilation over `win` tokens
    (a relevant token also marks the next `win` tokens, so the value following a matched key gets written)."""
    tc = r.shape[-1]
    kk = max(1, int(round(frac * tc)))
    thr = r.topk(kk, dim=-1).values[..., -1:]
    m = (r >= thr).float()
    if win > 0:
        shp = m.shape
        m2 = F.max_pool1d(F.pad(m.reshape(-1, 1, tc), (win, 0)), kernel_size=win + 1, stride=1)
        m = m2.reshape(shp)
    return m


def _patched_chunk(query, key, value, g=None, beta=None, **kw):
    c = CTL
    if c.active and query.shape[1] > 1 and c.ctx is not None and c.ctx.shape[0] == query.shape[1]:
        L = c.layer
        if c.capture is not None and (c.cap_layers is None or L in c.cap_layers):
            c.capture[L] = qk_relevance(query[0], key[0], c.qidx, c.ctx).half().cpu()
        cfg = c.cfg
        if cfg is not None and L in cfg["layers"]:
            T, H = query.shape[1], query.shape[2]
            m = torch.ones(T, H, device=query.device, dtype=torch.float32)
            if cfg["src"] == "gdn_qk":  # the proposal: same-layer, per-head question-q / context-k similarity
                r = qk_relevance(query[0], key[0], c.qidx, c.ctx, cfg.get("agg", "max"))
                if cfg.get("head_mean"):
                    r = r.mean(0, keepdim=True).expand_as(r)
                m[c.ctx] = topk_mask(r, cfg["frac"], cfg.get("win", 0)).T
            elif cfg["src"] == "randfrac":  # control: same keep-fraction/window as gdn_qk, random tokens
                gen = torch.Generator(device=query.device).manual_seed(c.seed * 100 + L)
                r = torch.rand(H, int(c.ctx.sum()), device=query.device, generator=gen)
                m[c.ctx] = topk_mask(r, cfg["frac"], cfg.get("win", 0)).T
            elif cfg["src"] == "betatop":   # control: query-agnostic, keep tokens the model itself writes hardest
                r = beta[0][c.ctx].T.float()
                m[c.ctx] = topk_mask(r, cfg["frac"], cfg.get("win", 0)).T
            else:
                pm = c.mask
                m[:] = pm[:, None] if pm.dim() == 1 else pm
            s, d = cfg["s"], cfg["d"]
            beta = beta * (s + (1 - s) * m).to(beta.dtype)[None]
            g = g * (d + (1 - d) * m).to(g.dtype)[None]
            c.stats.setdefault("kept_frac", []).append(m[c.ctx].mean().item())
    return _orig_chunk(query, key, value, g=g, beta=beta, **kw)


def install(model):
    M.torch_chunk_gated_delta_rule = _patched_chunk
    n = 0
    for mod in model.modules():
        if isinstance(mod, M.Qwen3_5GatedDeltaNet):
            mod.register_forward_pre_hook(lambda m, a: setattr(CTL, "layer", m.layer_idx))
            n += 1
    return n


def gdn_layers(model):
    return [i for i, t in enumerate(model.config.layer_types) if t == "linear_attention"]


def attn_layers(model):
    return [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]


class AttnCapture:
    """Captures pre-RoPE q/k (after q_norm/k_norm) of full-attention layers to score question->context relevance."""

    def __init__(self, model, layers):
        self.hooks, self.q, self.k = [], {}, {}
        for i in layers:
            att = model.layers[i].self_attn
            self.hooks.append(att.q_norm.register_forward_hook(lambda m, a, o, i=i: self.q.__setitem__(i, o[0])))
            self.hooks.append(att.k_norm.register_forward_hook(lambda m, a, o, i=i: self.k.__setitem__(i, o[0])))

    def relevance(self, i, qidx, ctx):
        q, k = self.q[i], self.k[i]                     # [T, Hq, D], [T, Hkv, D]
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        sim = torch.einsum("qhd,thd->hqt", q[qidx].float(), k[ctx].float()) / q.shape[-1] ** 0.5
        return sim.amax(1)                              # [H, Tc]

    def clear(self):
        self.q.clear()
        self.k.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
