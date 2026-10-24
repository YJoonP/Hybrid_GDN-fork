"""CUDA-graph version of the continuous-batching engine.

One decode step is captured as one CUDA graph per batch bucket (1, 2, 4, ..., bs) and replayed:
  DFlash: draft (+ confidence / slot selection) -> verify -> state save -> accept & token write -> state restore ->
          length update;   AR: forward -> token write & length update.
A step with n active rows replays the smallest bucket >= n; rows n.. of the bucket are padding (free slots whose
results are discarded and whose buffers are overwritten when a sequence is inserted there).

Static shapes: attention always spans the whole KV buffer (max_len) under the causal / validity mask, the draft
context is always the last `block` positions (a prompt's context is written into the draft KV when the sequence is
inserted, timed and counted as draft work), and the rollback strategies are the fixed-shape implementations
(rollback.STATIC_ROLLBACKS). The only host synchronisation per step is the read of the produced counts after replay.
Section times are GPU times between CUDA events recorded inside the graph; the step total is wall time.
"""
import torch
import torch.nn.functional as F
from dflash.model import extract_context_feature
from transformers import DynamicCache

from .engine import Engine, _rope, sync_time

SECTIONS_DFLASH = ["draft", "verify", "save", "accept", "restore", "kv_truncate"]
SECTIONS_AR = ["verify", "accept"]


def buckets_for(bs):
    out, b = [], 1
    while b < bs:
        out.append(b)
        b *= 2
    return out + [bs]


class GraphEngine(Engine):
    def __init__(self, target, draft, method, bs, max_prompt, max_new, block=16, rollback="fullstate",
                 rb_kwargs=None, debug=False):
        super().__init__(target, draft, method, bs, max_prompt, max_new, block, rollback, rb_kwargs,
                         static_rollback=True, ctx_cols=block)
        dev = self.dev
        self.n_dev = torch.zeros((), dtype=torch.long, device=dev)
        self.prod_buf = torch.ones(bs, dtype=torch.long, device=dev)
        self.buckets = buckets_for(bs)
        self.graphs, self.events = {}, {}
        self.pool = None
        self.ingest_s = 0.0
        if self.dflash:
            self.rb.time_kernels = False
        self.debug = debug and self.dflash
        if self.debug:
            self._init_debug()

    # ------------------------------------------------------------ insert (prefill + draft context)
    @torch.inference_mode()
    def insert(self, qid, ids):
        r, n = self.n, ids.shape[0]
        t0 = sync_time()
        cache = DynamicCache(config=self.cfg)
        out = self.t(ids[None].to(self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1,
                     output_hidden_states=self.dflash)
        for i in self.att:
            self.K[i][r, :, :n] = cache.layers[i].keys[0]
            self.V[i][r, :, :n] = cache.layers[i].values[0]
        for i in self.gdn:
            self.conv[i][r] = cache.layers[i].conv_states[0][0, :, -self.conv[i].shape[-1]:]
            self.rec[i][r] = cache.layers[i].recurrent_states[0][0]
        self.out[r, :n] = ids.to(self.dev)
        self.out[r, n] = out.logits[0, -1].argmax()
        self.start[r] = n
        self.n_prompt[r] = n
        self.max_total[r] = n + self.max_new
        feat = extract_context_feature(out.hidden_states, self.d.target_layer_ids)[0] if self.dflash else None
        del cache, out
        t1 = sync_time()
        if self.dflash:
            self._ingest(r, feat)
            self.ctx_cnt[r] = 0
            t2 = sync_time()
            self.ingest_s += t2 - t1
        self.rows.append(dict(qid=qid, prompt_len=n, t_decode=0.0, accept=[]))
        self.n += 1
        return t1 - t0

    def _ingest(self, r, feat):
        """Draft context K/V of the prompt (what the first draft step does for the context part)."""
        d, n = self.d, feat.shape[0]
        th = d.hidden_norm(d.fc(feat[None]))
        cos, sin = d.rotary_emb(th, torch.arange(n, device=self.dev)[None])
        for li, layer in enumerate(d.layers):
            a = layer.self_attn
            kc = a.k_norm(a.k_proj(th).view(1, n, -1, a.head_dim)).transpose(1, 2)
            vc = a.v_proj(th).view(1, n, -1, a.head_dim).transpose(1, 2)
            self.DK[li][r, :, :n] = _rope(kc, cos, sin)[0]
            self.DV[li][r, :, :n] = vc[0]

    # ------------------------------------------------------------ captured step bodies (fixed shapes)
    def _body_dflash(self, nb, ev):
        S, dev, L = self.S, self.dev, self.max_len
        jS = torch.arange(S, device=dev)[None]
        start = self.start[:nb]
        pos = (start[:, None] + jS).clamp(max=L - 1)
        block = self.out[:nb].gather(1, pos)
        block[:, 1:] = self.d.mask_token_id
        if self.debug:
            for i in self.gdn:
                self.dbg["pre_rec"][i].copy_(self.rec[i][:1])
        ev[0].record()
        need_conf = getattr(self.rb, "needs_confidence", False)
        toks, conf = self.draft_forward(self.ctx_feat[:nb], self.ctx_cnt[:nb], start, block, need_conf, Lc=L)
        block = torch.cat([block[:, :1], toks], 1)
        if need_conf == "hidden":
            conf = self.rb.confidence(conf, block[:, :-1])
        if need_conf:
            self.rb.predict(conf)
        if hasattr(self.rb, "pre_verify"):
            self.rb.pre_verify(nb)
        ev[1].record()
        post, f_new = self.target_forward(block, pos, verify=True, L=L)
        ev[2].record()
        self.rb.save(nb)
        ev[3].record()
        acc = (block[:, 1:] == post[:, :-1]).long().cumprod(1).sum(1)
        bonus = post.gather(1, acc[:, None])[:, 0]
        produced = torch.minimum(acc + 1, self.max_total[:nb] - start - 1).clamp(1, S)
        new = torch.where(jS < acc[:, None], F.pad(block[:, 1:], (0, 1)), bonus[:, None])
        wpos = (start[:, None] + 1 + jS).clamp(max=L - 1)
        cur = self.out[:nb].gather(1, wpos)
        self.out[:nb].scatter_(1, wpos, torch.where(jS < produced[:, None], new, cur))
        self.prod_buf[:nb].copy_(produced)
        ev[4].record()
        self.rb.restore(produced, nb, acc + 1)
        ev[5].record()
        self.start[:nb] += produced
        self.ctx_feat[:nb].copy_(f_new)
        self.ctx_cnt[:nb].copy_(produced)
        ev[6].record()

    def _body_ar(self, nb, ev):
        L = self.max_len
        start = self.start[:nb]
        pos = start[:, None].clamp(max=L - 1)
        ids = self.out[:nb].gather(1, pos)
        ev[0].record()
        post, _ = self.target_forward(ids, pos, verify=False, L=L)
        ev[1].record()
        self.out[:nb].scatter_(1, (start + 1).clamp(max=L - 1)[:, None], post)
        self.start[:nb] += 1
        ev[2].record()

    @torch.inference_mode()
    def capture(self):
        """Capture every bucket once, before any sequence is active (warm-up runs execute on scratch state)."""
        self.pool = torch.cuda.graph_pool_handle()
        body = self._body_dflash if self.dflash else self._body_ar
        n_ev = len(SECTIONS_DFLASH if self.dflash else SECTIONS_AR) + 1
        for nb in sorted(self.buckets, reverse=True):
            ev = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(n_ev)]
            self.n_dev.fill_(nb)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body(nb, ev)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body(nb, ev)
            torch.cuda.synchronize()
            self.graphs[nb], self.events[nb] = g, ev
        self._scrub()

    def _scrub(self):
        """Reset state touched by the warm-up runs."""
        for x in (self.out, self.start, self.n_prompt, self.max_total):
            x.zero_()
        if self.dflash:
            self.ctx_cnt.zero_()
            self.rb.reset_stats()
        self.prod_buf.fill_(1)
        self.reset()

    # ------------------------------------------------------------ one step
    @torch.inference_mode()
    def step(self):
        n = self.n
        nb = next(b for b in self.buckets if b >= n)
        self.n_dev.fill_(n)
        if self.dflash and hasattr(self.rb, "load_oracle"):
            self.rb.load_oracle(n)                      # oracle AL from a recorded run, outside the timed step
        t0 = sync_time()
        self.graphs[nb].replay()
        t1 = sync_time()
        ev = self.events[nb]
        names = SECTIONS_DFLASH if self.dflash else SECTIONS_AR
        T = {nm: ev[k].elapsed_time(ev[k + 1]) / 1e3 for k, nm in enumerate(names)}
        T["_wall"] = t1 - t0
        prod = self.prod_buf[:n].tolist()
        if self.dflash and hasattr(self.rb, "check_oracle"):
            self.rb.check_oracle(prod)                  # host side, after the timed step
        if self.dflash and hasattr(self.rb, "m_dev"):
            self.rb.m_trace.append(int(self.rb.m_dev))
        for r in range(n):
            self.rows[r]["t_decode"] += t1 - t0
            self.rows[r]["accept"].append(prod[r])
        finished = (self.start[:n] + 1 >= self.max_total[:n]).nonzero().flatten().tolist()
        return T, prod, finished, 0.0

    def pop_ingest(self):
        t, self.ingest_s = self.ingest_s, 0.0
        return t

    # ------------------------------------------------------------ Phase 3 support
    def _init_debug(self):
        """Static copies of row 0's pre-step GDN state and verify inputs (written inside the captured step)."""
        S, c = self.S, self.cfg
        hv, kd, vd = c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        dev, bf = self.dev, torch.bfloat16
        self.dbg = dict(pre_rec={i: torch.zeros_like(self.rec[i][:1]) for i in self.gdn},
                        k={i: torch.zeros(1, S, hv, kd, device=dev, dtype=bf) for i in self.gdn},
                        v={i: torch.zeros(1, S, hv, vd, device=dev, dtype=bf) for i in self.gdn},
                        g={i: torch.zeros(1, S, hv, device=dev) for i in self.gdn},
                        beta={i: torch.zeros(1, S, hv, device=dev, dtype=bf) for i in self.gdn},
                        full={i: torch.zeros(1, self.conv[i].shape[1], self.conv[i].shape[2] + S, device=dev,
                                             dtype=bf) for i in self.gdn})
        orig = self.rb.verify_gdn

        def spy(i, q, k, v, g, beta, conv_full, n):
            for name, x in (("k", k), ("v", v), ("g", g), ("beta", beta), ("full", conv_full)):
                self.dbg[name][i].copy_(x[:1])
            return orig(i, q, k, v, g, beta, conv_full, n)

        self.rb.verify_gdn = spy
