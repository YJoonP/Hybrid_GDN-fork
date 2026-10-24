"""GDN rollback strategies behind one interface (selected by name, e.g. --rollback fullstate).

The engine calls, for every DFlash step over the n active rows:
  verify_gdn(i, q, k, v, g, beta, conv_full, n) -> o   inside the verify forward of GDN layer i. Must only READ the
                                                         durable states engine.rec[i] / engine.conv[i].
  save(n)                                               after the verify forward (timed as "state save").
  restore(produced, n, accepted)                        after acceptance: durable state of row r <- state after the
                                                         first produced[r] tokens of the block (timed as "restore").
conv_full[r] = [k-1 previous conv inputs | the block's conv inputs]  ([n, C, k-1+S]).
"""
import json
import os

import torch

from .kernels import cond_copy_rows, gdn_replay, gdn_verify


class Rollback:
    name = "base"
    stores_snapshots = True     # False: nothing is written during verification (no store-overhead calibration)
    time_kernels = True         # per-kernel CUDA events around the verify kernels (off inside CUDA graphs)

    def _ev(self):
        if not self.time_kernels:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _ev_end(self, e0):
        if e0 is not None:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            self.events.append((e0, e1))

    def __init__(self, engine, bs, block):
        self.eng, self.bs, self.S = engine, bs, block

    def buffer_bytes(self):
        return {}

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        raise NotImplementedError

    def save(self, n):
        pass

    def restore(self, produced, n, accepted=None):
        raise NotImplementedError

    def reset_stats(self):
        pass

    def store_kwargs(self):
        """gdn_verify keyword arguments that reproduce this strategy's snapshot writes (calibration)."""
        return {}


class FullStateRollback(Rollback):
    """Keep the recurrent state and the short-conv state after every block position; restore = copy, no recompute.

    Recurrent snapshots are written by the verify kernel itself (one fused pass: cheapest way to obtain per-position
    states). Conv snapshots are materialised from conv_full in save(). Buffers are allocated once for `bs` rows and
    reused every step (stale positions are simply overwritten)."""
    name = "fullstate"

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        e = engine
        self.snap_rec = {i: torch.empty(bs, block, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype)
                         for i in e.gdn}
        self.snap_conv = {i: torch.empty(bs, block, *e.conv[i].shape[1:], device=e.dev, dtype=e.conv[i].dtype)
                          for i in e.gdn}
        self.conv_full = {}
        self.events = []          # (start, end) CUDA events around the fused verify kernels of the current step
        self.capture = None       # dict to stash kernel inputs (calibration / state check), or None

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        conv = sum(x.numel() * x.element_size() for x in self.snap_conv.values())
        per_row_layer = (self.snap_rec[self.eng.gdn[0]][0].numel() * 4
                         + self.snap_conv[self.eng.gdn[0]][0].numel() * self.snap_conv[self.eng.gdn[0]].element_size())
        return dict(rec_bytes=rec, conv_bytes=conv, total_bytes=rec + conv, layers=len(self.snap_rec),
                    positions=self.S, batch=self.bs, bytes_per_row_per_layer=per_row_layer)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True)
        self._ev_end(e0)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            # conv state after t+1 block tokens = conv inputs [t+1, t+1+km1) of conv_full
            win = full.unfold(2, km1, 1)[:, :, 1:self.S + 1]                       # [n, C, S, km1]
            self.snap_conv[i][:n].copy_(win.permute(0, 2, 1, 3))
        self.conv_full = {}

    def restore(self, produced, n, accepted=None):
        idx = torch.arange(n, device=produced.device) * self.S + (produced - 1)
        for i in self.snap_rec:
            torch.index_select(self.snap_rec[i][:n].flatten(0, 1), 0, idx, out=self.eng.rec[i][:n])
            torch.index_select(self.snap_conv[i][:n].flatten(0, 1), 0, idx, out=self.eng.conv[i][:n])

    def kernel_ms(self):
        """GPU time (ms) of this step's fused verify kernels (call after a synchronize)."""
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ModeALRollback(Rollback):
    """Snapshot only at the most frequent acceptance length so far (m); restore = copy, or replay (SpecLA-style
    recomputation) from the nearest state at or below the accepted length:
        AL == m : copy the snapshot
        AL >  m : copy the snapshot, replay block tokens [m, AL)
        AL <  m : replay block tokens [0, AL) from the durable state S_0 (never modified by verification)
    AL = tokens committed in a step (pending token + accepted drafts). The histogram is online (past steps only),
    reset per run by reset_stats(); before the first observation m = block // 2; ties go to the smaller AL.
    Conv state: window at m snapshotted; other ALs take their window from the verify pass's conv inputs."""
    name = "modeal"

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        e = engine
        self.snap_rec = {i: torch.empty(bs, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype) for i in e.gdn}
        self.snap_conv = {i: torch.empty(bs, *e.conv[i].shape[1:], device=e.dev, dtype=e.conv[i].dtype) for i in e.gdn}
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None
        self.reset_stats()

    def store_kwargs(self):
        return dict(store_t=self.m - 1)

    def reset_stats(self):
        self.hist = [0] * (self.S + 1)
        self.m = self.S // 2
        self.m_trace = []

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        conv = sum(x.numel() * x.element_size() for x in self.snap_conv.values())
        return dict(rec_bytes=rec, conv_bytes=conv, total_bytes=rec + conv, layers=len(self.snap_rec), positions=1,
                    batch=self.bs)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True, store_t=self.m - 1)
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            self.snap_conv[i][:n].copy_(full[:, :, self.m:self.m + km1])

    def restore(self, produced, n, accepted=None):
        m = self.m
        dev = produced.device
        ge = produced >= m
        rows_ge = ge.nonzero().flatten()
        offs = torch.where(ge, m, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            rec.index_copy_(0, rows_ge, self.snap_rec[i][:n].index_select(0, rows_ge))
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        # update the acceptance-length histogram with this step's (uncapped) acceptance lengths
        for a in (accepted if accepted is not None else produced).tolist():
            self.hist[a] += 1
        best = max(range(1, self.S + 1), key=lambda a: (self.hist[a], -a))
        self.m = best
        self.m_trace.append(best)

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class SpecLARollback(Rollback):
    """Pure recomputation: verification writes no snapshot; after acceptance every row replays its first AL block
    tokens from the durable state S_0 (never modified by verification), reusing the verify pass's k, v, g, beta.
    Conv state = the conv-input window ending at the last committed token."""
    name = "specla"
    stores_snapshots = False

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None

    def buffer_bytes(self):
        return dict(rec_bytes=0, conv_bytes=0, total_bytes=0, layers=len(self.eng.gdn), positions=0, batch=self.bs)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        h0 = self.eng.rec[i][:n]
        o = gdn_verify(q, k, v, g, beta, h0, h0, store=False)     # hs unused when store=False
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        offs = torch.zeros_like(produced)
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=produced.device)[None])[:, None, :]
        for i, (k, v, g, beta) in self.inputs.items():
            gdn_replay(k, v, g, beta, self.eng.rec[i][:n], offs, produced)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ConfKRollback(Rollback):
    """modeal-style restore, but the snapshot rows are chosen per sequence from the draft's confidences, before
    verification. c_i = draft probability of its own token at block position i (i = 1..S-1).
      select "topk": P(j) = (c_1 ... c_j)(1 - c_{j+1}), j = 0..S-2 (j = accepted drafts); keep the K largest
                     (ties -> smaller j)
      select "thr" : j = number of leading drafts with c_i >= thr (consecutive from i = 1)
    Row j = state after the pending token + j drafts (block token index j). Row S-1 (all accepted) is always kept,
    so a sequence has K+1 (topk) or 2 (thr) slots. The verify kernel writes only those rows (slot map [n, S]).
    Restore with A = committed tokens (row A-1): hit -> copy the slot; miss -> copy the largest kept row r < A-1
    (or start from S_0) and replay GDN rows r+1..A-1 with the verify pass's k, v, g, beta. Conv state = conv-input
    window ending at row A-1; attention KV / draft features come from the verify pass (as modeal)."""
    name = "confk"
    needs_confidence = True

    def __init__(self, engine, bs, block, select="topk", k=1, thr=0.5, slots=None):
        super().__init__(engine, bs, block)
        assert select in ("topk", "thr")
        e = engine
        self.select, self.k, self.thr = select, k, thr
        self.ns = slots or ((k + 1) if select == "topk" else 2)
        self.snap_rec = {i: torch.empty(bs, self.ns, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype)
                         for i in e.gdn}
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None
        self.slot_map = self.rows = None
        self.reset_stats()

    def reset_stats(self):
        if getattr(self, "stat", None) is not None:     # in place: a captured CUDA graph keeps adding to these tensors
            for x in self.stat.values():
                x.zero_()
            return
        z = lambda: torch.zeros((), dtype=torch.long, device=self.eng.dev)  # noqa: E731
        self.stat = dict(n=z(), hit=z(), from_s0=z(), replay=z(),
                         replay_hist=torch.zeros(self.S + 1, dtype=torch.long, device=self.eng.dev))

    def pred_stats(self):
        s = {k: v.tolist() for k, v in self.stat.items()}
        n = max(s["n"], 1)
        return dict(select=self.select, k=self.k if self.select == "topk" else None,
                    thr=self.thr if self.select == "thr" else None, slots_per_seq=self.ns, row_steps=s["n"],
                    hit_rate=s["hit"] / n, from_s0_rate=s["from_s0"] / n, replay_tokens_per_row_step=s["replay"] / n,
                    replay_hist={str(i): c for i, c in enumerate(s["replay_hist"]) if c})

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        return dict(rec_bytes=rec, conv_bytes=0, total_bytes=rec, layers=len(self.snap_rec), positions=self.ns,
                    batch=self.bs)

    def store_kwargs(self):
        return dict(slot_map=self.slot_map)

    def predict(self, conf):
        """conf [n, S-1] -> rows [n, ns] (kept block rows; -1 = unused) and slot_map [n, S]."""
        n, S = conf.shape[0], self.S
        dev = conf.device
        if self.select == "topk":
            P = torch.cumprod(conf, 1)                                          # c_1 .. c_1...c_{S-1}
            Pa = torch.cat([torch.ones_like(P[:, :1]), P[:, :-1]], 1)           # P_0 .. P_{S-2}
            pj = Pa * (1 - conf)                                                # P(j), j = 0..S-2
            chosen = torch.argsort(-pj, dim=1, stable=True)[:, : self.k]        # ties -> smaller j
        else:
            chosen = (conf >= self.thr).long().cumprod(1).sum(1, keepdim=True)  # leading run length, 0..S-1
        rows = torch.cat([chosen, torch.full((n, 1), S - 1, device=dev, dtype=torch.long)], 1)
        rows = torch.where((rows == S - 1) & (torch.arange(rows.shape[1], device=dev) < rows.shape[1] - 1), -1, rows)
        smap = torch.full((n, S), -1, device=dev, dtype=torch.int32)
        valid = rows >= 0
        slot = torch.arange(self.ns, device=dev).expand(n, -1)
        smap[torch.arange(n, device=dev)[:, None].expand(-1, self.ns)[valid], rows[valid]] = slot[valid].int()
        self.rows, self.slot_map = rows, smap

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True, slot_map=self.slot_map)
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]                                          # row A-1
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)                                                # largest kept row <= A-1
        have = best >= 0
        rows_have = have.nonzero().flatten()
        flat = rows_have * self.ns + slot[rows_have]
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            rec.index_copy_(0, rows_have, self.snap_rec[i][:n].flatten(0, 1).index_select(0, flat))
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        self.stat["n"] += n
        self.stat["hit"] += (lens == 0).sum()
        self.stat["from_s0"] += (~have).sum()
        self.stat["replay"] += lens.sum()
        self.stat["replay_hist"].index_add_(0, lens, torch.ones_like(lens))

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ModeALStatic(ModeALRollback):
    """CUDA-graph version of ModeALRollback (same decisions): m and the AL histogram live on the device and are updated
    inside the captured step; the snapshot position reaches the verify kernel as a slot map; the restore copies the
    snapshot only for rows with AL >= m (conditional row copy). Before the first observation the histogram holds a
    prior of 0.5 at block//2, so m = block//2 until then; argmax keeps the first (smaller) AL on ties."""

    def __init__(self, engine, bs, block):
        dev = engine.dev
        self.m_dev = torch.full((), block // 2, dtype=torch.long, device=dev)
        self.hist_dev = torch.zeros(block + 1, device=dev)
        super().__init__(engine, bs, block)

    def reset_stats(self):
        super().reset_stats()
        self.hist_dev.zero_()
        self.hist_dev[self.S // 2] = 0.5
        self.m_dev.fill_(self.S // 2)

    def observed_hist(self):
        h = self.hist_dev.clone()
        h[self.S // 2] -= 0.5
        return [int(round(x)) for x in h.tolist()]

    def store_kwargs(self):
        return dict(slot_map=self._smap(self.eng.n))

    def _smap(self, n):
        pos = torch.arange(self.S, device=self.m_dev.device)[None]
        return torch.where(pos == self.m_dev - 1, 0, -1).to(torch.int32).expand(n, self.S).contiguous()

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n].unsqueeze(1), store=True,
                       slot_map=self._smap(n))
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            idx = (self.m_dev + torch.arange(km1, device=full.device)).view(1, 1, km1).expand(n, full.shape[1], km1)
            self.snap_conv[i][:n].copy_(torch.gather(full, 2, idx))

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        m = self.m_dev
        have = produced >= m
        src = torch.where(have, torch.arange(n, device=dev), -1)
        offs = torch.where(have, m, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            cond_copy_rows(rec, self.snap_rec[i][:n], src)
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        valid = torch.arange(n, device=dev) < self.eng.n_dev          # padded rows of a batch bucket do not count
        a = accepted if accepted is not None else produced
        self.hist_dev.index_add_(0, torch.where(valid, a, 0), valid.float())
        self.m_dev.copy_(torch.argmax(self.hist_dev[1:]) + 1)


class ConfKStatic(ConfKRollback):
    """CUDA-graph version of ConfKRollback (same selection and restore): the slot map is built with a fixed-shape
    scatter and the snapshot is gathered with a conditional row copy; statistics skip padded rows."""

    def predict(self, conf):
        n, S = conf.shape[0], self.S
        dev = conf.device
        if self.select == "topk":
            P = torch.cumprod(conf, 1)
            Pa = torch.cat([torch.ones_like(P[:, :1]), P[:, :-1]], 1)
            chosen = torch.argsort(-(Pa * (1 - conf)), dim=1, stable=True)[:, : self.k]
        else:
            chosen = (conf >= self.thr).long().cumprod(1).sum(1, keepdim=True)
        rows = torch.cat([chosen, torch.full((n, 1), S - 1, device=dev, dtype=torch.long)], 1)
        rows = torch.where((rows == S - 1) & (torch.arange(rows.shape[1], device=dev) < rows.shape[1] - 1), -1, rows)
        smap = torch.full((n, S + 1), -1, device=dev, dtype=torch.int32)      # column S collects unused slots
        slot = torch.arange(self.ns, device=dev, dtype=torch.int32).expand(n, -1)
        smap.scatter_(1, torch.where(rows >= 0, rows, S), slot)
        self.rows, self.slot_map = rows, smap[:, :S].contiguous()

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)
        have = best >= 0
        src = torch.where(have, torch.arange(n, device=dev) * self.ns + slot, -1)
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            cond_copy_rows(rec, self.snap_rec[i][:n].flatten(0, 1), src)
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        valid = torch.arange(n, device=dev) < self.eng.n_dev
        vl = valid.long()
        self.stat["n"] += vl.sum()
        self.stat["hit"] += ((lens == 0) & valid).sum()
        self.stat["from_s0"] += ((~have) & valid).sum()
        self.stat["replay"] += (lens * vl).sum()
        self.stat["replay_hist"].index_add_(0, lens, vl)


def load_conf_head(path, device):
    """conf-K method B head (confk-head-B): net = Linear(5120, 1) ("linear") or Linear(5120, 256) - SiLU -
    Linear(256, 1) ("mlp"), with the input normalisation mu / sd stored next to the weights."""
    d = torch.load(path, map_location=device, weights_only=True)
    dim = d["mu"].shape[0]
    if d["kind"] == "linear":
        net = torch.nn.Linear(dim, 1)
    elif d["kind"] == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.SiLU(), torch.nn.Linear(256, 1))
    else:
        raise ValueError(f"conf head must be linear or mlp, got {d['kind']}")
    net.load_state_dict(d["state"])
    return dict(kind=d["kind"], net=net.to(device).float().eval().requires_grad_(False), mu=d["mu"].float().to(device),
                sd=d["sd"].float().to(device))


class _ConfHead:
    """conf-K method B: the softmax probability c_k is replaced by a learned head on the draft side,
    c_k = sigmoid(net(([h_k ; E[prev_k]] - mu) / sd)), k = 1..S-1, where h_k is the draft's final-norm hidden at draft
    position k (the lm_head input), prev_k the verify-block token before it (the pending token for k = 1) and E the
    target's input embeddings. Selection and restore are those of conf-K."""
    name = "confkb"
    needs_confidence = "hidden"

    def __init__(self, engine, bs, block, select="topk", k=1, thr=0.5, head=None):
        super().__init__(engine, bs, block, select, k, thr)
        self.head_path = head
        self.head = load_conf_head(head, engine.dev)

    def confidence(self, hidden, prev):
        E = self.eng.t.get_input_embeddings().weight
        x = torch.cat([hidden.float(), E[prev].float()], -1)
        c = torch.sigmoid(self.head["net"]((x - self.head["mu"]) / self.head["sd"])[..., 0])
        self.last = (hidden, prev, c)       # references only (static graph tensors after capture): integration check
        return c

    def pred_stats(self):
        return dict(super().pred_stats(), conf_source="head", head=self.head_path, head_kind=self.head["kind"])


class ConfKBRollback(_ConfHead, ConfKRollback):
    pass


class ConfKBStatic(_ConfHead, ConfKStatic):
    pass


class _Oracle:
    """Upper bound for AL prediction: one snapshot slot per sequence, at the row the step will actually commit (A - 1),
    read from a recorded run of the same engine and batch size (outputs and acceptance are identical across
    strategies). load_oracle() fills the device buffer on the host before the timed step (not counted); pre_verify()
    turns it into the slot map after the draft. Restore is conf-K's, so a wrong guess still restores exactly."""
    name = "oracle"
    needs_confidence = False

    def __init__(self, engine, bs, block, trace=None):
        super().__init__(engine, bs, block, slots=1)
        self.trace_path = os.path.join(trace, "raw", f"dflash_bs{bs}_run1.jsonl")
        self.trace = {r["order"]: r["accept"] for r in map(json.loads, open(self.trace_path))}   # engine row id = order
        self.oracle_host = torch.ones(bs, dtype=torch.long).pin_memory()
        self.oracle_dev = torch.ones(bs, dtype=torch.long, device=engine.dev)

    def load_oracle(self, n):
        for r, row in enumerate(self.eng.rows[:n]):
            t, k = self.trace.get(row["qid"], ()), len(row["accept"])
            self.oracle_host[r] = t[k] if k < len(t) else 1
        self.oracle_dev[:n].copy_(self.oracle_host[:n], non_blocking=True)

    def pre_verify(self, n):
        row = (self.oracle_dev[:n] - 1).clamp(0, self.S - 1)[:, None]
        smap = torch.full((n, self.S), -1, device=row.device, dtype=torch.int32)
        smap.scatter_(1, row, torch.zeros_like(row, dtype=torch.int32))
        self.rows, self.slot_map = row, smap

    def pred_stats(self):
        return dict(super().pred_stats(), oracle_trace=self.trace_path)


class OracleRollback(_Oracle, ConfKRollback):
    pass


class OracleStatic(_Oracle, ConfKStatic):
    pass


class OracleCopyStatic(OracleStatic):
    """oracle with the lightest restore: the single snapshot slot always holds the committed row, so the restore is
    one copy of the recurrent state and one gather of the conv window per GDN layer -- no fallback replay, no slot
    search, no device statistics. Correctness relies on the oracle being right; check_oracle() compares the oracle
    with the produced counts on the host after the timed step and counts mismatches."""
    name = "oracle_copy"

    def reset_stats(self):
        super().reset_stats()
        self.n_rows = self.n_miss = 0

    def restore(self, produced, n, accepted=None):
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=produced.device)[None])[:, None, :]
        for i in self.snap_rec:
            self.eng.rec[i][:n].copy_(self.snap_rec[i][:n, 0])
            full = self.conv_full[i]
            torch.gather(full, 2, widx.expand(-1, full.shape[1], -1), out=self.eng.conv[i][:n])
        self.inputs, self.conv_full = {}, {}

    def check_oracle(self, prod):
        self.n_rows += len(prod)
        self.n_miss += sum(int(self.oracle_host[r]) != a for r, a in enumerate(prod))

    def pred_stats(self):
        n = max(self.n_rows, 1)
        return dict(select="oracle_copy", slots_per_seq=1, row_steps=self.n_rows, oracle_misses=self.n_miss,
                    hit_rate=1 - self.n_miss / n, from_s0_rate=0.0, replay_tokens_per_row_step=0.0,
                    oracle_trace=self.trace_path)


ROLLBACKS = {"fullstate": FullStateRollback, "modeal": ModeALRollback, "specla": SpecLARollback,
             "confk": ConfKRollback, "confkb": ConfKBRollback, "oracle": OracleRollback}
STATIC_ROLLBACKS = {"fullstate": FullStateRollback, "modeal": ModeALStatic, "specla": SpecLARollback,
                    "confk": ConfKStatic, "confkb": ConfKBStatic, "oracle": OracleStatic,
                    "oracle_copy": OracleCopyStatic}                                          # CUDA-graph capturable


def make_rollback(name, engine, bs, block, static=False, **kwargs):
    table = STATIC_ROLLBACKS if static else ROLLBACKS
    if name not in table:
        raise ValueError(f"unknown rollback {name!r}; available: {sorted(table)}")
    return table[name](engine, bs, block, **kwargs)
