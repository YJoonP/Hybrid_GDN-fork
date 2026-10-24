"""Triton kernels for Gated DeltaNet verification with state snapshots, and for replaying accepted tokens."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "store_t"])
def _gdn_verify_kernel(q, k, v, g, beta, o, h0, hs, scale, T, store_t, slot_map, HV: tl.constexpr, K: tl.constexpr,
                       V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
                       MODE: tl.constexpr, NS: tl.constexpr):
    """Gated delta rule over T tokens from h0 (read only), same op order as fla fused_recurrent.
    MODE 0: no snapshot. MODE 1: state after every token -> hs[n, t] (full state).
    MODE 2: state after token index store_t only -> hs[n] (single snapshot).
    MODE 4: per-row slots. slot_map[n, t] = slot index s (or -1): state after token index t -> hs[n, s] (NS slots)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    bos = i_n * T
    p_q = q + (bos * HV + i_hv) * K + o_k
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    if MODE == 1:
        p_h = hs + (bos * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
    else:
        p_h = hs + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mask_h, other=0).to(tl.float32)
    for t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_b).to(tl.float32)
        b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if MODE == 1:
            tl.store(p_h, b_h, mask=mask_h)
            p_h += HV * K * V
        if MODE == 2:
            if t == store_t:
                tl.store(p_h, b_h, mask=mask_h)
        if MODE == 4:
            s = tl.load(slot_map + i_n * T + t)
            if s >= 0:
                tl.store(p_h + (i_n * (NS - 1) + s) * HV * K * V, b_h, mask=mask_h)
        p_q += HV * K
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
        p_o += HV * V


def gdn_verify(q, k, v, g, beta, h0, hs, store=True, store_t=None, slot_map=None):
    """q, k: [B, T, HV, K]; v: [B, T, HV, V]; g, beta: [B, T, HV]; h0: [B, HV, K, V] fp32 (not modified).
    store_t None: hs [B, T, HV, K, V] gets the state after every token (when store).
    store_t = t : hs [B, HV, K, V] gets the state after token index t (when store).
    slot_map    : int tensor [B, T] (slot or -1), hs [B, NS, HV, K, V] gets row n's state after token t in slot
                  slot_map[n, t]. Returns o [B, T, HV, V]."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    assert h0.is_contiguous() and hs.is_contiguous()
    mode = 0 if not store else (4 if slot_map is not None else (1 if store_t is None else 2))
    smap = slot_map.to(torch.int32).contiguous() if slot_map is not None else h0     # dummy pointer unless MODE 4
    ns = hs.shape[1] if mode == 4 else 1
    o = torch.empty_like(v)
    grid = (triton.cdiv(V, 8), B * HV)
    _gdn_verify_kernel[grid](q, k, v, g, beta, o, h0, hs, K ** -0.5, T, -1 if store_t is None else store_t, smap,
                             HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=8, USE_QK_L2NORM=True, MODE=mode, NS=ns,
                             num_warps=1, num_stages=3)
    return o


@triton.jit(do_not_specialize=["T"])
def _gdn_replay_kernel(k, v, g, beta, h, offs, lens, T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """In place: h[n] <- h[n] advanced over block tokens [offs[n], offs[n] + lens[n]) (fla fused_recurrent order)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    n_t = tl.load(lens + i_n)
    if n_t > 0:
        off = tl.load(offs + i_n)
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        bos = i_n * T + off
        p_k = k + (bos * HV + i_hv) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_g = g + bos * HV + i_hv
        p_b = beta + bos * HV + i_hv
        p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        for _ in range(0, n_t):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_beta = tl.load(p_b).to(tl.float32)
            b_h *= tl.exp(tl.load(p_g).to(tl.float32))
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v[None, :]
            p_k += HV * K
            p_v += HV * V
            p_g += HV
            p_b += HV
        tl.store(p_h, b_h, mask=mask_h)


def gdn_replay(k, v, g, beta, h, offs, lens):
    """k: [B, T, HV, K]; v: [B, T, HV, V]; g, beta: [B, T, HV]; h: [B, HV, K, V] fp32 updated in place."""
    k, v, g, beta = (x.contiguous() for x in (k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    assert h.is_contiguous()
    grid = (triton.cdiv(V, 8), B * HV)
    _gdn_replay_kernel[grid](k, v, g, beta, h, offs.to(torch.int32), lens.to(torch.int32), T, HV=HV, K=K, V=V,
                             BK=triton.next_power_of_2(K), BV=8, USE_QK_L2NORM=True, num_warps=1, num_stages=3)


@triton.jit
def _cond_copy_kernel(dst, src, idx, D, BLOCK: tl.constexpr):
    """dst[r, :] = src[idx[r], :] where idx[r] >= 0; rows with idx[r] < 0 are left untouched (no memory traffic)."""
    i_b, r = tl.program_id(0), tl.program_id(1)
    s = tl.load(idx + r)
    if s >= 0:
        o = i_b * BLOCK + tl.arange(0, BLOCK)
        m = o < D
        tl.store(dst + r.to(tl.int64) * D + o, tl.load(src + s.to(tl.int64) * D + o, mask=m), mask=m)


def cond_copy_rows(dst, src, idx):
    """Fixed-shape gather of whole rows for CUDA-graph capture. dst: [B, ...], src: [M, ...] (same row shape, both
    contiguous), idx: int [B] (row of src, or -1 to skip)."""
    assert dst.is_contiguous() and src.is_contiguous()
    B, D = dst.shape[0], dst[0].numel()
    BLOCK = 4096
    _cond_copy_kernel[(triton.cdiv(D, BLOCK), B)](dst, src, idx.contiguous(), D, BLOCK=BLOCK)
