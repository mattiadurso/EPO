"""Fused Triton kernels for the EPO geometric hot paths.

This module hosts two custom autograd ops:

1. **project + bilinear-DT-sample** — used in the per-step forward batch.
   Replaces the chain ``xyz_world @ R^T + t → K @ xyz_cam / z →
   F.grid_sample(dt_field, uv)`` with a single Triton kernel and an
   analytical backward.
2. **unproject (pixel → world)** — used once per iteration in
   ``unproject_edges_to_3D``. Replaces
   ``K_inv @ [u, v, 1] · depth → R_inv @ xyz_cam + t_inv`` with one fused
   kernel and an analytical backward.

Why custom autograd in both cases:
  * Each reference chain expands into 4–6 PyTorch ops, every one adding an
    autograd node + intermediate tensor allocation. The custom Function
    collapses each chain to one node.
  * The non-grad inputs (``dt_fields``, ``xy0``) become pure gathers in the
    backward — no scatter / atomic accumulation needed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _project_sample_fwd_kernel(
    XYZ_ptr,  # (B, N, 3) float
    K_ptr,  # (B, 3, 3) float
    P_ptr,  # (B, 4, 4) float
    DT_ptr,  # (N_img, H, W) float — *source* DT tensor (not gathered)
    DT_IDX_ptr,  # (B,) int64 — maps batch row → image index in DT_ptr
    IMG_HW_ptr,  # (B, 2) int32 — per-row real (H, W) of the target image
    OUT_ptr,  # (B, N) float — residuals
    MASK_ptr,  # (B, N) uint8 — 1 if inside
    # Saved intermediates for backward
    XC_ptr,
    YC_ptr,
    ZC_ptr,
    DSDU_ptr,
    DSDV_ptr,
    # Sizes — H, W here are the *padded* canvas (DT memory layout)
    B,
    N,
    H,
    W,
    # Strides (all element-wise)
    s_xyz_b,
    s_xyz_n,
    s_K_b,
    s_P_b,
    s_dt_b,
    s_dt_h,
    s_hw_b,
    s_out_b,
    BLOCK_N: tl.constexpr,
):
    """Forward: project xyz_world → pixel → bilinear-sample dt_field.

    Grid layout: ``(B, ceil(N / BLOCK_N))``. Each program handles one batch row
    and a block of ``BLOCK_N`` points.

    DT_ptr is the *source* tensor — same memory for every program; ``DT_IDX_ptr``
    tells this program which image's DT field to sample. This avoids
    materialising a per-batch ``(B, H, W)`` copy, which on 518² fields is a
    ~550 MB gather per mini-batch.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Load xyz_world (B, N, 3) -----------------------------------------
    xyz_row = pid_b * s_xyz_b + n_offs * s_xyz_n
    x = tl.load(XYZ_ptr + xyz_row + 0, mask=n_mask, other=0.0)
    y = tl.load(XYZ_ptr + xyz_row + 1, mask=n_mask, other=0.0)
    z = tl.load(XYZ_ptr + xyz_row + 2, mask=n_mask, other=0.0)

    # ---- Load P[b] (4x4, row-major) ---------------------------------------
    Pb = P_ptr + pid_b * s_P_b
    R00 = tl.load(Pb + 0)
    R01 = tl.load(Pb + 1)
    R02 = tl.load(Pb + 2)
    t0 = tl.load(Pb + 3)
    R10 = tl.load(Pb + 4)
    R11 = tl.load(Pb + 5)
    R12 = tl.load(Pb + 6)
    t1 = tl.load(Pb + 7)
    R20 = tl.load(Pb + 8)
    R21 = tl.load(Pb + 9)
    R22 = tl.load(Pb + 10)
    t2 = tl.load(Pb + 11)

    # ---- Load K[b] (only fx, fy, cx, cy) ----------------------------------
    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)  # K[0,0]
    cx = tl.load(Kb + 2)  # K[0,2]
    fy = tl.load(Kb + 4)  # K[1,1]
    cy = tl.load(Kb + 5)  # K[1,2]

    # ---- xyz_cam = R @ xyz + t -------------------------------------------
    xc = R00 * x + R01 * y + R02 * z + t0
    yc = R10 * x + R11 * y + R12 * z + t1
    zc = R20 * x + R21 * y + R22 * z + t2

    # ---- Perspective divide ----------------------------------------------
    # Guard against divide-by-zero: if zc is ~0 we mark as outside below.
    inv_z = 1.0 / zc
    u = fx * xc * inv_z + cx
    v = fy * yc * inv_z + cy

    # ---- Bounds + numerical-validity check ---------------------------
    # Inside iff u, v are inside the *real* (unpadded) target image *and*
    # zc > 0 *and* u/v are finite. The finiteness check guards against
    # zc ⇒ 0⁺ producing ±Inf/NaN in u, v (which the comparison-based bounds
    # test alone would reject, but only by accident — being explicit makes
    # the intent obvious).
    #
    # IMG_HW_ptr is (B, 2) int32 with (H_real, W_real) per row. Using the
    # padded canvas H, W here would let projections that land in the padded
    # zone count as "inside" on mixed-resolution datasets (mipnerf360).
    hw_off = pid_b * s_hw_b
    H_real = tl.load(IMG_HW_ptr + hw_off + 0)
    W_real = tl.load(IMG_HW_ptr + hw_off + 1)
    Wf = tl.cast(W_real, tl.float32)
    Hf = tl.cast(H_real, tl.float32)
    finite_uv = (u == u) & (v == v) & (u * 0.0 == 0.0) & (v * 0.0 == 0.0)
    inside = (u >= 0.0) & (u < Wf) & (v >= 0.0) & (v < Hf) & (zc > 0.0) & finite_uv

    # ---- Bilinear sample (with border-padding semantics) -----------------
    # Clamp corner indices to [0, W-1] / [0, H-1] to mimic padding_mode='border'.
    u_safe = tl.where(inside, u, 0.0)
    v_safe = tl.where(inside, v, 0.0)
    u0 = tl.floor(u_safe).to(tl.int32)
    v0 = tl.floor(v_safe).to(tl.int32)
    du = u_safe - tl.cast(u0, tl.float32)
    dv = v_safe - tl.cast(v0, tl.float32)
    u1 = tl.minimum(u0 + 1, W - 1)
    v1 = tl.minimum(v0 + 1, H - 1)
    u0c = tl.maximum(u0, 0)
    v0c = tl.maximum(v0, 0)

    # Look up which image this batch row reads from. The source DT tensor is
    # shared across batches; no per-batch (B, H, W) copy needed.
    img_idx = tl.load(DT_IDX_ptr + pid_b)
    base = img_idx * s_dt_b
    load_mask = inside & n_mask
    d00 = tl.load(DT_ptr + base + v0c * s_dt_h + u0c, mask=load_mask, other=0.0)
    d01 = tl.load(DT_ptr + base + v0c * s_dt_h + u1, mask=load_mask, other=0.0)
    d10 = tl.load(DT_ptr + base + v1 * s_dt_h + u0c, mask=load_mask, other=0.0)
    d11 = tl.load(DT_ptr + base + v1 * s_dt_h + u1, mask=load_mask, other=0.0)

    w00 = (1.0 - du) * (1.0 - dv)
    w01 = du * (1.0 - dv)
    w10 = (1.0 - du) * dv
    w11 = du * dv
    sampled = d00 * w00 + d01 * w01 + d10 * w10 + d11 * w11
    sampled = tl.where(inside, sampled, 0.0)

    # ---- Bilinear gradient (saved for bwd — avoids re-gather in bwd) ----
    # ds_du, ds_dv depend on the 4 DT corners that are already loaded here.
    # Computing in fwd eliminates the 4 random DT gathers + bilinear-grad
    # compute in bwd, at a cost of 2 extra (B, N) stores in fwd.
    ds_du_val = (d01 - d00) * (1.0 - dv) + (d11 - d10) * dv
    ds_dv_val = (d10 - d00) * (1.0 - du) + (d11 - d01) * du
    ds_du_val = tl.where(inside, ds_du_val, 0.0)
    ds_dv_val = tl.where(inside, ds_dv_val, 0.0)

    # ---- Store outputs ---------------------------------------------------
    out_off = pid_b * s_out_b + n_offs
    # Defensive: ensure residual is finite even though the math above should
    # already guarantee it (sampled = 0 for !inside via tl.where).
    sampled = tl.where(sampled == sampled, sampled, 0.0)  # NaN ⇒ 0
    tl.store(OUT_ptr + out_off, sampled, mask=n_mask)
    tl.store(MASK_ptr + out_off, inside.to(tl.uint8), mask=n_mask)
    # Intermediates for backward — store SAFE values. For !inside points the
    # raw xc/yc/zc/u/v can be ±Inf or NaN (e.g. point behind camera ⇒ zc ≤ 0
    # ⇒ inv_z = ±Inf ⇒ u, v = ±Inf/NaN). The backward must never see those:
    # ``0 * NaN = NaN`` in IEEE-754, so even a zeroed ``grad_residuals`` would
    # poison the chain. Substitute neutral finite values for !inside points.
    xc_s = tl.where(inside, xc, 0.0)
    yc_s = tl.where(inside, yc, 0.0)
    zc_s = tl.where(inside, zc, 1.0)  # 1.0 ⇒ inv_z = 1.0, finite in bwd
    tl.store(XC_ptr + out_off, xc_s, mask=n_mask)
    tl.store(YC_ptr + out_off, yc_s, mask=n_mask)
    tl.store(ZC_ptr + out_off, zc_s, mask=n_mask)
    tl.store(DSDU_ptr + out_off, ds_du_val, mask=n_mask)
    tl.store(DSDV_ptr + out_off, ds_dv_val, mask=n_mask)


# ---------------------------------------------------------------------------
# K-grad reduction kernel — loops over N in scalar accumulators
# ---------------------------------------------------------------------------
#
# Replaces 9 PyTorch `(grad_* * X_*).sum(dim=1)` calls + 4 stacks.
# Grid is 1D `(B,)` — one program per batch. Each program loops over N in
# BLOCK_N tiles, doing `tl.sum(axis=0)` per tile and accumulating into 9
# scalar registers. The single-program-per-batch loop preserves a sequential
# reduction order over N, which is closer to PyTorch's monolithic sum order
# than a 2-level per-tile + cross-tile reduction would be (avoiding the
# FP-order drift that broke an earlier per-tile variant on flowers).


@triton.jit
def _bwd_reduce_kernel(
    XC_ptr,
    YC_ptr,
    ZC_ptr,
    GU_ptr,
    GV_ptr,
    XYZW_ptr,  # (B, N, 3) world points — needed for grad_R
    K_ptr,
    GK_ptr,  # (B, 9) flat K-grad
    GR_ptr,  # (B, 9) flat grad_R
    GT_ptr,  # (B, 3) grad_t
    B,
    N,
    s_xc_b,
    s_xyzw_b,
    s_xyzw_n,
    s_K_b,
    s_gk_b,
    s_gr_b,
    s_gt_b,
    BLOCK_N: tl.constexpr,
):
    """Fused K-grad + grad_R + grad_t reduction.

    One program per batch row; loops over N, accumulating 9 K-grad + 9 grad_R
    + 3 grad_t scalars in registers. Recomputes the projection chain
    (grad_xc, grad_yc, grad_zc) from saved (grad_u, grad_v, xc, yc, zc, K)
    so the (B, N, 3) grad_xyz_cam intermediate is never materialised.
    """
    pid_b = tl.program_id(0)
    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)
    cx = tl.load(Kb + 2)
    fy = tl.load(Kb + 4)
    cy = tl.load(Kb + 5)

    # K-grad accumulators
    aK00 = 0.0
    aK01 = 0.0
    aK02 = 0.0
    aK10 = 0.0
    aK11 = 0.0
    aK12 = 0.0
    aK20 = 0.0
    aK21 = 0.0
    aK22 = 0.0
    # grad_R accumulators (R[i, j] = sum_n grad_xc[i] * xyz_world[j])
    aR00 = 0.0
    aR01 = 0.0
    aR02 = 0.0
    aR10 = 0.0
    aR11 = 0.0
    aR12 = 0.0
    aR20 = 0.0
    aR21 = 0.0
    aR22 = 0.0
    # grad_t accumulators
    aT0 = 0.0
    aT1 = 0.0
    aT2 = 0.0

    for block_start in range(0, N, BLOCK_N):
        n_offs = block_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        row = pid_b * s_xc_b + n_offs
        xc = tl.load(XC_ptr + row, mask=n_mask, other=0.0)
        yc = tl.load(YC_ptr + row, mask=n_mask, other=0.0)
        zc = tl.load(ZC_ptr + row, mask=n_mask, other=1.0)
        gu = tl.load(GU_ptr + row, mask=n_mask, other=0.0)
        gv = tl.load(GV_ptr + row, mask=n_mask, other=0.0)

        # Load xyz_world (B, N, 3) for grad_R reduction.
        xyzw_row = pid_b * s_xyzw_b + n_offs * s_xyzw_n
        x_w = tl.load(XYZW_ptr + xyzw_row + 0, mask=n_mask, other=0.0)
        y_w = tl.load(XYZW_ptr + xyzw_row + 1, mask=n_mask, other=0.0)
        z_w = tl.load(XYZW_ptr + xyzw_row + 2, mask=n_mask, other=0.0)

        inv_z = 1.0 / zc
        X0 = xc * inv_z
        X1 = yc * inv_z
        X2 = zc * inv_z
        u = fx * xc * inv_z + cx
        v = fy * yc * inv_z + cy
        guv = -(gu * u + gv * v)

        aK00 += tl.sum(gu * X0, axis=0)
        aK01 += tl.sum(gu * X1, axis=0)
        aK02 += tl.sum(gu * X2, axis=0)
        aK10 += tl.sum(gv * X0, axis=0)
        aK11 += tl.sum(gv * X1, axis=0)
        aK12 += tl.sum(gv * X2, axis=0)
        aK20 += tl.sum(guv * X0, axis=0)
        aK21 += tl.sum(guv * X1, axis=0)
        aK22 += tl.sum(guv * X2, axis=0)

        # Recompute projection chain so we don't need to write grad_xyz_cam.
        gxc = gu * fx * inv_z
        gyc = gv * fy * inv_z
        gzc = -(gu * fx * xc + gv * fy * yc) * inv_z * inv_z

        aR00 += tl.sum(gxc * x_w, axis=0)
        aR01 += tl.sum(gxc * y_w, axis=0)
        aR02 += tl.sum(gxc * z_w, axis=0)
        aR10 += tl.sum(gyc * x_w, axis=0)
        aR11 += tl.sum(gyc * y_w, axis=0)
        aR12 += tl.sum(gyc * z_w, axis=0)
        aR20 += tl.sum(gzc * x_w, axis=0)
        aR21 += tl.sum(gzc * y_w, axis=0)
        aR22 += tl.sum(gzc * z_w, axis=0)

        aT0 += tl.sum(gxc, axis=0)
        aT1 += tl.sum(gyc, axis=0)
        aT2 += tl.sum(gzc, axis=0)

    base_k = pid_b * s_gk_b
    tl.store(GK_ptr + base_k + 0, aK00)
    tl.store(GK_ptr + base_k + 1, aK01)
    tl.store(GK_ptr + base_k + 2, aK02)
    tl.store(GK_ptr + base_k + 3, aK10)
    tl.store(GK_ptr + base_k + 4, aK11)
    tl.store(GK_ptr + base_k + 5, aK12)
    tl.store(GK_ptr + base_k + 6, aK20)
    tl.store(GK_ptr + base_k + 7, aK21)
    tl.store(GK_ptr + base_k + 8, aK22)
    base_r = pid_b * s_gr_b
    tl.store(GR_ptr + base_r + 0, aR00)
    tl.store(GR_ptr + base_r + 1, aR01)
    tl.store(GR_ptr + base_r + 2, aR02)
    tl.store(GR_ptr + base_r + 3, aR10)
    tl.store(GR_ptr + base_r + 4, aR11)
    tl.store(GR_ptr + base_r + 5, aR12)
    tl.store(GR_ptr + base_r + 6, aR20)
    tl.store(GR_ptr + base_r + 7, aR21)
    tl.store(GR_ptr + base_r + 8, aR22)
    base_t = pid_b * s_gt_b
    tl.store(GT_ptr + base_t + 0, aT0)
    tl.store(GT_ptr + base_t + 1, aT1)
    tl.store(GT_ptr + base_t + 2, aT2)


# ---------------------------------------------------------------------------
# Triton backward kernel (per-point, no atomics)
# ---------------------------------------------------------------------------


@triton.jit
def _project_sample_bwd_kernel(
    # Saved from fwd / inputs — u, v dropped (not used in bwd kernel post-H6;
    # the fused reduce kernel recomputes u, v internally from xc/yc/zc + K).
    XC_ptr,
    YC_ptr,
    ZC_ptr,
    DSDU_ptr,
    DSDV_ptr,
    MASK_ptr,
    GR_ptr,  # grad_residuals (B, N)
    K_ptr,  # (B, 3, 3)
    P_ptr,  # (B, 4, 4)
    # Per-point outputs
    GXYZ_ptr,  # grad_xyz_world (B, N, 3)
    GU_ptr,  # grad_u (B, N) — for grad_K in PyTorch
    GV_ptr,  # grad_v (B, N)
    # Sizes
    B,
    N,
    # Strides
    s_xc_b,  # stride between batches in (B, N) scalar tensors
    s_gxyz_b,
    s_gxyz_n,
    s_K_b,
    s_P_b,
    BLOCK_N: tl.constexpr,
):
    """Backward: per-point gradient of the project+bilinear-sample fwd kernel.

    All computation is per-point (no scatter / atomics). Reductions over N
    (for grad_R, grad_t, grad_K) are done outside this kernel via bmm + sum.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Load per-point intermediates ----------------------------------
    row = pid_b * s_xc_b + n_offs
    xc = tl.load(XC_ptr + row, mask=n_mask, other=0.0)
    yc = tl.load(YC_ptr + row, mask=n_mask, other=0.0)
    zc = tl.load(ZC_ptr + row, mask=n_mask, other=1.0)
    inside_u8 = tl.load(MASK_ptr + row, mask=n_mask, other=0)
    inside = inside_u8 != 0
    gr = tl.load(GR_ptr + row, mask=n_mask, other=0.0)
    # Zero-out grad for outside points up-front; they contributed 0 to the loss.
    gr = tl.where(inside, gr, 0.0)

    # ---- Bilinear gradient: ds/du, ds/dv loaded directly from fwd ------
    # Fwd computed these from the DT corners it already had loaded; loading
    # them here is one coalesced read instead of 4 random DT gathers + the
    # 8-op bilinear-grad compute.
    ds_du = tl.load(DSDU_ptr + row, mask=n_mask, other=0.0)
    ds_dv = tl.load(DSDV_ptr + row, mask=n_mask, other=0.0)
    grad_u = gr * ds_du
    grad_v = gr * ds_dv

    # ---- Chain through projection: u = fx*xc/zc + cx; v = fy*yc/zc + cy
    # zc was stored as 1.0 for !inside points (see fwd kernel), so inv_z is
    # finite even when a point went behind the camera in the upstream graph.
    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)
    fy = tl.load(Kb + 4)
    inv_z = 1.0 / zc
    grad_xc = grad_u * fx * inv_z
    grad_yc = grad_v * fy * inv_z
    grad_zc = -(grad_u * fx * xc + grad_v * fy * yc) * inv_z * inv_z

    # ---- Chain through xyz_cam = R @ xyz + t: grad_xyz = R^T @ grad_xyz_cam
    Pb = P_ptr + pid_b * s_P_b
    R00 = tl.load(Pb + 0)
    R01 = tl.load(Pb + 1)
    R02 = tl.load(Pb + 2)
    R10 = tl.load(Pb + 4)
    R11 = tl.load(Pb + 5)
    R12 = tl.load(Pb + 6)
    R20 = tl.load(Pb + 8)
    R21 = tl.load(Pb + 9)
    R22 = tl.load(Pb + 10)
    # grad_xyz_world[k] = sum_i R[i,k] * grad_xyz_cam[i]
    g_x = R00 * grad_xc + R10 * grad_yc + R20 * grad_zc
    g_y = R01 * grad_xc + R11 * grad_yc + R21 * grad_zc
    g_z = R02 * grad_xc + R12 * grad_yc + R22 * grad_zc

    # ---- Final defensive mask -------------------------------------------
    # Every value below is mathematically zero for !inside points (gr=0 and
    # safe intermediates), but a stray NaN would still poison the optimizer.
    # Explicitly clamp to 0.0 for !inside ⇒ guaranteed-finite gradients.
    g_x = tl.where(inside, g_x, 0.0)
    g_y = tl.where(inside, g_y, 0.0)
    g_z = tl.where(inside, g_z, 0.0)
    grad_xc = tl.where(inside, grad_xc, 0.0)
    grad_yc = tl.where(inside, grad_yc, 0.0)
    grad_zc = tl.where(inside, grad_zc, 0.0)
    grad_u = tl.where(inside, grad_u, 0.0)
    grad_v = tl.where(inside, grad_v, 0.0)

    # ---- Store outputs --------------------------------------------------
    gxyz_row = pid_b * s_gxyz_b + n_offs * s_gxyz_n
    tl.store(GXYZ_ptr + gxyz_row + 0, g_x, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 1, g_y, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 2, g_z, mask=n_mask)
    # grad_xyz_cam is NOT stored — _bwd_reduce_kernel recomputes the
    # projection chain in-tile so the (B, N, 3) intermediate is dead memory.
    # grad_u, grad_v: (B, N) — still needed for the K-grad reduce kernel.
    tl.store(GU_ptr + row, grad_u, mask=n_mask)
    tl.store(GV_ptr + row, grad_v, mask=n_mask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class _ProjectAndSampleTriton(torch.autograd.Function):
    """Fused project+sample with analytical gradients.

    Forward (Triton): residuals, inside_mask.
    Backward (PyTorch, gather-only): grad wrt xyz_world, K, P. dt_fields is
    assumed to not require gradients.
    """

    @staticmethod
    def forward(ctx, xyz_world, K, P, dt_fields_src, dt_indices, img_hw):
        """Args
        xyz_world: (B, N, 3)
        K: (B, 3, 3), P: (B, 4, 4)
        dt_fields_src: (N_img, 1, H, W) or (N_img, H, W) — the *source*
            DT tensor (not gathered per batch).
        dt_indices: (B,) int64 — for each batch row, the image index to
            read from in ``dt_fields_src``.
        img_hw: (B, 2) — per-row real (H, W) of the target image, used
            to gate the inside-mask against the unpadded image extent
            rather than the padded DT canvas. Accepts any numeric dtype;
            cast to int32 internally.
        """
        assert xyz_world.is_cuda and K.is_cuda and P.is_cuda
        assert dt_fields_src.is_cuda and dt_indices.is_cuda
        assert img_hw.is_cuda
        if dt_fields_src.dim() == 4:
            dt_fields_src = dt_fields_src.squeeze(1)
        assert dt_fields_src.dim() == 3, (
            f"dt_fields_src must be (N_img, H, W) or (N_img, 1, H, W), "
            f"got {dt_fields_src.shape}"
        )

        xyz_c = xyz_world.contiguous()
        K_c = K.contiguous()
        P_c = P.contiguous()
        dt_c = dt_fields_src.contiguous()
        idx_c = dt_indices.contiguous().to(torch.int64)

        B, N, _ = xyz_c.shape
        _, H, W = dt_c.shape
        assert idx_c.shape == (B,), f"dt_indices shape ({B},), got {idx_c.shape}"
        assert img_hw.shape == (
            B,
            2,
        ), f"img_hw must be shape ({B}, 2), got {tuple(img_hw.shape)}"
        hw_c = img_hw.contiguous().to(torch.int32)
        device = xyz_c.device
        dtype = xyz_c.dtype

        residuals = torch.empty((B, N), device=device, dtype=dtype)
        mask = torch.empty((B, N), device=device, dtype=torch.uint8)
        xc = torch.empty((B, N), device=device, dtype=dtype)
        yc = torch.empty((B, N), device=device, dtype=dtype)
        zc = torch.empty((B, N), device=device, dtype=dtype)
        ds_du = torch.empty((B, N), device=device, dtype=dtype)
        ds_dv = torch.empty((B, N), device=device, dtype=dtype)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        # Triton launches against the current CUDA device, not the tensors'
        # device — pin it so multi-GPU callers (e.g. ``device="cuda:4"``) work.
        with torch.cuda.device(device):
            _project_sample_fwd_kernel[grid](
                xyz_c,
                K_c,
                P_c,
                dt_c,
                idx_c,
                hw_c,
                residuals,
                mask,
                xc,
                yc,
                zc,
                ds_du,
                ds_dv,
                B,
                N,
                H,
                W,
                xyz_c.stride(0),
                xyz_c.stride(1),
                K_c.stride(0),
                P_c.stride(0),
                dt_c.stride(0),
                dt_c.stride(1),
                hw_c.stride(0),
                residuals.stride(0),
                BLOCK_N=BLOCK_N,
            )

        # Save references for backward. u, v are NOT saved — the fused
        # reduce kernel recomputes them in-kernel from xc/yc/zc + K, and
        # the bwd kernel no longer touches u, v post-H6.
        ctx.save_for_backward(xyz_c, K_c, P_c, xc, yc, zc, ds_du, ds_dv, mask)
        return residuals, mask.bool()

    @staticmethod
    def backward(ctx, grad_residuals, grad_mask_unused):
        """Analytical backward for the fused project+sample op.

        Returns gradients w.r.t. ``(xyz_world, K, P)``; the DT field, dt
        indices, and ``img_hw`` are non-grad inputs and yield ``None``.
        """
        xyz_world, K, P, xc, yc, zc, ds_du, ds_dv, mask = ctx.saved_tensors
        B, N, _ = xyz_world.shape
        device, dtype = xyz_world.device, xyz_world.dtype

        gr = grad_residuals.contiguous()

        # Per-point outputs from the bwd kernel: grad_xyz_world, grad_u,
        # grad_v. grad_xyz_cam is NO LONGER materialised — the fused reduce
        # kernel below recomputes the projection chain in-tile.
        grad_xyz_world = torch.empty_like(xyz_world)
        grad_u = torch.empty((B, N), device=device, dtype=dtype)
        grad_v = torch.empty((B, N), device=device, dtype=dtype)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        with torch.cuda.device(device):
            _project_sample_bwd_kernel[grid](
                xc,
                yc,
                zc,
                ds_du,
                ds_dv,
                mask,
                gr,
                K,
                P,
                grad_xyz_world,
                grad_u,
                grad_v,
                B,
                N,
                xc.stride(0),
                grad_xyz_world.stride(0),
                grad_xyz_world.stride(1),
                K.stride(0),
                P.stride(0),
                BLOCK_N=BLOCK_N,
            )

        # ---- Fused K-grad + grad_R + grad_t reduction in one kernel ----
        # Replaces: bmm + sum for grad_R/grad_t and 9 sums for grad_K. The
        # kernel loops over N per batch, accumulating 9 + 9 + 3 scalars in
        # registers. No (B, N, 3) grad_xyz_cam intermediate needed.
        grad_K_flat = torch.empty((B, 9), device=device, dtype=dtype)
        grad_R_flat = torch.empty((B, 9), device=device, dtype=dtype)
        grad_t = torch.empty((B, 3), device=device, dtype=dtype)
        with torch.cuda.device(device):
            _bwd_reduce_kernel[(B,)](
                xc,
                yc,
                zc,
                grad_u,
                grad_v,
                xyz_world,
                K,
                grad_K_flat,
                grad_R_flat,
                grad_t,
                B,
                N,
                xc.stride(0),
                xyz_world.stride(0),
                xyz_world.stride(1),
                K.stride(0),
                grad_K_flat.stride(0),
                grad_R_flat.stride(0),
                grad_t.stride(0),
                BLOCK_N=1024,
            )
        grad_K = grad_K_flat.view(B, 3, 3)
        grad_R = grad_R_flat.view(B, 3, 3)
        grad_P = torch.zeros_like(P)
        grad_P[:, :3, :3] = grad_R
        grad_P[:, :3, 3] = grad_t

        # dt_fields_src, dt_indices, img_hw do not require gradients.
        return grad_xyz_world, grad_K, grad_P, None, None, None


def project_and_sample_triton(
    xyz_world: torch.Tensor,
    K1: torch.Tensor,
    P1: torch.Tensor,
    dt_fields_src: torch.Tensor,
    dt_indices: torch.Tensor,
    img_hw: torch.Tensor,
):
    """Fused project + bilinear DT sample (no per-batch DT gather).

    Args:
        xyz_world: ``(B, N, 3)`` 3D points in world coords.
        K1: ``(B, 3, 3)`` intrinsics.
        P1: ``(B, 4, 4)`` world-to-camera extrinsics.
        dt_fields_src: ``(N_img, 1, H, W)`` or ``(N_img, H, W)`` *source*
            distance fields — the underlying ``self.dt_fields.params`` tensor,
            not a per-batch copy.
        dt_indices: ``(B,)`` int64 — for each batch row, the image index to
            sample from in ``dt_fields_src``. Avoids materialising the
            ``(B, H, W)`` gather PyTorch would otherwise produce.
        img_hw: ``(B, 2)`` — per-row real ``(H, W)`` of the target image.
            Gates the inside-mask against the unpadded image extent so
            projections that land in the padded zone of the DT canvas are
            correctly rejected on mixed-resolution datasets.

    Returns:
        ``(residuals, inside_mask)`` of shapes ``(B, N)`` and ``(B, N)`` bool.
        Outside points get residual ``0.0``; the mask is ``True`` for inside.
    """
    return _ProjectAndSampleTriton.apply(
        xyz_world, K1, P1, dt_fields_src, dt_indices, img_hw
    )


# ===========================================================================
# Fused unproject kernel: (xy, K, depth, P)  →  xyz_world
# ===========================================================================
#
# Reference math (per point, fixed pixel coords ``(u, v)``, learnable
# ``K, depth, P``):
#
#   xyz_cam = depth · K_inv · [u, v, 1]
#           = ( depth · (u - cx) / fx,
#               depth · (v - cy) / fy,
#               depth )
#
#   xyz_world = R^T · (xyz_cam - t)      where (R, t) = (P[:3,:3], P[:3,3])
#
# ``K_inv`` is the closed-form pinhole inverse; the reference's
# :func:`invert_K` only touches ``fx, fy, cx, cy`` so autograd only emits
# gradients for those four entries — we match exactly. ``invert_P`` uses
# ``R_inv = R^T`` and ``t_inv = -R^T t`` so the gradient flows to ``R, t``
# of the original ``P``.
#
# Tensor shapes (B = N_images, N = max_edges per image):
#   xy0    (B, N, 2)   pixel coords — *no grad*
#   depth  (B, N)      depth values — grad through depth-scale/shift
#   K      (B, 3, 3)   intrinsics   — grad on (0,0), (1,1), (0,2), (1,2)
#   P      (B, 4, 4)   extrinsics   — grad on the top-left 3×4 block
#   out    (B, N, 3)   xyz_world


@triton.jit
def _unproject_fwd_kernel(
    XY_ptr,  # (B, N, 2) — pixel coords
    DEPTH_ptr,  # (B, N)    — corrected depth (already a·z+b·)
    K_ptr,  # (B, 3, 3) — intrinsics (only fx, fy, cx, cy read)
    P_ptr,  # (B, 4, 4) — extrinsics
    XYZW_ptr,  # (B, N, 3) — output xyz_world
    XYZC_ptr,  # (B, N, 3) — saved xyz_cam (used by backward reductions)
    B,
    N,
    s_xy_b,
    s_xy_n,
    s_d_b,
    s_K_b,
    s_P_b,
    s_xyzw_b,
    s_xyzw_n,
    BLOCK_N: tl.constexpr,
):
    """Pixel → world: collapses the four-op chain into one kernel.

    Grid layout ``(B, ceil(N / BLOCK_N))`` — one program per image and tile.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Load xy0 (u, v) ------------------------------------------------
    xy_row = pid_b * s_xy_b + n_offs * s_xy_n
    u = tl.load(XY_ptr + xy_row + 0, mask=n_mask, other=0.0)
    v = tl.load(XY_ptr + xy_row + 1, mask=n_mask, other=0.0)

    # ---- Load depth -----------------------------------------------------
    z = tl.load(DEPTH_ptr + pid_b * s_d_b + n_offs, mask=n_mask, other=0.0)

    # ---- Load K (only the 4 entries pinhole uses) -----------------------
    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)  # K[0,0]
    cx = tl.load(Kb + 2)  # K[0,2]
    fy = tl.load(Kb + 4)  # K[1,1]
    cy = tl.load(Kb + 5)  # K[1,2]

    # ---- Load P (R 3×3 + t) --------------------------------------------
    Pb = P_ptr + pid_b * s_P_b
    R00 = tl.load(Pb + 0)
    R01 = tl.load(Pb + 1)
    R02 = tl.load(Pb + 2)
    t0 = tl.load(Pb + 3)
    R10 = tl.load(Pb + 4)
    R11 = tl.load(Pb + 5)
    R12 = tl.load(Pb + 6)
    t1 = tl.load(Pb + 7)
    R20 = tl.load(Pb + 8)
    R21 = tl.load(Pb + 9)
    R22 = tl.load(Pb + 10)
    t2 = tl.load(Pb + 11)

    # ---- xyz_cam = depth · K_inv · [u, v, 1] ---------------------------
    # K_inv = [[1/fx, 0, -cx/fx],
    #          [0, 1/fy, -cy/fy],
    #          [0,    0,    1   ]]
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    xc = z * (u - cx) * inv_fx
    yc = z * (v - cy) * inv_fy
    zc = z

    # ---- xyz_world = R^T · (xyz_cam - t) -------------------------------
    Yx = xc - t0
    Yy = yc - t1
    Yz = zc - t2
    # R^T row-i, col-k uses R[k, i]; here we accumulate per output component.
    xw = R00 * Yx + R10 * Yy + R20 * Yz
    yw = R01 * Yx + R11 * Yy + R21 * Yz
    zw = R02 * Yx + R12 * Yy + R22 * Yz

    # ---- Store outputs --------------------------------------------------
    out_off = pid_b * s_xyzw_b + n_offs * s_xyzw_n
    tl.store(XYZW_ptr + out_off + 0, xw, mask=n_mask)
    tl.store(XYZW_ptr + out_off + 1, yw, mask=n_mask)
    tl.store(XYZW_ptr + out_off + 2, zw, mask=n_mask)
    # Save xyz_cam for the small bmm/sum reductions performed in PyTorch.
    tl.store(XYZC_ptr + out_off + 0, xc, mask=n_mask)
    tl.store(XYZC_ptr + out_off + 1, yc, mask=n_mask)
    tl.store(XYZC_ptr + out_off + 2, zc, mask=n_mask)


@triton.jit
def _unproject_bwd_kernel(
    XY_ptr,  # (B, N, 2)
    DEPTH_ptr,  # (B, N)
    K_ptr,  # (B, 3, 3)
    P_ptr,  # (B, 4, 4) — only R is used here
    GXYZW_ptr,  # (B, N, 3) — upstream gradient
    GXYZC_ptr,  # (B, N, 3) — output: grad_xyz_cam (used for grad_K in PyTorch)
    GDEPTH_ptr,  # (B, N)    — output: grad_depth (per-point)
    B,
    N,
    s_xy_b,
    s_xy_n,
    s_d_b,
    s_K_b,
    s_P_b,
    s_gxyzw_b,
    s_gxyzw_n,
    BLOCK_N: tl.constexpr,
):
    """Per-point backward: emits grad_xyz_cam and grad_depth.

    grad_R, grad_t and grad_K are produced by small PyTorch reductions
    outside the kernel using the saved ``xyz_cam`` from the forward.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Re-load static inputs -----------------------------------------
    xy_row = pid_b * s_xy_b + n_offs * s_xy_n
    u = tl.load(XY_ptr + xy_row + 0, mask=n_mask, other=0.0)
    v = tl.load(XY_ptr + xy_row + 1, mask=n_mask, other=0.0)

    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)
    cx = tl.load(Kb + 2)
    fy = tl.load(Kb + 4)
    cy = tl.load(Kb + 5)
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    Pb = P_ptr + pid_b * s_P_b
    R00 = tl.load(Pb + 0)
    R01 = tl.load(Pb + 1)
    R02 = tl.load(Pb + 2)
    R10 = tl.load(Pb + 4)
    R11 = tl.load(Pb + 5)
    R12 = tl.load(Pb + 6)
    R20 = tl.load(Pb + 8)
    R21 = tl.load(Pb + 9)
    R22 = tl.load(Pb + 10)

    # ---- Upstream gradient ---------------------------------------------
    g_row = pid_b * s_gxyzw_b + n_offs * s_gxyzw_n
    gxw = tl.load(GXYZW_ptr + g_row + 0, mask=n_mask, other=0.0)
    gyw = tl.load(GXYZW_ptr + g_row + 1, mask=n_mask, other=0.0)
    gzw = tl.load(GXYZW_ptr + g_row + 2, mask=n_mask, other=0.0)

    # ---- grad through xyz_world = R^T · Y -------------------------------
    # ∂xyz_world[k] / ∂Y[j] = R[j, k]
    # grad_Y[j] = Σ_k R[j, k] · grad_xyz_world[k]
    gYx = R00 * gxw + R01 * gyw + R02 * gzw
    gYy = R10 * gxw + R11 * gyw + R12 * gzw
    gYz = R20 * gxw + R21 * gyw + R22 * gzw

    # ---- grad through Y = xyz_cam - t -----------------------------------
    # Per-point: grad_xyz_cam = grad_Y. (grad_t is a per-batch reduction
    # done in PyTorch.)
    grad_xc = gYx
    grad_yc = gYy
    grad_zc = gYz

    # ---- grad through xyz_cam = depth · K_inv · [u, v, 1] ---------------
    # ∂xc_cam/∂z = (u - cx) / fx   ;  ∂yc_cam/∂z = (v - cy) / fy
    # ∂zc_cam/∂z = 1
    grad_z = grad_xc * (u - cx) * inv_fx + grad_yc * (v - cy) * inv_fy + grad_zc

    # ---- Store outputs --------------------------------------------------
    tl.store(GXYZC_ptr + g_row + 0, grad_xc, mask=n_mask)
    tl.store(GXYZC_ptr + g_row + 1, grad_yc, mask=n_mask)
    tl.store(GXYZC_ptr + g_row + 2, grad_zc, mask=n_mask)
    tl.store(GDEPTH_ptr + pid_b * s_d_b + n_offs, grad_z, mask=n_mask)


class _UnprojectTriton(torch.autograd.Function):
    """Fused unproject with analytical gradients.

    Forward: ``xyz_world`` from ``(xy0, depth, K, P)``.
    Backward: ``grad_depth, grad_K, grad_P`` (``xy0`` has no grad).
    """

    @staticmethod
    def forward(ctx, xy0, depth, K, P):
        """Forward: lift pixel coords + depth to world via ``K`` and ``P``.

        Args:
            ctx: autograd context for saving tensors for backward.
            xy0: ``(B, N, 2)`` pixel coordinates (integer-center convention).
            depth: ``(B, N)`` per-pixel depth.
            K: ``(B, 3, 3)`` intrinsics.
            P: ``(B, 4, 4)`` world-to-cam extrinsics.

        Returns:
            ``(B, N, 3)`` world-space points.
        """
        assert xy0.is_cuda and depth.is_cuda and K.is_cuda and P.is_cuda
        assert xy0.dim() == 3 and xy0.shape[-1] == 2, f"bad xy0 shape: {xy0.shape}"
        assert depth.dim() == 2, f"depth must be (B, N), got {depth.shape}"

        xy_c = xy0.contiguous()
        d_c = depth.contiguous()
        K_c = K.contiguous()
        P_c = P.contiguous()

        B, N, _ = xy_c.shape
        device, dtype = xy_c.device, xy_c.dtype

        xyz_world = torch.empty((B, N, 3), device=device, dtype=dtype)
        xyz_cam = torch.empty((B, N, 3), device=device, dtype=dtype)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        with torch.cuda.device(device):
            _unproject_fwd_kernel[grid](
                xy_c,
                d_c,
                K_c,
                P_c,
                xyz_world,
                xyz_cam,
                B,
                N,
                xy_c.stride(0),
                xy_c.stride(1),
                d_c.stride(0),
                K_c.stride(0),
                P_c.stride(0),
                xyz_world.stride(0),
                xyz_world.stride(1),
                BLOCK_N=BLOCK_N,
            )

        ctx.save_for_backward(xy_c, d_c, K_c, P_c, xyz_cam)
        return xyz_world

    @staticmethod
    def backward(ctx, grad_xyz_world):
        """Analytical backward for the fused unproject op.

        Returns gradients w.r.t. ``(depth, K, P)``; ``xy0`` is a non-grad
        input and yields ``None``.
        """
        xy0, depth, K, P, xyz_cam = ctx.saved_tensors
        B, N, _ = xy0.shape

        gr = grad_xyz_world.contiguous()

        grad_xyz_cam = torch.empty_like(xyz_cam)
        grad_depth = torch.empty_like(depth)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        with torch.cuda.device(xy0.device):
            _unproject_bwd_kernel[grid](
                xy0,
                depth,
                K,
                P,
                gr,
                grad_xyz_cam,
                grad_depth,
                B,
                N,
                xy0.stride(0),
                xy0.stride(1),
                depth.stride(0),
                K.stride(0),
                P.stride(0),
                gr.stride(0),
                gr.stride(1),
                BLOCK_N=BLOCK_N,
            )

        # ---- Reductions over N (small, well-optimised PyTorch bmm/sum) --
        R = P[:, :3, :3]
        t = P[:, :3, 3]
        # Y = xyz_cam - t (broadcast t over N)
        Y = xyz_cam - t.unsqueeze(1)  # (B, N, 3)

        # grad_R[b, i, k] = Σ_n Y[b, n, i] · grad_xyz_world[b, n, k]
        grad_R = torch.bmm(Y.transpose(-1, -2), gr)  # (B, 3, 3)

        # grad_t[b, i] = -Σ_n Σ_k R[b, i, k] · grad_xyz_world[b, n, k]
        total_gxyzw = gr.sum(dim=1)  # (B, 3)
        grad_t = -torch.bmm(R, total_gxyzw.unsqueeze(-1)).squeeze(-1)

        grad_P = torch.zeros_like(P)
        grad_P[:, :3, :3] = grad_R
        grad_P[:, :3, 3] = grad_t

        # ---- grad_K: only fx, fy, cx, cy (autograd through invert_K
        #              would emit zero for off-diagonals; we match it). ----
        # xc_cam = z · (u - cx) / fx  ⇒
        #   ∂xc_cam/∂fx = -xc_cam / fx     ;  ∂xc_cam/∂cx = -z / fx
        # similar for yc_cam / fy / cy.
        fx_b = K[:, 0, 0]  # (B,)
        fy_b = K[:, 1, 1]
        inv_fx_b = 1.0 / fx_b
        inv_fy_b = 1.0 / fy_b

        gxc = grad_xyz_cam[:, :, 0]
        gyc = grad_xyz_cam[:, :, 1]
        xc = xyz_cam[:, :, 0]
        yc = xyz_cam[:, :, 1]

        grad_fx = -(gxc * xc).sum(dim=1) * inv_fx_b  # (B,)
        grad_fy = -(gyc * yc).sum(dim=1) * inv_fy_b
        grad_cx = -(gxc * depth).sum(dim=1) * inv_fx_b
        grad_cy = -(gyc * depth).sum(dim=1) * inv_fy_b

        grad_K = torch.zeros_like(K)
        grad_K[:, 0, 0] = grad_fx
        grad_K[:, 1, 1] = grad_fy
        grad_K[:, 0, 2] = grad_cx
        grad_K[:, 1, 2] = grad_cy

        # xy0 is fixed (no grad).
        return None, grad_depth, grad_K, grad_P


def unproject_2D_to_world_triton(
    xy0: torch.Tensor,
    K0: torch.Tensor,
    depth0: torch.Tensor,
    P0: torch.Tensor,
) -> torch.Tensor:
    """Fused pixel → world projection (Triton).

    Drop-in replacement for :func:`helpers.reprojection.unproject_2D_to_world`.

    Args:
        xy0: ``(B, N, 2)`` pixel coordinates (no grad).
        K0: ``(B, 3, 3)`` intrinsics (only the pinhole entries are read).
        depth0: ``(B, N)`` per-pixel depth.
        P0: ``(B, 4, 4)`` world-to-camera extrinsics.

    Returns:
        ``(B, N, 3)`` 3D points in world coordinates.
    """
    return _UnprojectTriton.apply(xy0, depth0, K0, P0)


# ---------------------------------------------------------------------------
# Exact L2 Euclidean Distance Transform (Felzenszwalb-Huttenlocher 2004)
# ---------------------------------------------------------------------------
#
# Replaces ``cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)``.
# OpenCV's precise path implements the same algorithm; running it as a Triton
# kernel keeps the DT field on-GPU (no cpu round-trip) and parallelises the
# per-row 1D pass over CUDA programs.
#
# 2D EDT is the separable composition of two 1D passes:
#   pass 1: 1D EDT along each row     → squared horizontal dist to nearest edge
#   pass 2: 1D EDT along each column  → final squared 2D dist
# The 1D EDT itself is the FH lower-envelope-of-parabolas algorithm — O(N) per
# row, sequential within a row but trivially parallel across rows.
#
# Non-edge pixels are seeded with ``LARGE = H*H + W*W`` (a finite upper bound
# on any squared 2D distance in the image), not +Inf — this keeps the parabola
# intersection arithmetic ``(f[q]+q² - f[v]-v²) / (2(q-v))`` strictly finite,
# while still ensuring those parabolas never appear in any lower envelope when
# at least one real edge exists in the row/column.


@triton.jit
def _edt_1d_sq_kernel(
    F_ptr,  # (B, N) float32 — input function values (0 at edges, LARGE elsewhere)
    OUT_ptr,  # (B, N) float32 — output squared distances
    V_ptr,  # (B, N) int32 — per-row workspace: envelope parabola centres
    Z_ptr,  # (B, N+1) float32 — per-row workspace: envelope boundaries
    N,
    s_f_b,
    s_o_b,
    s_v_b,
    s_z_b,
    INF: tl.constexpr,
):
    """One program per row; sequentially runs FH 1D EDT on that row."""
    pid = tl.program_id(0)

    f_row = F_ptr + pid * s_f_b
    o_row = OUT_ptr + pid * s_o_b
    v_row = V_ptr + pid * s_v_b
    z_row = Z_ptr + pid * s_z_b

    # Init lower envelope with parabola at index 0
    tl.store(v_row + 0, 0)
    tl.store(z_row + 0, -INF)
    tl.store(z_row + 1, INF)
    k = 0

    # Forward sweep: build lower envelope of parabolas rooted at f[q]+q²
    s = 0.0
    for q in range(1, N):
        fq = tl.load(f_row + q)
        q_f = q.to(tl.float32)
        qq = q_f * q_f

        # Pop dominated parabolas. Defensive: also stop if k would go
        # negative, so we never form a pointer outside the workspace row.
        # Triton doesn't support `break`, so use a continuation flag.
        keep_popping = True
        while keep_popping:
            vk = tl.load(v_row + k)
            fvk = tl.load(f_row + vk)
            vk_f = vk.to(tl.float32)
            s = (fq + qq - fvk - vk_f * vk_f) / (2.0 * (q_f - vk_f))
            zk = tl.load(z_row + k)
            if s > zk:
                keep_popping = False
            else:
                if k == 0:
                    keep_popping = False  # would underflow; stop here
                else:
                    k -= 1

        k += 1
        tl.store(v_row + k, q)
        tl.store(z_row + k, s)
        tl.store(z_row + (k + 1), INF)

    # Backward fill: each output pixel q reads from its dominating parabola
    k = 0
    for q in range(N):
        q_f = q.to(tl.float32)
        keep_advancing = True
        while keep_advancing:
            zk1 = tl.load(z_row + (k + 1))
            if zk1 >= q_f:
                keep_advancing = False
            else:
                k += 1

        vk = tl.load(v_row + k)
        fvk = tl.load(f_row + vk)
        vk_f = vk.to(tl.float32)
        dq = q_f - vk_f
        tl.store(o_row + q, dq * dq + fvk)


def _edt_1d_sq_triton(f: torch.Tensor) -> torch.Tensor:
    """Squared 1D EDT along last dim. ``f`` must be (B, N) float32, CUDA, contiguous."""
    B, N = f.shape
    out = torch.empty_like(f)
    v = torch.zeros((B, N), dtype=torch.int32, device=f.device)
    # Pre-fill z with +inf so any defensive read past the envelope tail
    # is well-defined and stops the backward sweep instead of walking off.
    z = torch.full((B, N + 1), float("inf"), dtype=torch.float32, device=f.device)

    with torch.cuda.device(f.device):
        _edt_1d_sq_kernel[(B,)](
            f,
            out,
            v,
            z,
            N,
            f.stride(0),
            out.stride(0),
            v.stride(0),
            z.stride(0),
            INF=float("inf"),
        )
    return out


def distance_transform_l2_triton(edges_map: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean distance transform via Felzenszwalb-Huttenlocher (Triton).

    Drop-in replacement for ``cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)``
    where ``mask`` is built with edge pixels = 0 and background = 1 (i.e. distance
    is measured to the *edge* pixels of ``edges_map``).

    Args:
        edges_map: (H, W) tensor. Values > 0 are treated as edges (distance 0).

    Returns:
        (H, W) float32 tensor of Euclidean distances to the nearest edge, on the
        same device as ``edges_map``.
    """
    assert edges_map.is_cuda, "Triton EDT requires a CUDA tensor"
    H, W = edges_map.shape[-2:]
    device = edges_map.device

    # Seed: 0 at edge pixels, LARGE elsewhere. LARGE is a finite upper bound on
    # the squared 2D distance, so the FH arithmetic stays in-range and parabolas
    # rooted at non-edge pixels are dominated wherever any edge exists.
    LARGE = float(H * H + W * W)
    f = torch.where(
        edges_map > 0,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.tensor(LARGE, dtype=torch.float32, device=device),
    ).contiguous()

    # Pass 1: 1D EDT along rows → squared horizontal distances
    f = _edt_1d_sq_triton(f)

    # Pass 2: 1D EDT along columns. Transpose so columns become the last dim,
    # run the same kernel, transpose back.
    f = _edt_1d_sq_triton(f.t().contiguous()).t().contiguous()

    return torch.sqrt(f)
