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
    # Saved intermediates for backward. xc/yc/zc are NOT saved — the fused
    # bwd kernel (cycle 19) re-derives them from (xyz_world, P) using the
    # same source expression, then uses them for *both* the per-point grad
    # AND the per-tile K/R/t partial sums within one program. The cycle-18
    # cross-kernel FP-order issue cannot arise because there's only one
    # consumer kernel now.
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

    Grid layout: ``(ceil(N / BLOCK_N), B)``. ``pid_n`` is the fast-varying
    axis (axis 0 = CUDA ``blockIdx.x``), so the hardware dispatches
    consecutive blocks with the same ``pid_b`` — i.e. they all sample the
    same DT image. The 1 MB DT field for that image stays hot in L2 across
    the ~ceil(N / BLOCK_N) tiles of a single batch row. The original layout
    ``(B, N / BLOCK_N)`` put ``pid_b`` on the fast-varying axis, so
    consecutive blocks hit different DT images and L2 went cold on the
    dominant memory bottleneck of this kernel.

    DT_ptr is the *source* tensor — same memory for every program; ``DT_IDX_ptr``
    tells this program which image's DT field to sample. This avoids
    materialising a per-batch ``(B, H, W)`` copy, which on 518² fields is a
    ~550 MB gather per mini-batch.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

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
    # xc/yc/zc are NOT stored — see kernel-signature comment. The fused bwd
    # kernel reads xyz_world+P and recomputes them. ds_du/ds_dv ARE stored
    # because recomputing them in bwd would re-do the 4 random DT gathers,
    # which are the dominant memory cost.
    tl.store(DSDU_ptr + out_off, ds_du_val, mask=n_mask)
    tl.store(DSDV_ptr + out_off, ds_dv_val, mask=n_mask)


# ---------------------------------------------------------------------------
# Fused backward + per-tile reduction kernel (cycle 19)
# ---------------------------------------------------------------------------
#
# One kernel does what `_project_sample_bwd_kernel` and `_bwd_reduce_kernel`
# previously did separately. Per program:
#   1. Re-derive xc/yc/zc from xyz_world + P (same source expression as fwd,
#      bit-identical FP since it's compiled in the same kernel context).
#   2. Compute grad_xyz_world per point → write to GMEM.
#   3. Accumulate 9 K-grad + 9 R-grad + 3 t-grad partial sums over the tile
#      via `tl.sum(axis=0)` → write 21 scalars to per-tile partial buffers.
#
# A small `_combine_partials_kernel` (grid (B,)) then sequentially sums the
# partials into final grad_K / grad_R / grad_t, matching the old reduce
# kernel's left-associative outer-loop accumulation order.
#
# Why fused vs the old split: cycle 18 measured +5.94% from dropping the
# xc/yc/zc fwd saves, but the AUC drifted because bwd and reduce each
# recomputed xc/yc/zc separately and Triton scheduled their FMAs slightly
# differently. Doing both in one kernel keeps xc/yc/zc bit-identical
# between the per-point grad and the per-tile reduction; the cross-tile
# combine is FP-order-identical to the old reduce kernel's outer loop.


@triton.jit
def _project_sample_bwd_kernel(
    XYZW_ptr,  # (B, N, 3) world points — used for xc/yc/zc derive + grad_R
    DSDU_ptr,  # (B, N) fwd-saved bilinear gradient
    DSDV_ptr,
    MASK_ptr,  # (B, N) uint8 — inside mask
    GR_ptr,  # (B, N) grad_residuals (upstream cotangent)
    K_ptr,  # (B, 3, 3)
    P_ptr,  # (B, 4, 4)
    # Outputs
    GXYZ_ptr,  # (B, N, 3) grad_xyz_world (per-point)
    PK_ptr,  # (B, n_tiles, 9) per-tile K-grad partials
    PR_ptr,  # (B, n_tiles, 9) per-tile R-grad partials
    PT_ptr,  # (B, n_tiles, 3) per-tile t-grad partials
    # Sizes
    B,
    N,
    # Strides
    s_xyzw_b,
    s_xyzw_n,
    s_dsdu_b,  # stride between batches in (B, N) scalar tensors
    s_gxyz_b,
    s_gxyz_n,
    s_K_b,
    s_P_b,
    s_pk_b,
    s_pk_t,
    s_pr_b,
    s_pr_t,
    s_pt_b,
    s_pt_t,
    BLOCK_N: tl.constexpr,
):
    """Fused per-point bwd + per-tile K/R/t partial reductions.

    Grid layout: ``(ceil(N / BLOCK_N), B)``. ``pid_n`` is fast-varying
    (axis 0 = CUDA ``blockIdx.x``) so consecutive blocks share ``pid_b``
    and reuse the per-batch K/P scalars and per-row (B, N) tensors in L1
    — same locality argument as cycles 8/9. One tile of BLOCK_N points
    per program; one row of (PK, PR, PT) partial sums per program.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Load xyz_world (used for xc/yc/zc derive AND for grad_R sums) -
    xyzw_row = pid_b * s_xyzw_b + n_offs * s_xyzw_n
    x_w = tl.load(XYZW_ptr + xyzw_row + 0, mask=n_mask, other=0.0)
    y_w = tl.load(XYZW_ptr + xyzw_row + 1, mask=n_mask, other=0.0)
    z_w = tl.load(XYZW_ptr + xyzw_row + 2, mask=n_mask, other=0.0)

    # ---- Load P (R + t) once; reused for xc/yc/zc + grad_xyz_world -----
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

    # ---- Recompute xc/yc/zc (same source expression as fwd) ------------
    xc = R00 * x_w + R01 * y_w + R02 * z_w + t0
    yc = R10 * x_w + R11 * y_w + R12 * z_w + t1
    zc = R20 * x_w + R21 * y_w + R22 * z_w + t2

    # ---- Mask + zc !inside-safe substitution ---------------------------
    row = pid_b * s_dsdu_b + n_offs
    inside_u8 = tl.load(MASK_ptr + row, mask=n_mask, other=0)
    inside = inside_u8 != 0
    # xc/yc are inherently finite (linear combo of finite world points);
    # only zc needs the substitution so 1/zc stays finite.
    zc = tl.where(inside, zc, 1.0)

    # ---- Load grad_residuals + ds_du/ds_dv → grad_u/grad_v -------------
    gr = tl.load(GR_ptr + row, mask=n_mask, other=0.0)
    gr = tl.where(inside, gr, 0.0)
    ds_du = tl.load(DSDU_ptr + row, mask=n_mask, other=0.0)
    ds_dv = tl.load(DSDV_ptr + row, mask=n_mask, other=0.0)
    grad_u = gr * ds_du
    grad_v = gr * ds_dv

    # ---- K (fx, fy, cx, cy) + perspective divide -----------------------
    Kb = K_ptr + pid_b * s_K_b
    fx = tl.load(Kb + 0)
    cx = tl.load(Kb + 2)
    fy = tl.load(Kb + 4)
    cy = tl.load(Kb + 5)
    inv_z = 1.0 / zc

    # ---- grad_xc/yc/zc — used for BOTH per-point AND partial sums ------
    grad_xc = grad_u * fx * inv_z
    grad_yc = grad_v * fy * inv_z
    grad_zc = -(grad_u * fx * xc + grad_v * fy * yc) * inv_z * inv_z

    # ---- grad_xyz_world = R^T @ grad_xyz_cam (per-point output) --------
    g_x = R00 * grad_xc + R10 * grad_yc + R20 * grad_zc
    g_y = R01 * grad_xc + R11 * grad_yc + R21 * grad_zc
    g_z = R02 * grad_xc + R12 * grad_yc + R22 * grad_zc

    # Defensive: mathematically zero for !inside (gr=0), but tl.where
    # guards against any stray NaN coming through.
    g_x = tl.where(inside, g_x, 0.0)
    g_y = tl.where(inside, g_y, 0.0)
    g_z = tl.where(inside, g_z, 0.0)

    gxyz_row = pid_b * s_gxyz_b + n_offs * s_gxyz_n
    tl.store(GXYZ_ptr + gxyz_row + 0, g_x, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 1, g_y, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 2, g_z, mask=n_mask)

    # ---- Per-tile partial reductions (same operand sequence as the old
    #      reduce kernel's per-tile sums) --------------------------------
    X0 = xc * inv_z
    X1 = yc * inv_z
    X2 = zc * inv_z
    u = fx * xc * inv_z + cx
    v = fy * yc * inv_z + cy
    guv = -(grad_u * u + grad_v * v)

    pK00 = tl.sum(grad_u * X0, axis=0)
    pK01 = tl.sum(grad_u * X1, axis=0)
    pK02 = tl.sum(grad_u * X2, axis=0)
    pK10 = tl.sum(grad_v * X0, axis=0)
    pK11 = tl.sum(grad_v * X1, axis=0)
    pK12 = tl.sum(grad_v * X2, axis=0)
    pK20 = tl.sum(guv * X0, axis=0)
    pK21 = tl.sum(guv * X1, axis=0)
    pK22 = tl.sum(guv * X2, axis=0)

    pR00 = tl.sum(grad_xc * x_w, axis=0)
    pR01 = tl.sum(grad_xc * y_w, axis=0)
    pR02 = tl.sum(grad_xc * z_w, axis=0)
    pR10 = tl.sum(grad_yc * x_w, axis=0)
    pR11 = tl.sum(grad_yc * y_w, axis=0)
    pR12 = tl.sum(grad_yc * z_w, axis=0)
    pR20 = tl.sum(grad_zc * x_w, axis=0)
    pR21 = tl.sum(grad_zc * y_w, axis=0)
    pR22 = tl.sum(grad_zc * z_w, axis=0)

    pT0 = tl.sum(grad_xc, axis=0)
    pT1 = tl.sum(grad_yc, axis=0)
    pT2 = tl.sum(grad_zc, axis=0)

    # ---- Store the 21 partials at slot (pid_b, pid_n, k) ---------------
    pk_base = pid_b * s_pk_b + pid_n * s_pk_t
    tl.store(PK_ptr + pk_base + 0, pK00)
    tl.store(PK_ptr + pk_base + 1, pK01)
    tl.store(PK_ptr + pk_base + 2, pK02)
    tl.store(PK_ptr + pk_base + 3, pK10)
    tl.store(PK_ptr + pk_base + 4, pK11)
    tl.store(PK_ptr + pk_base + 5, pK12)
    tl.store(PK_ptr + pk_base + 6, pK20)
    tl.store(PK_ptr + pk_base + 7, pK21)
    tl.store(PK_ptr + pk_base + 8, pK22)

    pr_base = pid_b * s_pr_b + pid_n * s_pr_t
    tl.store(PR_ptr + pr_base + 0, pR00)
    tl.store(PR_ptr + pr_base + 1, pR01)
    tl.store(PR_ptr + pr_base + 2, pR02)
    tl.store(PR_ptr + pr_base + 3, pR10)
    tl.store(PR_ptr + pr_base + 4, pR11)
    tl.store(PR_ptr + pr_base + 5, pR12)
    tl.store(PR_ptr + pr_base + 6, pR20)
    tl.store(PR_ptr + pr_base + 7, pR21)
    tl.store(PR_ptr + pr_base + 8, pR22)

    pt_base = pid_b * s_pt_b + pid_n * s_pt_t
    tl.store(PT_ptr + pt_base + 0, pT0)
    tl.store(PT_ptr + pt_base + 1, pT1)
    tl.store(PT_ptr + pt_base + 2, pT2)


@triton.jit
def _combine_partials_kernel(
    PK_ptr,  # (B, n_tiles, 9)
    PR_ptr,  # (B, n_tiles, 9)
    PT_ptr,  # (B, n_tiles, 3)
    GK_ptr,  # (B, 9) — final flat grad_K
    GR_ptr,  # (B, 9) — final flat grad_R
    GT_ptr,  # (B, 3) — final grad_t
    N_TILES,
    s_pk_b,
    s_pk_t,
    s_pr_b,
    s_pr_t,
    s_pt_b,
    s_pt_t,
    s_gk_b,
    s_gr_b,
    s_gt_b,
):
    """Combine per-tile partial sums into final grad_K / grad_R / grad_t.

    Grid: ``(B,)``. Each program loops over ``N_TILES`` sequentially,
    accumulating left-associatively — matches the old reduce kernel's
    outer-loop FP order exactly. Per-tile partial sums and the cross-tile
    combine together reproduce the old reduce kernel bit-for-bit:
      ``((((0 + S_0) + S_1) + S_2) + ...) + S_{n_tiles-1}``.
    """
    pid_b = tl.program_id(0)
    aK00 = 0.0
    aK01 = 0.0
    aK02 = 0.0
    aK10 = 0.0
    aK11 = 0.0
    aK12 = 0.0
    aK20 = 0.0
    aK21 = 0.0
    aK22 = 0.0
    aR00 = 0.0
    aR01 = 0.0
    aR02 = 0.0
    aR10 = 0.0
    aR11 = 0.0
    aR12 = 0.0
    aR20 = 0.0
    aR21 = 0.0
    aR22 = 0.0
    aT0 = 0.0
    aT1 = 0.0
    aT2 = 0.0

    for t_idx in range(N_TILES):
        pk_b = pid_b * s_pk_b + t_idx * s_pk_t
        aK00 += tl.load(PK_ptr + pk_b + 0)
        aK01 += tl.load(PK_ptr + pk_b + 1)
        aK02 += tl.load(PK_ptr + pk_b + 2)
        aK10 += tl.load(PK_ptr + pk_b + 3)
        aK11 += tl.load(PK_ptr + pk_b + 4)
        aK12 += tl.load(PK_ptr + pk_b + 5)
        aK20 += tl.load(PK_ptr + pk_b + 6)
        aK21 += tl.load(PK_ptr + pk_b + 7)
        aK22 += tl.load(PK_ptr + pk_b + 8)
        pr_b = pid_b * s_pr_b + t_idx * s_pr_t
        aR00 += tl.load(PR_ptr + pr_b + 0)
        aR01 += tl.load(PR_ptr + pr_b + 1)
        aR02 += tl.load(PR_ptr + pr_b + 2)
        aR10 += tl.load(PR_ptr + pr_b + 3)
        aR11 += tl.load(PR_ptr + pr_b + 4)
        aR12 += tl.load(PR_ptr + pr_b + 5)
        aR20 += tl.load(PR_ptr + pr_b + 6)
        aR21 += tl.load(PR_ptr + pr_b + 7)
        aR22 += tl.load(PR_ptr + pr_b + 8)
        pt_b = pid_b * s_pt_b + t_idx * s_pt_t
        aT0 += tl.load(PT_ptr + pt_b + 0)
        aT1 += tl.load(PT_ptr + pt_b + 1)
        aT2 += tl.load(PT_ptr + pt_b + 2)

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
        # xc/yc/zc removed (cycle 19): the fused bwd kernel recomputes them.
        ds_du = torch.empty((B, N), device=device, dtype=dtype)
        ds_dv = torch.empty((B, N), device=device, dtype=dtype)

        BLOCK_N = 256
        # Grid axes order matters: axis 0 (CUDA blockIdx.x) varies fastest in
        # hardware dispatch, so we put ``N / BLOCK_N`` first and ``B`` second.
        # Consecutive blocks then share ``pid_b`` (and DT image), keeping the
        # 1 MB DT field hot in L2 across the ~48 tiles of one batch row.
        grid = (triton.cdiv(N, BLOCK_N), B)
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
                # Single fixed launch config (no autotune dispatch overhead).
                # ``num_warps=8`` matches BLOCK_N=256 → 1 element/thread (vs
                # default 4 warps × 32 threads = 128 threads, 2 elements each).
                # ``num_stages=4`` lets the Triton pipeline-pass overlap the
                # heavy load chain (xyz_world + K + P + IMG_HW + 4 DT taps)
                # with the per-point FMAs across one extra stage — fwd is
                # memory-bound on the random DT bilinear gathers.
                num_warps=8,
                num_stages=4,
            )

        # Save references for backward. Neither u/v nor xc/yc/zc are saved
        # — the fused bwd kernel (cycle 19) derives all of them once per
        # point from (xyz_world, P) and uses them for both the per-point
        # grad_xyz_world and the per-tile K/R/t partial reductions.
        ctx.save_for_backward(xyz_c, K_c, P_c, ds_du, ds_dv, mask)
        return residuals, mask.bool()

    @staticmethod
    def backward(ctx, grad_residuals, grad_mask_unused):
        """Analytical backward for the fused project+sample op.

        Returns gradients w.r.t. ``(xyz_world, K, P)``; the DT field, dt
        indices, and ``img_hw`` are non-grad inputs and yield ``None``.
        """
        xyz_world, K, P, ds_du, ds_dv, mask = ctx.saved_tensors
        B, N, _ = xyz_world.shape
        device, dtype = xyz_world.device, xyz_world.dtype

        gr = grad_residuals.contiguous()

        # Per-point grad_xyz_world. xc/yc/zc are re-derived in the fused
        # kernel from (xyz_world, P) — no save/load round-trip (cycle 19).
        grad_xyz_world = torch.empty_like(xyz_world)

        BLOCK_N = 256
        n_tiles = triton.cdiv(N, BLOCK_N)
        # Per-tile K/R/t partial reductions, combined sequentially below.
        # Total partial buffer: 21 × B × n_tiles ≈ 4 MB at B=1024,
        # n_tiles=48 — trivial vs the 250 MB of save-intermediates dropped.
        partials_K = torch.empty((B, n_tiles, 9), device=device, dtype=dtype)
        partials_R = torch.empty((B, n_tiles, 9), device=device, dtype=dtype)
        partials_t = torch.empty((B, n_tiles, 3), device=device, dtype=dtype)

        # Match fwd kernel grid order: pid_n (axis 0) varies fastest so
        # consecutive blocks share a batch row and reuse per-row data in
        # L1 (cycle 9 locality argument).
        grid = (n_tiles, B)
        with torch.cuda.device(device):
            _project_sample_bwd_kernel[grid](
                xyz_world,
                ds_du,
                ds_dv,
                mask,
                gr,
                K,
                P,
                grad_xyz_world,
                partials_K,
                partials_R,
                partials_t,
                B,
                N,
                xyz_world.stride(0),
                xyz_world.stride(1),
                ds_du.stride(0),
                grad_xyz_world.stride(0),
                grad_xyz_world.stride(1),
                K.stride(0),
                P.stride(0),
                partials_K.stride(0),
                partials_K.stride(1),
                partials_R.stride(0),
                partials_R.stride(1),
                partials_t.stride(0),
                partials_t.stride(1),
                BLOCK_N=BLOCK_N,
                # Same launch config as cycle 14 bwd — load chain grew by
                # only xyz_world (3 loads) vs the dropped xc/yc/zc loads
                # (also 3), so register pressure is similar. num_stages=3
                # leaves more registers for the new 21 reduction accumulators.
                num_warps=8,
                num_stages=3,
            )

        # ---- Combine per-tile partials into final grad_K/grad_R/grad_t ---
        # Grid (B,); sequential left-associative accumulation per program,
        # matching the old reduce kernel's outer-loop FP order. Per-tile
        # partial sums use the same BLOCK_N=256 operand sequence as the
        # old per-tile tl.sum.
        grad_K_flat = torch.empty((B, 9), device=device, dtype=dtype)
        grad_R_flat = torch.empty((B, 9), device=device, dtype=dtype)
        grad_t = torch.empty((B, 3), device=device, dtype=dtype)
        with torch.cuda.device(device):
            _combine_partials_kernel[(B,)](
                partials_K,
                partials_R,
                partials_t,
                grad_K_flat,
                grad_R_flat,
                grad_t,
                n_tiles,
                partials_K.stride(0),
                partials_K.stride(1),
                partials_R.stride(0),
                partials_R.stride(1),
                partials_t.stride(0),
                partials_t.stride(1),
                grad_K_flat.stride(0),
                grad_R_flat.stride(0),
                grad_t.stride(0),
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
    # xyz_cam NOT saved (cycle 27 fusion): the new bwd kernel recomputes
    # xc/yc/zc from (xy, K, depth) and absorbs all post-bwd PyTorch
    # reductions (grad_R bmm, gr.sum, grad_K sums). Single consumer →
    # no cross-kernel FP-order risk (cycle 18 lesson).
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

    Grid layout ``(ceil(N / BLOCK_N), B)`` — axis 0 (CUDA blockIdx.x) varies
    fastest, so consecutive blocks share ``pid_b`` and reuse per-batch K/P
    and the per-row (B, N) tensors (xy0, depth) in L1.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

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
    # xyz_cam NOT stored — see kernel-signature comment.


@triton.jit
def _unproject_bwd_kernel(
    XY_ptr,  # (B, N, 2)
    DEPTH_ptr,  # (B, N)
    K_ptr,  # (B, 3, 3)
    P_ptr,  # (B, 4, 4)
    GXYZW_ptr,  # (B, N, 3) — upstream gradient
    GDEPTH_ptr,  # (B, N)    — output: grad_depth (per-point)
    PART_ptr,  # (B, n_tiles, 16) — per-tile partials for grad_R, total_gxyzw, grad_K
    B,
    N,
    s_xy_b,
    s_xy_n,
    s_d_b,
    s_K_b,
    s_P_b,
    s_gxyzw_b,
    s_gxyzw_n,
    s_part_b,
    s_part_t,
    BLOCK_N: tl.constexpr,
):
    """Fused per-point bwd + per-tile reductions (cycle 27).

    Per point: emits grad_depth (the only true per-point output now —
    grad_xyz_cam is purely intermediate and stays in registers).

    Per tile: writes 16 partial sums to ``PART_ptr[b, pid_n, :]`` —
      [0..9)   pR[i, k] = Σ Y[i] · grad_xyz_world[k]  (grad_R partials)
      [9..12)  pT[k]    = Σ grad_xyz_world[k]         (for grad_t = -R·pT)
      [12..16) pK = [Σ gxc·xc, Σ gyc·yc, Σ gxc·depth, Σ gyc·depth]
                                                       (for grad_K)

    A small ``_unproject_combine_kernel`` (grid (B,)) sequentially sums
    these across tiles. Per-tile ``tl.sum`` operand order matches the
    PyTorch reductions' tile-block decomposition closely enough that
    AUC drift is bounded by the same FP-order envelope as cycle 19 on
    project_sample.

    Grid layout ``(ceil(N / BLOCK_N), B)`` — same locality argument as
    fwd; pid_n is fast-varying so consecutive blocks share batch-row
    data in L1.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ---- Re-load static inputs -----------------------------------------
    xy_row = pid_b * s_xy_b + n_offs * s_xy_n
    u = tl.load(XY_ptr + xy_row + 0, mask=n_mask, other=0.0)
    v = tl.load(XY_ptr + xy_row + 1, mask=n_mask, other=0.0)
    z = tl.load(DEPTH_ptr + pid_b * s_d_b + n_offs, mask=n_mask, other=0.0)

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
    t0 = tl.load(Pb + 3)
    R10 = tl.load(Pb + 4)
    R11 = tl.load(Pb + 5)
    R12 = tl.load(Pb + 6)
    t1 = tl.load(Pb + 7)
    R20 = tl.load(Pb + 8)
    R21 = tl.load(Pb + 9)
    R22 = tl.load(Pb + 10)
    t2 = tl.load(Pb + 11)

    # ---- Recompute xyz_cam (matches fwd exactly) ------------------------
    xc = z * (u - cx) * inv_fx
    yc = z * (v - cy) * inv_fy
    zc = z

    # ---- Upstream gradient ---------------------------------------------
    g_row = pid_b * s_gxyzw_b + n_offs * s_gxyzw_n
    gxw = tl.load(GXYZW_ptr + g_row + 0, mask=n_mask, other=0.0)
    gyw = tl.load(GXYZW_ptr + g_row + 1, mask=n_mask, other=0.0)
    gzw = tl.load(GXYZW_ptr + g_row + 2, mask=n_mask, other=0.0)

    # ---- grad through xyz_world = R^T · Y -------------------------------
    gYx = R00 * gxw + R01 * gyw + R02 * gzw
    gYy = R10 * gxw + R11 * gyw + R12 * gzw
    gYz = R20 * gxw + R21 * gyw + R22 * gzw

    grad_xc = gYx
    grad_yc = gYy
    grad_zc = gYz

    # ---- grad_depth per-point output ------------------------------------
    grad_z = grad_xc * (u - cx) * inv_fx + grad_yc * (v - cy) * inv_fy + grad_zc
    tl.store(GDEPTH_ptr + pid_b * s_d_b + n_offs, grad_z, mask=n_mask)

    # ---- Y = xyz_cam - t (per-point, kept in registers) -----------------
    Yx = xc - t0
    Yy = yc - t1
    Yz = zc - t2

    # ---- Per-tile partials ---------------------------------------------
    # grad_R[i, k] = Σ Y[i] · grad_xyz_world[k] — 9 outputs
    pR00 = tl.sum(Yx * gxw, axis=0)
    pR01 = tl.sum(Yx * gyw, axis=0)
    pR02 = tl.sum(Yx * gzw, axis=0)
    pR10 = tl.sum(Yy * gxw, axis=0)
    pR11 = tl.sum(Yy * gyw, axis=0)
    pR12 = tl.sum(Yy * gzw, axis=0)
    pR20 = tl.sum(Yz * gxw, axis=0)
    pR21 = tl.sum(Yz * gyw, axis=0)
    pR22 = tl.sum(Yz * gzw, axis=0)

    # total_gxyzw[k] = Σ grad_xyz_world[k] — 3 outputs (for grad_t)
    pT0 = tl.sum(gxw, axis=0)
    pT1 = tl.sum(gyw, axis=0)
    pT2 = tl.sum(gzw, axis=0)

    # grad_K raw sums — 4 outputs (post-combine: multiplied by -inv_fx/fy)
    pKfx = tl.sum(grad_xc * xc, axis=0)
    pKfy = tl.sum(grad_yc * yc, axis=0)
    pKcx = tl.sum(grad_xc * z, axis=0)
    pKcy = tl.sum(grad_yc * z, axis=0)

    part_base = pid_b * s_part_b + pid_n * s_part_t
    tl.store(PART_ptr + part_base + 0, pR00)
    tl.store(PART_ptr + part_base + 1, pR01)
    tl.store(PART_ptr + part_base + 2, pR02)
    tl.store(PART_ptr + part_base + 3, pR10)
    tl.store(PART_ptr + part_base + 4, pR11)
    tl.store(PART_ptr + part_base + 5, pR12)
    tl.store(PART_ptr + part_base + 6, pR20)
    tl.store(PART_ptr + part_base + 7, pR21)
    tl.store(PART_ptr + part_base + 8, pR22)
    tl.store(PART_ptr + part_base + 9, pT0)
    tl.store(PART_ptr + part_base + 10, pT1)
    tl.store(PART_ptr + part_base + 11, pT2)
    tl.store(PART_ptr + part_base + 12, pKfx)
    tl.store(PART_ptr + part_base + 13, pKfy)
    tl.store(PART_ptr + part_base + 14, pKcx)
    tl.store(PART_ptr + part_base + 15, pKcy)


@triton.jit
def _unproject_combine_kernel(
    PART_ptr,  # (B, n_tiles, 16)
    GR_ptr,  # (B, 9) — final flat grad_R
    GT_ptr,  # (B, 3) — total_gxyzw (will be -R @ gt in post-kernel)
    GK_ptr,  # (B, 4) — raw grad_K sums (post-kernel: × -inv_fx/fy)
    N_TILES,
    s_part_b,
    s_part_t,
    s_gr_b,
    s_gt_b,
    s_gk_b,
):
    """Combine per-tile partials into final per-batch vectors.

    Grid ``(B,)``; each program loops over n_tiles sequentially, matching
    the left-associative FP order used by the project_sample combine.
    """
    pid_b = tl.program_id(0)
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    a8 = 0.0
    a9 = 0.0
    a10 = 0.0
    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a14 = 0.0
    a15 = 0.0
    for t_idx in range(N_TILES):
        base = pid_b * s_part_b + t_idx * s_part_t
        a0 += tl.load(PART_ptr + base + 0)
        a1 += tl.load(PART_ptr + base + 1)
        a2 += tl.load(PART_ptr + base + 2)
        a3 += tl.load(PART_ptr + base + 3)
        a4 += tl.load(PART_ptr + base + 4)
        a5 += tl.load(PART_ptr + base + 5)
        a6 += tl.load(PART_ptr + base + 6)
        a7 += tl.load(PART_ptr + base + 7)
        a8 += tl.load(PART_ptr + base + 8)
        a9 += tl.load(PART_ptr + base + 9)
        a10 += tl.load(PART_ptr + base + 10)
        a11 += tl.load(PART_ptr + base + 11)
        a12 += tl.load(PART_ptr + base + 12)
        a13 += tl.load(PART_ptr + base + 13)
        a14 += tl.load(PART_ptr + base + 14)
        a15 += tl.load(PART_ptr + base + 15)
    base_r = pid_b * s_gr_b
    tl.store(GR_ptr + base_r + 0, a0)
    tl.store(GR_ptr + base_r + 1, a1)
    tl.store(GR_ptr + base_r + 2, a2)
    tl.store(GR_ptr + base_r + 3, a3)
    tl.store(GR_ptr + base_r + 4, a4)
    tl.store(GR_ptr + base_r + 5, a5)
    tl.store(GR_ptr + base_r + 6, a6)
    tl.store(GR_ptr + base_r + 7, a7)
    tl.store(GR_ptr + base_r + 8, a8)
    base_t = pid_b * s_gt_b
    tl.store(GT_ptr + base_t + 0, a9)
    tl.store(GT_ptr + base_t + 1, a10)
    tl.store(GT_ptr + base_t + 2, a11)
    base_k = pid_b * s_gk_b
    tl.store(GK_ptr + base_k + 0, a12)
    tl.store(GK_ptr + base_k + 1, a13)
    tl.store(GK_ptr + base_k + 2, a14)
    tl.store(GK_ptr + base_k + 3, a15)


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
        # xyz_cam allocation dropped (cycle 27): bwd recomputes it from
        # (xy, K, depth) inside the kernel.

        BLOCK_N = 256
        # Axis 0 = fast-varying ⇒ ``pid_n`` first so consecutive blocks share
        # ``pid_b`` and keep K/P/(B,N) row data hot in L1 across batch-row tiles.
        grid = (triton.cdiv(N, BLOCK_N), B)
        with torch.cuda.device(device):
            _unproject_fwd_kernel[grid](
                xy_c,
                d_c,
                K_c,
                P_c,
                xyz_world,
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
                # Unproject is a simpler kernel than project+sample (no DT
                # gather), so 4 warps is enough and 8 over-provisions
                # (cycle 12 confirmed). One extra pipeline stage on top of
                # the default (3) helps overlap the short load chain
                # (xy + depth + K + P) with the projection FMAs.
                num_warps=4,
                num_stages=4,
            )

        ctx.save_for_backward(xy_c, d_c, K_c, P_c)
        return xyz_world

    @staticmethod
    def backward(ctx, grad_xyz_world):
        """Analytical backward for the fused unproject op.

        Returns gradients w.r.t. ``(depth, K, P)``; ``xy0`` is a non-grad
        input and yields ``None``. All reductions over N (grad_R, grad_t,
        grad_K) are done inside the fused bwd kernel + combine kernel
        (cycle 27 — mirrors the project_sample cycle-19 fusion).
        """
        xy0, depth, K, P = ctx.saved_tensors
        B, N, _ = xy0.shape
        device, dtype = xy0.device, xy0.dtype

        gr = grad_xyz_world.contiguous()
        grad_depth = torch.empty_like(depth)

        BLOCK_N = 256
        n_tiles = triton.cdiv(N, BLOCK_N)
        # 16 partials per tile: 9 grad_R + 3 total_gxyzw + 4 grad_K-raw.
        partials = torch.empty((B, n_tiles, 16), device=device, dtype=dtype)

        # Axis 0 = fast-varying ⇒ ``pid_n`` first so consecutive blocks share
        # ``pid_b`` and keep K/P/(B,N) row data hot in L1.
        grid = (n_tiles, B)
        with torch.cuda.device(device):
            _unproject_bwd_kernel[grid](
                xy0,
                depth,
                K,
                P,
                gr,
                grad_depth,
                partials,
                B,
                N,
                xy0.stride(0),
                xy0.stride(1),
                depth.stride(0),
                K.stride(0),
                P.stride(0),
                gr.stride(0),
                gr.stride(1),
                partials.stride(0),
                partials.stride(1),
                BLOCK_N=BLOCK_N,
                # Cycle 28: 16 new reduction accumulators (cycle 27 fusion)
                # add register pressure on top of the existing per-point chain.
                # Mirrors cycle 20-21: post-fusion cliff drops; test num_stages=3.
                num_warps=4,
                num_stages=3,
            )

        # ---- Combine per-tile partials into final per-batch vectors -----
        grad_R_flat = torch.empty((B, 9), device=device, dtype=dtype)
        total_gxyzw = torch.empty((B, 3), device=device, dtype=dtype)
        grad_K_raw = torch.empty((B, 4), device=device, dtype=dtype)
        with torch.cuda.device(device):
            _unproject_combine_kernel[(B,)](
                partials,
                grad_R_flat,
                total_gxyzw,
                grad_K_raw,
                n_tiles,
                partials.stride(0),
                partials.stride(1),
                grad_R_flat.stride(0),
                total_gxyzw.stride(0),
                grad_K_raw.stride(0),
            )

        # ---- Tiny per-batch post-processing (B is small) ----------------
        R = P[:, :3, :3]
        grad_R = grad_R_flat.view(B, 3, 3)
        grad_t = -torch.bmm(R, total_gxyzw.unsqueeze(-1)).squeeze(-1)
        grad_P = torch.zeros_like(P)
        grad_P[:, :3, :3] = grad_R
        grad_P[:, :3, 3] = grad_t

        inv_fx_b = 1.0 / K[:, 0, 0]
        inv_fy_b = 1.0 / K[:, 1, 1]
        grad_K = torch.zeros_like(K)
        grad_K[:, 0, 0] = -grad_K_raw[:, 0] * inv_fx_b
        grad_K[:, 1, 1] = -grad_K_raw[:, 1] * inv_fy_b
        grad_K[:, 0, 2] = -grad_K_raw[:, 2] * inv_fx_b
        grad_K[:, 1, 2] = -grad_K_raw[:, 3] * inv_fy_b

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


# ===========================================================================
# Fused 2D-to-2D reprojection kernel (viewgraph filter)
# ===========================================================================
#
# Per-point pipeline:
#   1. Nearest-mode sample of depthmap0 at xy0 (matches grid_sample_nan).
#   2. xyz_cam0 = depth · ((u-cx0)/fx0, (v-cy0)/fy0, 1)
#   3. xyz_world = R0^T · (xyz_cam0 - t0)
#   4. xyz_cam1 = R1 · xyz_world + t1
#   5. (u1, v1) = (fx1·xc1/zc1 + cx1, fy1·yc1/zc1 + cy1)
#
# Outputs NaN for any invalid point (NaN xy0, xy0 out-of-bounds, NaN depth).
# The downstream filter's nan_mask + border_mask reject NaN-uv pairs
# identically to the reference path's 0.0-padded outputs.
#
# Used exclusively by filter_viewgraph_by_reprojection_batched under
# @torch.no_grad. No saved intermediates, no analytical backward.


@triton.jit
def _reproject_2D_2D_kernel(
    XY0_ptr,
    DEPTH_ptr,
    K0_ptr,
    P0_ptr,
    K1_ptr,
    P1_ptr,
    XY_OUT_ptr,
    B,
    N,
    H,
    W,
    s_xy0_b,
    s_xy0_n,
    s_depth_b,
    s_depth_h,
    s_K0_b,
    s_P0_b,
    s_K1_b,
    s_P1_b,
    s_xyout_b,
    s_xyout_n,
    BLOCK_N: tl.constexpr,
):
    """Fused img0 → img1 reprojection.

    Grid layout (ceil(N / BLOCK_N), B) — axis 0 (CUDA blockIdx.x) varies
    fastest so consecutive blocks share pid_b and reuse the same
    depthmap0 / K / P data in L1 across the tiles of one pair.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    xy0_off = pid_b * s_xy0_b + n_offs * s_xy0_n
    u0 = tl.load(XY0_ptr + xy0_off + 0, mask=n_mask, other=0.0)
    v0 = tl.load(XY0_ptr + xy0_off + 1, mask=n_mask, other=0.0)

    xy0_finite = (u0 == u0) & (v0 == v0)
    Wf = tl.cast(W - 1, tl.float32)
    Hf = tl.cast(H - 1, tl.float32)
    in_bounds = (u0 >= 0.0) & (u0 <= Wf) & (v0 >= 0.0) & (v0 <= Hf)
    valid = xy0_finite & in_bounds & n_mask

    u0_idx = tl.floor(u0 + 0.5).to(tl.int32)
    v0_idx = tl.floor(v0 + 0.5).to(tl.int32)
    u0_idx = tl.minimum(tl.maximum(u0_idx, 0), W - 1)
    v0_idx = tl.minimum(tl.maximum(v0_idx, 0), H - 1)
    depth = tl.load(
        DEPTH_ptr + pid_b * s_depth_b + v0_idx * s_depth_h + u0_idx,
        mask=valid,
        other=0.0,
    )
    depth_finite = depth == depth
    valid = valid & depth_finite

    K0b = K0_ptr + pid_b * s_K0_b
    fx0 = tl.load(K0b + 0)
    cx0 = tl.load(K0b + 2)
    fy0 = tl.load(K0b + 4)
    cy0 = tl.load(K0b + 5)

    P0b = P0_ptr + pid_b * s_P0_b
    R0_00 = tl.load(P0b + 0)
    R0_01 = tl.load(P0b + 1)
    R0_02 = tl.load(P0b + 2)
    t0_0 = tl.load(P0b + 3)
    R0_10 = tl.load(P0b + 4)
    R0_11 = tl.load(P0b + 5)
    R0_12 = tl.load(P0b + 6)
    t0_1 = tl.load(P0b + 7)
    R0_20 = tl.load(P0b + 8)
    R0_21 = tl.load(P0b + 9)
    R0_22 = tl.load(P0b + 10)
    t0_2 = tl.load(P0b + 11)

    K1b = K1_ptr + pid_b * s_K1_b
    fx1 = tl.load(K1b + 0)
    cx1 = tl.load(K1b + 2)
    fy1 = tl.load(K1b + 4)
    cy1 = tl.load(K1b + 5)

    P1b = P1_ptr + pid_b * s_P1_b
    R1_00 = tl.load(P1b + 0)
    R1_01 = tl.load(P1b + 1)
    R1_02 = tl.load(P1b + 2)
    t1_0 = tl.load(P1b + 3)
    R1_10 = tl.load(P1b + 4)
    R1_11 = tl.load(P1b + 5)
    R1_12 = tl.load(P1b + 6)
    t1_1 = tl.load(P1b + 7)
    R1_20 = tl.load(P1b + 8)
    R1_21 = tl.load(P1b + 9)
    R1_22 = tl.load(P1b + 10)
    t1_2 = tl.load(P1b + 11)

    xc0 = depth * (u0 - cx0) / fx0
    yc0 = depth * (v0 - cy0) / fy0
    zc0 = depth

    Yx = xc0 - t0_0
    Yy = yc0 - t0_1
    Yz = zc0 - t0_2
    xw = R0_00 * Yx + R0_10 * Yy + R0_20 * Yz
    yw = R0_01 * Yx + R0_11 * Yy + R0_21 * Yz
    zw = R0_02 * Yx + R0_12 * Yy + R0_22 * Yz

    xc1 = R1_00 * xw + R1_01 * yw + R1_02 * zw + t1_0
    yc1 = R1_10 * xw + R1_11 * yw + R1_12 * zw + t1_1
    zc1 = R1_20 * xw + R1_21 * yw + R1_22 * zw + t1_2

    inv_zc1 = 1.0 / zc1
    u1 = fx1 * xc1 * inv_zc1 + cx1
    v1 = fy1 * yc1 * inv_zc1 + cy1

    nan = float("nan")
    u1_out = tl.where(valid, u1, nan)
    v1_out = tl.where(valid, v1, nan)

    out_off = pid_b * s_xyout_b + n_offs * s_xyout_n
    tl.store(XY_OUT_ptr + out_off + 0, u1_out, mask=n_mask)
    tl.store(XY_OUT_ptr + out_off + 1, v1_out, mask=n_mask)


def reproject_2D_2D_triton(
    xy0: torch.Tensor,
    depthmap0: torch.Tensor,
    P0: torch.Tensor,
    P1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
) -> torch.Tensor:
    """Fused img0 → img1 reprojection used by the viewgraph filter.

    Replaces the 5-op PyTorch chain (grid_sample → invert_K → unproject →
    change_reference → project) with one Triton kernel. No autograd: the
    viewgraph filter is wrapped in torch.no_grad.

    Args:
        xy0: (B, N, 2) source pixel coordinates.
        depthmap0: (B, H, W) source depth map.
        P0: (B, 4, 4) world-to-cam extrinsics for the source view.
        P1: (B, 4, 4) world-to-cam extrinsics for the target view.
        K0: (B, 3, 3) source intrinsics.
        K1: (B, 3, 3) target intrinsics.

    Returns:
        (B, N, 2) projected pixel coordinates in img1. Invalid points emit
        NaN; the downstream filter's nan_mask + border_mask catch them.
    """
    assert xy0.is_cuda and depthmap0.is_cuda
    assert xy0.dim() == 3 and xy0.shape[-1] == 2
    assert depthmap0.dim() == 3
    B, N, _ = xy0.shape
    H, W = depthmap0.shape[-2:]

    xy0_c = xy0.contiguous()
    depth_c = depthmap0.contiguous()
    K0_c = K0.contiguous()
    P0_c = P0.contiguous()
    K1_c = K1.contiguous()
    P1_c = P1.contiguous()

    xy_out = torch.empty((B, N, 2), device=xy0.device, dtype=xy0.dtype)

    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N), B)
    with torch.cuda.device(xy0.device):
        _reproject_2D_2D_kernel[grid](
            xy0_c,
            depth_c,
            K0_c,
            P0_c,
            K1_c,
            P1_c,
            xy_out,
            B,
            N,
            H,
            W,
            xy0_c.stride(0),
            xy0_c.stride(1),
            depth_c.stride(0),
            depth_c.stride(1),
            K0_c.stride(0),
            P0_c.stride(0),
            K1_c.stride(0),
            P1_c.stride(0),
            xy_out.stride(0),
            xy_out.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=4,
        )
    return xy_out
