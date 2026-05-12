"""Fused Triton kernel for project-world-to-pixel + bilinear DT sampling.

Replaces the chain
    ``xyz_world @ R^T + t  →  K @ xyz_cam / z  →  F.grid_sample(dt_field, uv)``
with a single Triton kernel for the forward and an analytical PyTorch backward.

Why custom autograd:
  * The reference path expands into ~6 PyTorch ops, each adding an autograd
    node + an intermediate tensor allocation. The custom Function collapses
    this to one node.
  * ``dt_fields`` does not require gradients, so the bilinear backward is a
    pure gather (4 corner reads per point) — no scatter / atomic accumulation
    into the DT field is needed.
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
    XYZ_ptr,    # (B, N, 3) float
    K_ptr,      # (B, 3, 3) float
    P_ptr,      # (B, 4, 4) float
    DT_ptr,     # (N_img, H, W) float — *source* DT tensor (not gathered)
    DT_IDX_ptr, # (B,) int64 — maps batch row → image index in DT_ptr
    OUT_ptr,    # (B, N) float — residuals
    MASK_ptr,   # (B, N) uint8 — 1 if inside
    # Saved intermediates for backward
    XC_ptr, YC_ptr, ZC_ptr,
    U_ptr, V_ptr,
    # Sizes
    B, N, H, W,
    # Strides (all element-wise)
    s_xyz_b, s_xyz_n,
    s_K_b,
    s_P_b,
    s_dt_b, s_dt_h,
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
    R00 = tl.load(Pb + 0); R01 = tl.load(Pb + 1); R02 = tl.load(Pb + 2);  t0 = tl.load(Pb + 3)
    R10 = tl.load(Pb + 4); R11 = tl.load(Pb + 5); R12 = tl.load(Pb + 6);  t1 = tl.load(Pb + 7)
    R20 = tl.load(Pb + 8); R21 = tl.load(Pb + 9); R22 = tl.load(Pb + 10); t2 = tl.load(Pb + 11)

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
    # Inside iff u, v are inside the image *and* zc > 0 *and* u/v are finite.
    # The finiteness check guards against zc ⇒ 0⁺ producing ±Inf/NaN in u, v
    # (which the comparison-based bounds test alone would reject, but only by
    # accident — being explicit makes the intent obvious).
    Wf = tl.cast(W, tl.float32)
    Hf = tl.cast(H, tl.float32)
    finite_uv = (u == u) & (v == v) & (u * 0.0 == 0.0) & (v * 0.0 == 0.0)
    inside = (
        (u >= 0.0) & (u < Wf) & (v >= 0.0) & (v < Hf) & (zc > 0.0) & finite_uv
    )

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
    d01 = tl.load(DT_ptr + base + v0c * s_dt_h + u1,  mask=load_mask, other=0.0)
    d10 = tl.load(DT_ptr + base + v1  * s_dt_h + u0c, mask=load_mask, other=0.0)
    d11 = tl.load(DT_ptr + base + v1  * s_dt_h + u1,  mask=load_mask, other=0.0)

    w00 = (1.0 - du) * (1.0 - dv)
    w01 = du * (1.0 - dv)
    w10 = (1.0 - du) * dv
    w11 = du * dv
    sampled = d00 * w00 + d01 * w01 + d10 * w10 + d11 * w11
    sampled = tl.where(inside, sampled, 0.0)

    # ---- Store outputs ---------------------------------------------------
    out_off = pid_b * s_out_b + n_offs
    # Defensive: ensure residual is finite even though the math above should
    # already guarantee it (sampled = 0 for !inside via tl.where).
    sampled = tl.where(sampled == sampled, sampled, 0.0)  # NaN ⇒ 0
    tl.store(OUT_ptr  + out_off, sampled,            mask=n_mask)
    tl.store(MASK_ptr + out_off, inside.to(tl.uint8), mask=n_mask)
    # Intermediates for backward — store SAFE values. For !inside points the
    # raw xc/yc/zc/u/v can be ±Inf or NaN (e.g. point behind camera ⇒ zc ≤ 0
    # ⇒ inv_z = ±Inf ⇒ u, v = ±Inf/NaN). The backward must never see those:
    # ``0 * NaN = NaN`` in IEEE-754, so even a zeroed ``grad_residuals`` would
    # poison the chain. Substitute neutral finite values for !inside points.
    xc_s = tl.where(inside, xc, 0.0)
    yc_s = tl.where(inside, yc, 0.0)
    zc_s = tl.where(inside, zc, 1.0)  # 1.0 ⇒ inv_z = 1.0, finite in bwd
    tl.store(XC_ptr + out_off, xc_s,   mask=n_mask)
    tl.store(YC_ptr + out_off, yc_s,   mask=n_mask)
    tl.store(ZC_ptr + out_off, zc_s,   mask=n_mask)
    tl.store(U_ptr  + out_off, u_safe, mask=n_mask)
    tl.store(V_ptr  + out_off, v_safe, mask=n_mask)


# ---------------------------------------------------------------------------
# Triton backward kernel (per-point, no atomics)
# ---------------------------------------------------------------------------


@triton.jit
def _project_sample_bwd_kernel(
    # Saved from fwd / inputs
    XC_ptr, YC_ptr, ZC_ptr,
    U_ptr, V_ptr,
    MASK_ptr,
    GR_ptr,        # grad_residuals (B, N)
    DT_ptr,        # (N_img, H, W) source DT — same lookup as fwd
    DT_IDX_ptr,    # (B,) int64 — batch row → image index
    K_ptr,         # (B, 3, 3)
    P_ptr,         # (B, 4, 4)
    # Per-point outputs
    GXYZ_ptr,      # grad_xyz_world (B, N, 3)
    GXC_ptr,       # grad_xyz_cam (B, N, 3) — for grad_R/grad_t in PyTorch
    GU_ptr,        # grad_u (B, N) — for grad_K in PyTorch
    GV_ptr,        # grad_v (B, N)
    # Sizes
    B, N, H, W,
    # Strides
    s_xc_b,        # stride between batches in (B, N) scalar tensors
    s_gxyz_b, s_gxyz_n,
    s_dt_b, s_dt_h,
    s_K_b, s_P_b,
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
    u  = tl.load(U_ptr  + row, mask=n_mask, other=0.0)
    v  = tl.load(V_ptr  + row, mask=n_mask, other=0.0)
    inside_u8 = tl.load(MASK_ptr + row, mask=n_mask, other=0)
    inside = inside_u8 != 0
    gr = tl.load(GR_ptr + row, mask=n_mask, other=0.0)
    # Zero-out grad for outside points up-front; they contributed 0 to the loss.
    gr = tl.where(inside, gr, 0.0)

    # ---- Re-derive bilinear corner indices (must match fwd kernel) -----
    # The fwd already stored u_safe/v_safe (== 0 for !inside), so these are
    # guaranteed finite. Belt-and-braces: an extra ``where`` keeps any
    # downstream surprise from poisoning the gradient.
    u_s = tl.where(inside, u, 0.0)
    v_s = tl.where(inside, v, 0.0)
    u0 = tl.floor(u_s).to(tl.int32)
    v0 = tl.floor(v_s).to(tl.int32)
    du = u_s - tl.cast(u0, tl.float32)
    dv = v_s - tl.cast(v0, tl.float32)
    u0c = tl.maximum(u0, 0)
    v0c = tl.maximum(v0, 0)
    u1  = tl.minimum(u0 + 1, W - 1)
    v1  = tl.minimum(v0 + 1, H - 1)

    # Look up which image this batch row reads from. The source DT tensor is
    # shared across batches; no per-batch (B, H, W) copy needed.
    img_idx = tl.load(DT_IDX_ptr + pid_b)
    base = img_idx * s_dt_b
    load_mask = inside & n_mask
    d00 = tl.load(DT_ptr + base + v0c * s_dt_h + u0c, mask=load_mask, other=0.0)
    d01 = tl.load(DT_ptr + base + v0c * s_dt_h + u1,  mask=load_mask, other=0.0)
    d10 = tl.load(DT_ptr + base + v1  * s_dt_h + u0c, mask=load_mask, other=0.0)
    d11 = tl.load(DT_ptr + base + v1  * s_dt_h + u1,  mask=load_mask, other=0.0)

    # ---- Bilinear gradient: ds/du, ds/dv -------------------------------
    ds_du = (d01 - d00) * (1.0 - dv) + (d11 - d10) * dv
    ds_dv = (d10 - d00) * (1.0 - du) + (d11 - d01) * du
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
    R00 = tl.load(Pb + 0); R01 = tl.load(Pb + 1); R02 = tl.load(Pb + 2)
    R10 = tl.load(Pb + 4); R11 = tl.load(Pb + 5); R12 = tl.load(Pb + 6)
    R20 = tl.load(Pb + 8); R21 = tl.load(Pb + 9); R22 = tl.load(Pb + 10)
    # grad_xyz_world[k] = sum_i R[i,k] * grad_xyz_cam[i]
    g_x = R00 * grad_xc + R10 * grad_yc + R20 * grad_zc
    g_y = R01 * grad_xc + R11 * grad_yc + R21 * grad_zc
    g_z = R02 * grad_xc + R12 * grad_yc + R22 * grad_zc

    # ---- Final defensive mask -------------------------------------------
    # Every value below is mathematically zero for !inside points (gr=0 and
    # safe intermediates), but a stray NaN would still poison the optimizer.
    # Explicitly clamp to 0.0 for !inside ⇒ guaranteed-finite gradients.
    g_x     = tl.where(inside, g_x,     0.0)
    g_y     = tl.where(inside, g_y,     0.0)
    g_z     = tl.where(inside, g_z,     0.0)
    grad_xc = tl.where(inside, grad_xc, 0.0)
    grad_yc = tl.where(inside, grad_yc, 0.0)
    grad_zc = tl.where(inside, grad_zc, 0.0)
    grad_u  = tl.where(inside, grad_u,  0.0)
    grad_v  = tl.where(inside, grad_v,  0.0)

    # ---- Store outputs --------------------------------------------------
    gxyz_row = pid_b * s_gxyz_b + n_offs * s_gxyz_n
    tl.store(GXYZ_ptr + gxyz_row + 0, g_x, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 1, g_y, mask=n_mask)
    tl.store(GXYZ_ptr + gxyz_row + 2, g_z, mask=n_mask)
    # grad_xyz_cam: same layout (B, N, 3)
    tl.store(GXC_ptr + gxyz_row + 0, grad_xc, mask=n_mask)
    tl.store(GXC_ptr + gxyz_row + 1, grad_yc, mask=n_mask)
    tl.store(GXC_ptr + gxyz_row + 2, grad_zc, mask=n_mask)
    # grad_u, grad_v: (B, N)
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
    def forward(ctx, xyz_world, K, P, dt_fields_src, dt_indices):
        """Args
            xyz_world: (B, N, 3)
            K: (B, 3, 3), P: (B, 4, 4)
            dt_fields_src: (N_img, 1, H, W) or (N_img, H, W) — the *source*
                DT tensor (not gathered per batch).
            dt_indices: (B,) int64 — for each batch row, the image index to
                read from in ``dt_fields_src``.
        """
        assert xyz_world.is_cuda and K.is_cuda and P.is_cuda
        assert dt_fields_src.is_cuda and dt_indices.is_cuda
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
        assert idx_c.shape == (B,), (
            f"dt_indices must be shape ({B},), got {idx_c.shape}"
        )
        device = xyz_c.device
        dtype = xyz_c.dtype

        residuals = torch.empty((B, N), device=device, dtype=dtype)
        mask = torch.empty((B, N), device=device, dtype=torch.uint8)
        xc = torch.empty((B, N), device=device, dtype=dtype)
        yc = torch.empty((B, N), device=device, dtype=dtype)
        zc = torch.empty((B, N), device=device, dtype=dtype)
        u = torch.empty((B, N), device=device, dtype=dtype)
        v = torch.empty((B, N), device=device, dtype=dtype)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        _project_sample_fwd_kernel[grid](
            xyz_c, K_c, P_c, dt_c, idx_c,
            residuals, mask,
            xc, yc, zc, u, v,
            B, N, H, W,
            xyz_c.stride(0), xyz_c.stride(1),
            K_c.stride(0),
            P_c.stride(0),
            dt_c.stride(0), dt_c.stride(1),
            residuals.stride(0),
            BLOCK_N=BLOCK_N,
        )

        # Save references for backward. dt_c and idx_c are non-grad tensors
        # used only to re-read the same DT pixels in the bwd corner gather.
        ctx.save_for_backward(xyz_c, K_c, P_c, dt_c, idx_c, xc, yc, zc, u, v, mask)
        return residuals, mask.bool()

    @staticmethod
    def backward(ctx, grad_residuals, grad_mask_unused):
        xyz_world, K, P, dt, idx, xc, yc, zc, u, v, mask = ctx.saved_tensors
        B, N, _ = xyz_world.shape
        _, H, W = dt.shape
        device, dtype = xyz_world.device, xyz_world.dtype

        gr = grad_residuals.contiguous()

        # Allocate per-point outputs. The Triton bwd kernel does:
        #   1) gather 4 dt corners, 2) bilinear ds/du, ds/dv,
        #   3) chain through projection (u = fx*xc/zc + cx),
        #   4) chain through R: grad_xyz = R^T @ grad_xyz_cam.
        # All steps stay in registers — no intermediate (B,N,3) allocs and no
        # ~20 PyTorch elementwise launches.
        grad_xyz_world = torch.empty_like(xyz_world)
        grad_xyz_cam = torch.empty_like(xyz_world)
        grad_u = torch.empty((B, N), device=device, dtype=dtype)
        grad_v = torch.empty((B, N), device=device, dtype=dtype)

        BLOCK_N = 256
        grid = (B, triton.cdiv(N, BLOCK_N))
        _project_sample_bwd_kernel[grid](
            xc, yc, zc, u, v, mask,
            gr, dt, idx, K, P,
            grad_xyz_world, grad_xyz_cam, grad_u, grad_v,
            B, N, H, W,
            xc.stride(0),
            grad_xyz_world.stride(0), grad_xyz_world.stride(1),
            dt.stride(0), dt.stride(1),
            K.stride(0), P.stride(0),
            BLOCK_N=BLOCK_N,
        )

        # ---- Reductions over N (small ops, well-optimized in PyTorch) -----
        # grad_R[b,i,j] = sum_n grad_xyz_cam[b,n,i] * xyz_world[b,n,j]
        grad_R = grad_xyz_cam.transpose(-1, -2) @ xyz_world  # (B, 3, 3)
        grad_t = grad_xyz_cam.sum(dim=1)                     # (B, 3)
        grad_P = torch.zeros_like(P)
        grad_P[:, :3, :3] = grad_R
        grad_P[:, :3, 3] = grad_t

        # Used by the K-grad block below — kept here so the math reads in order.
        inv_z = 1.0 / zc

        # ---- Grad wrt K (full 3x3, matching reference autograd) ---------
        # The forward implicitly treats K as the general matrix
        #   u_hom = K[0,:] · xyz_cam ; v_hom = K[1,:] · xyz_cam ; w_hom = K[2,:] · xyz_cam
        #   u = u_hom / w_hom ; v = v_hom / w_hom
        # Even though the standard K is sparse (only fx, fy, cx, cy nonzero),
        # autograd still emits gradients for the structural-zero entries; we
        # match that for drop-in parity.
        # Letting X = (xc, yc, zc), w_hom = zc:
        #   ∂u/∂K[0,j] =  X[j]/zc           ;  ∂v/∂K[0,j] = 0
        #   ∂u/∂K[1,j] = 0                  ;  ∂v/∂K[1,j] =  X[j]/zc
        #   ∂u/∂K[2,j] = -u · X[j]/zc       ;  ∂v/∂K[2,j] = -v · X[j]/zc
        X0 = xc * inv_z   # (B, N)
        X1 = yc * inv_z
        X2 = zc * inv_z   # ≡ 1, kept symbolic for clarity & numerical parity
        # Row 0 of K: only u contributes
        gK00 = (grad_u * X0).sum(dim=1)
        gK01 = (grad_u * X1).sum(dim=1)
        gK02 = (grad_u * X2).sum(dim=1)
        # Row 1 of K: only v contributes
        gK10 = (grad_v * X0).sum(dim=1)
        gK11 = (grad_v * X1).sum(dim=1)
        gK12 = (grad_v * X2).sum(dim=1)
        # Row 2 of K (perspective row): both u and v contribute (with a -1 factor)
        guv = -(grad_u * u + grad_v * v)
        gK20 = (guv * X0).sum(dim=1)
        gK21 = (guv * X1).sum(dim=1)
        gK22 = (guv * X2).sum(dim=1)
        grad_K = torch.stack(
            [
                torch.stack([gK00, gK01, gK02], dim=-1),
                torch.stack([gK10, gK11, gK12], dim=-1),
                torch.stack([gK20, gK21, gK22], dim=-1),
            ],
            dim=1,
        )  # (B, 3, 3)

        # dt_fields_src and dt_indices do not require gradients.
        return grad_xyz_world, grad_K, grad_P, None, None


def project_and_sample_triton(
    xyz_world: torch.Tensor,
    K1: torch.Tensor,
    P1: torch.Tensor,
    dt_fields_src: torch.Tensor,
    dt_indices: torch.Tensor,
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

    Returns:
        ``(residuals, inside_mask)`` of shapes ``(B, N)`` and ``(B, N)`` bool.
        Outside points get residual ``0.0``; the mask is ``True`` for inside.
    """
    return _ProjectAndSampleTriton.apply(
        xyz_world, K1, P1, dt_fields_src, dt_indices
    )
