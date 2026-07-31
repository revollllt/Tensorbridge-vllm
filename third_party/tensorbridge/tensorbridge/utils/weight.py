import os

import torch

from tensorbridge import dtypes


def _ops():
    from tensorbridge import ops

    return ops


def quantize_weight(
    weight: torch.Tensor,
    dtype: dtypes.DataType,
    scale_dtype: dtypes.DataType | None,
    group_size: int,
    group_size_n: int | None = None,
    has_zero_point: bool = False,
    has_global_scale: bool = False,
    is_fp_zero_point: bool = False,
    pack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.ndim in [2, 3]
    assert not has_zero_point or scale_dtype is not None

    weight = weight.cuda()
    origin_ndim = weight.ndim
    weight = weight.unsqueeze(0) if weight.ndim == 2 else weight
    origin_dtype = dtypes.DataType.from_torch_dtype(weight.dtype)
    e, n, k = weight.shape
    group_size = group_size if group_size > 0 else k

    if group_size_n is not None:
        assert n % group_size_n == 0
        weight = weight.view(e, n // group_size_n, group_size_n, k // group_size, group_size)
        weight = weight.permute(0, 1, 3, 2, 4).contiguous()
        weight = weight.view(e, n * k // group_size_n // group_size, -1)
        group_size = group_size_n * group_size

    quant_group_size = 0
    if scale_dtype is not None:
        quant_group_size = group_size
    elif has_global_scale:
        quant_group_size = weight.nelement() // e
    flatten_weight = weight.view(e, 1, -1)
    use_flatten_weight = scale_dtype is None and has_global_scale
    weight_scale: torch.Tensor | None
    quanted_weight, weight_scale, zero_point = _ops().quant_weight(
        flatten_weight if use_flatten_weight else weight,
        source_dtype_str=str(origin_dtype),
        target_dtype_str=str(dtype),
        group_size=quant_group_size,
        use_e8m0_scale=scale_dtype == dtypes.float8e8m0,
        has_scale=scale_dtype is not None or has_global_scale,
        has_zero_point=has_zero_point,
        is_fp_zero_point=is_fp_zero_point,
    )

    if zero_point.dtype == torch.float32:
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        zero_point = zero_point.to(torch_dtype)

    global_scale = None
    if scale_dtype is None and has_global_scale:
        global_scale = weight_scale.view(-1)
        weight_scale = None
        quanted_weight = quanted_weight.view(e, n, k)
    elif has_global_scale and scale_dtype == dtypes.float8e8m0:
        global_scale = weight_scale.float().view(e, -1).log2().mean(1).exp2()
        weight_scale = (weight_scale.float() / global_scale.view(e, 1, 1)).to(torch.float8_e8m0fnu)
    elif scale_dtype in [dtypes.float16, dtypes.bfloat16]:
        if has_global_scale:
            global_scale = weight_scale.view(e, -1).abs().mean(1)
            weight_scale_view = weight_scale.view(e, -1)
            weight_scale_view = weight_scale_view / global_scale.unsqueeze(1)
            weight_scale = weight_scale_view.view(weight_scale.shape)
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        weight_scale = weight_scale.to(torch_dtype)
    elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
        max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
        torch_dtype = torch.float8_e4m3fn if scale_dtype == dtypes.float8e4m3 else torch.float8_e5m2
        if has_global_scale:
            global_scale1 = weight_scale.view(e, -1).max(1)[0] / max_value
            global_scale2 = weight_scale.view(e, -1).abs().mean(1)
            # NVFP4 (fp4e2m1 weight + fp8e4m3 group scale): the FPMA bridge
            # requires every per-group scale to satisfy code(s) >= 0x1C, i.e.,
            # roughly s >= 0.094. mean-based normalization can leave outlier
            # tiny groups below threshold; max/448-based normalization keeps
            # the spread tight to fp8e4m3's high range for the FPMA contract.
            is_nvfp4 = dtype == dtypes.float4e2m1 and scale_dtype == dtypes.float8e4m3
            use_scale1 = is_nvfp4 or (global_scale1 > global_scale2).any()
            global_scale = global_scale1 if use_scale1 else global_scale2
            weight_scale = weight_scale / global_scale.view(-1, 1, 1)
        weight_scale = weight_scale.to(torch_dtype)

    if group_size_n is not None:
        group_size = group_size // group_size_n
        quanted_weight = quanted_weight.view(
            e,
            n // group_size_n,
            k // group_size,
            group_size_n,
            group_size,
        )
        quanted_weight.permute(0, 1, 3, 2, 4).contiguous()
        quanted_weight = quanted_weight.view(e, n, k)
        assert weight_scale is not None
        weight_scale = weight_scale.view(e, n // group_size_n, k // group_size)

    if origin_ndim == 2:
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None and weight_scale.nelement() > 0:
            weight_scale = weight_scale.squeeze(0)
        if zero_point is not None and zero_point.nelement() > 0:
            zero_point = zero_point.squeeze(0)
        if global_scale is not None and global_scale.nelement() > 0:
            global_scale = global_scale.squeeze(0)

    if pack:
        quanted_weight = _ops().pack_weight(quanted_weight, dtype.num_bits)
        if has_zero_point and not is_fp_zero_point:
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = _ops().pack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)

    final_zero_point = zero_point if zero_point.nelement() > 0 else None

    return quanted_weight, weight_scale, final_zero_point, global_scale


def dequantize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor:
    assert weight.dtype == torch.int32
    weight = weight.cuda()

    if packed:
        weight = _ops().unpack_weight(weight, dtype.num_bits)
        if zero_point is not None and zero_point.dtype == torch.int32:
            zero_point = zero_point.transpose(-1, -2).contiguous().cuda()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = _ops().unpack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape).float()

    if isinstance(dtype, dtypes.FloatingPointType):
        weight = _ops().dequant_weight(weight, dtype.exponent_bits, dtype.mantissa_bits, True)
    else:
        assert isinstance(dtype, dtypes.InergerType)
        assert not dtype.is_signed
        weight = weight.float()

    if zero_point is not None:
        assert weight.size(-1) % zero_point.size(-1) == 0
        group_size = weight.size(-1) // zero_point.size(-1)
        zero_point = zero_point.repeat_interleave(group_size, -1)
        weight = weight - zero_point
    elif isinstance(dtype, dtypes.InergerType):
        assert not dtype.is_signed
        weight = weight - (1 << (dtype.num_bits - 1))

    if weight_scale is not None:
        assert weight.size(-1) % weight_scale.size(-1) == 0
        group_size = weight.size(-1) // weight_scale.size(-1)
        weight_scale = weight_scale.float()
        weight_scale = weight_scale.repeat_interleave(group_size, -1)
        weight = weight * weight_scale

    if global_scale is not None:
        global_scale = global_scale.view(-1, 1, 1)
        if weight.ndim == 2:
            global_scale = global_scale.squeeze(0)
        weight = weight * global_scale

    return weight


def nvfp4_raw_s2r_deint_perm(true_n_warps: int, half_group: bool) -> list[int]:
    """Generic raw-S2R deint permutation: the exact inverse of the device S2R B-load
    (`loader_b.cuh:144`, kUseNvfp4RawS2RDeint branch, K_WARPS==1):

        smem_idx = 64*n_warp_id + 32*(warp_id % 2) + lane_id

    Parameterized over the math-warp count and the half-group flag. Reduces to the
    verified 256-element table at the current config (true_n_warps=4, half_group=True):
    warp_pair_stride = 32*H, warp_half_stride = 32 (lanes/warp), period = 32*W*H, with
    W=true_n_warps and H = 2 (half-group) else 1. `old_for_new[smem_idx] = natural_idx`
    so index_select places each natural element where the loader reads it. Validated as a
    valid loader-inverse (and bit-identical at 256) by tests/test_nvfp4_raw_s2r_deint_perm.py.
    """
    W = true_n_warps
    H = 2 if half_group else 1
    period = 32 * W * H
    warp_pair_stride = 32 * H
    warp_half_stride = 32
    old_for_new = [0] * period
    for warp_pair in range(W):
        for warp_half in range(H):
            for lane in range(32):
                old_idx = (32 * warp_pair + lane) * H + warp_half
                new_idx = warp_pair_stride * warp_pair + warp_half_stride * warp_half + lane
                old_for_new[new_idx] = old_idx
    return old_for_new


def deinterleave_nvfp4_raw_s2r_weight(weight: torch.Tensor, shape_n: int) -> torch.Tensor:
    """Permute repacked NVFP4 raw-B words for the raw S2R layout.

    The permutation is derived from `nvfp4_raw_s2r_deint_perm` (generic over warp count +
    half-group); the current NVFP4 W4A8 config pins true_n_warps=4, half_group=True
    (period 256), bit-identical to the prior hardcoded table.
    """
    assert weight.dtype == torch.int32
    assert weight.is_contiguous()
    assert shape_n % 128 == 0

    ints_per_n128 = 128 * 4
    assert weight.size(-1) % ints_per_n128 == 0
    assert weight.size(-1) == shape_n * 4

    old_for_new = nvfp4_raw_s2r_deint_perm(true_n_warps=4, half_group=True)
    assert len(old_for_new) == 256

    perm = torch.tensor(old_for_new, dtype=torch.long, device=weight.device)
    orig_shape = weight.shape
    view = weight.view(*orig_shape[:-1], weight.size(-1) // ints_per_n128, 256, 2)
    return view.index_select(-2, perm).reshape(orig_shape).contiguous()


def remap_nvfp4_e2m1_for_fpma(packed_weight: torch.Tensor) -> torch.Tensor:
    """Apply the SNC nibble remap used by the FPMA NVFP4 path."""
    assert packed_weight.is_contiguous()

    if packed_weight.dtype == torch.int32:
        bytes_view = packed_weight.view(torch.uint8)
    elif packed_weight.dtype == torch.uint8:
        bytes_view = packed_weight
    else:
        raise TypeError(
            "remap_nvfp4_e2m1_for_fpma expects int32 or uint8 input, "
            f"got {packed_weight.dtype}"
        )

    def remap_nibble(nib: torch.Tensor) -> torch.Tensor:
        mag = nib & 0x07
        sign = nib & 0x08
        remapped_mag = torch.where(
            mag == 0,
            torch.ones_like(mag),
            torch.where(mag == 1, torch.zeros_like(mag), mag),
        )
        return torch.where(mag == 0, torch.ones_like(nib), sign | remapped_mag).to(torch.uint8)

    lo_nib = bytes_view & 0x0F
    hi_nib = (bytes_view >> 4) & 0x0F
    remapped_bytes = (remap_nibble(hi_nib) << 4) | remap_nibble(lo_nib)
    return remapped_bytes.contiguous().view(packed_weight.dtype).view(packed_weight.shape)


def select_nvfp4_prefold_scale(
    packed_weight: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    *,
    group_size: int = 16,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Choose prefold or prefold-1 per group to minimize normal-B8 squared error."""
    if packed_weight.dtype not in (torch.int32, torch.uint8) or packed_weight.ndim != 2:
        raise TypeError("packed_weight must be a 2D int32 or uint8 tensor")
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn or weight_scale_e4m3.ndim != 2:
        raise TypeError("weight_scale_e4m3 must be a 2D E4M3 tensor")
    if packed_weight.device != weight_scale_e4m3.device:
        raise ValueError("weight and weight_scale must be on the same device")
    if group_size != 16:
        raise ValueError("NVFP4 prefold selection currently requires group_size=16")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    packed_bytes = packed_weight.contiguous().view(torch.uint8)
    packed_bytes = packed_bytes.reshape(packed_weight.shape[0], -1)
    expected_bytes = weight_scale_e4m3.shape[1] * group_size // 2
    if packed_bytes.shape != (weight_scale_e4m3.shape[0], expected_bytes):
        raise ValueError("packed weight and scale shapes do not describe the same g16 matrix")

    device = packed_weight.device
    raw_values = torch.arange(0x1C, 0x7F, dtype=torch.uint8, device=device)
    scales = raw_values.view(torch.float8_e4m3fn).float().unsqueeze(1)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    target = (magnitudes * scales / 6.0).to(torch.bfloat16).to(torch.float8_e4m3fn)
    target = target.float()

    prefold = raw_values.to(torch.int16) - 0x1C
    stored_magnitude = torch.tensor(
        [1, 0, 2, 3, 4, 5, 6, 7],
        dtype=torch.int16,
        device=device,
    ).unsqueeze(0)
    base_bytes = prefold.unsqueeze(1) + (stored_magnitude << 2)
    minus_bytes = base_bytes - 1
    base = base_bytes.to(torch.uint8).contiguous().view(torch.float8_e4m3fn).float()
    minus = minus_bytes.to(torch.uint8).contiguous().view(torch.float8_e4m3fn).float()
    base[:, 0] = 0.0
    minus[:, 0] = 0.0
    loss_delta = (minus - target).square() - (base - target).square()

    delta_lut = torch.zeros((256, 8), dtype=torch.float32, device=device)
    delta_lut[0x1C:0x7F] = loss_delta
    raw = weight_scale_e4m3.contiguous().view(torch.uint8)
    output = (raw.to(torch.int16) - 0x1C).clamp_(min=0, max=255).to(torch.uint8)

    groups_per_row = raw.shape[1]
    bytes_per_group = group_size // 2
    for row_start in range(0, raw.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, raw.shape[0])
        grouped = packed_bytes[row_start:row_end].view(
            row_end - row_start,
            groups_per_row,
            bytes_per_group,
        )
        lo_mag = grouped & 0x07
        hi_mag = (grouped >> 4) & 0x07
        raw_chunk = raw[row_start:row_end].long().unsqueeze(-1)
        score = delta_lut[raw_chunk, lo_mag.long()]
        score = score + delta_lut[raw_chunk, hi_mag.long()]
        choose_minus_one = score.sum(dim=-1) < 0.0
        choose_minus_one &= output[row_start:row_end] > 0
        output[row_start:row_end] -= choose_minus_one.to(torch.uint8)

    return output


def reorder_nvfp4_swizzle64_dual_mma_preinterleaved_raw_weight(packed_weight: torch.Tensor) -> torch.Tensor:
    """Pack dual-MMA slots after offline nibble interleave, matching regs_qb layout."""
    weight = packed_weight.contiguous()
    bytes_view = weight.view(torch.uint8)
    n = weight.shape[-2]
    bytes_per_row = weight.shape[-1] * 4
    assert n % 128 == 0, "dual-MMA preinterleaved layout requires 128-row N tiles"
    assert bytes_per_row % 64 == 0, "dual-MMA preinterleaved layout requires 64B K-byte tiles"
    bytes_view = bytes_view.view(*weight.shape[:-1], bytes_per_row)

    chunk_starts = []
    logical_positions = []
    for pair_id in range(2):
        iter0 = pair_id * 2
        iter1 = iter0 + 1
        for warp_id in range(8):
            for lane_id in range(32):
                n_base = lane_id // 4 + (warp_id % 4) * 16 + (warp_id // 4) * 64
                k_byte = (lane_id % 4) * 2
                for iter_id in (iter0, iter1):
                    iter_base = iter_id * 16
                    for n_row in (n_base, n_base + 8):
                        for k_delta in (k_byte, k_byte + 8):
                            chunk_starts.append(n_row * 64 + iter_base + k_delta)
                for slot_byte in range(16):
                    physical = pair_id * 4096 + (warp_id * 32 + lane_id) * 16 + slot_byte
                    logical_positions.append(physical ^ (((physical >> 7) & 0x3) << 4))

    prefix_ndim = bytes_view.ndim - 2
    n_tiles = n // 128
    k_tiles = bytes_per_row // 64
    tiled = bytes_view.view(*bytes_view.shape[:-2], n_tiles, 128, k_tiles, 64)
    tiled = tiled.permute(*range(prefix_ndim), prefix_ndim, prefix_ndim + 2, prefix_ndim + 1, prefix_ndim + 3)
    tiled = tiled.contiguous().view(*bytes_view.shape[:-2], n_tiles, k_tiles, 128 * 64)

    starts = torch.tensor(chunk_starts, device=weight.device, dtype=torch.long)
    lo = tiled.index_select(-1, starts).to(torch.int64)
    hi = tiled.index_select(-1, starts + 1).to(torch.int64)
    chunks = (lo | (hi << 8)).view(*tiled.shape[:-1], 512, 8)

    def interleave(lo_src: torch.Tensor, hi_src: torch.Tensor) -> torch.Tensor:
        out = (lo_src & 0x0F) | ((hi_src & 0x0F) << 4)
        out = out | (((lo_src >> 4) & 0x0F) << 8) | (((hi_src >> 4) & 0x0F) << 12)
        out = out | (((lo_src >> 8) & 0x0F) << 16) | (((hi_src >> 8) & 0x0F) << 20)
        out = out | (((lo_src >> 12) & 0x0F) << 24) | (((hi_src >> 12) & 0x0F) << 28)
        return out

    regs = torch.stack([
        interleave(chunks[..., 0], chunks[..., 1]),
        interleave(chunks[..., 2], chunks[..., 3]),
        interleave(chunks[..., 4], chunks[..., 5]),
        interleave(chunks[..., 6], chunks[..., 7]),
    ], dim=-1)
    reg_bytes = torch.stack([(regs >> shift) & 0xFF for shift in (0, 8, 16, 24)], dim=-1).to(torch.uint8)
    slot_bytes = reg_bytes.reshape(*tiled.shape[:-1], 128 * 64)

    out = torch.empty_like(tiled)
    logical_index = torch.tensor(logical_positions, device=weight.device, dtype=torch.long)
    logical_index = logical_index.view(*([1] * (slot_bytes.ndim - 1)), 128 * 64).expand_as(slot_bytes)
    out.scatter_(-1, logical_index, slot_bytes)

    out = out.view(*bytes_view.shape[:-2], n_tiles, k_tiles, 128, 64)
    out = out.permute(*range(prefix_ndim), prefix_ndim, prefix_ndim + 2, prefix_ndim + 1, prefix_ndim + 3)
    return out.contiguous().view(*weight.shape[:-1], bytes_per_row).view(weight.dtype)


def prepare_tensorbridge_weight(
    weight: torch.Tensor,
    b_dtype: dtypes.DataType,
    a_dtype: dtypes.DataType,
    zero_point: torch.Tensor | None = None,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    use_fused_e4m3_scale: bool = False,
    use_nvfp4_snc: bool = False,
    packed: bool = False,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    use_nvfp4_raw_s2r_deint: bool = False,
    use_nvfp4_swizzle64_raw: bool = False,
) -> torch.Tensor:
    is_moe = weight.ndim == 3
    weight = weight.unsqueeze(0) if not is_moe else weight
    if zero_point is not None:
        zero_point = zero_point.unsqueeze(0) if zero_point.ndim == 2 else zero_point
    shape_n = weight.size(-2)
    if packed:
        assert weight.size(-1) * 32 % b_dtype.num_bits == 0
        shape_k = weight.size(-1) * 32 // b_dtype.num_bits
    else:
        shape_k = weight.size(-1)

    padded_shape_n = shape_n if padded_shape_n is None else padded_shape_n
    padded_shape_k = shape_k if padded_shape_k is None else padded_shape_k
    packed_block_size_k = 256 // a_dtype.num_bits

    assert padded_shape_n % 64 == 0
    assert padded_shape_k % (2 * packed_block_size_k) == 0

    should_preprocess_for_int2fp = False
    has_zero_point = zero_point is not None and zero_point.nelement() > 0
    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if a_dtype.num_bits < 16:
            should_preprocess_for_int2fp = True
        elif a_dtype == dtypes.bfloat16 and has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 6
        elif a_dtype == dtypes.bfloat16 and not has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 7

    if a_dtype == dtypes.int8 and b_dtype in [dtypes.int8, dtypes.uint8]:
        weight = (weight.view(torch.int8) - 128).view(torch.int32)

    if not should_preprocess_for_int2fp and has_zero_point:
        has_zero_point = False

    should_preprocess_with_zp = has_zero_point
    if zero_point is not None and zero_point.dtype.is_floating_point:
        should_preprocess_with_zp = False
        should_preprocess_for_int2fp = False

    if not has_zero_point:
        group_size_zp = 0
    else:
        assert zero_point is not None
        group_size_zp = shape_k // zero_point.size(-1)

    if use_nvfp4_swizzle64_raw:
        # CUTLASS-aligned 2D + 64B-swizzle TMA-B path: bypass `repack_weight`'s
        # 3D-box-friendly K-chunk permutation. Use the input weight as-is in
        # plain (..., N, K_int32) row-major int32 packed layout — the swizzle
        # TMA-B descriptor will read this directly and apply 64B XOR on SMEM.
        assert packed, "swizzle64_raw expects pre-packed int32 weight"
        assert b_dtype == dtypes.float4e2m1, "swizzle64_raw is validated for NVFP4 weights"
        assert a_dtype == dtypes.float8e4m3, "swizzle64_raw is validated for FP8 activations"
        assert not has_zero_point, "swizzle64_raw does not support zero points"
        if use_nvfp4_snc:
            weight = remap_nvfp4_e2m1_for_fpma(weight.contiguous())
        if os.environ.get("TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT", "0") == "1":
            weight = reorder_nvfp4_swizzle64_dual_mma_preinterleaved_raw_weight(weight)
        return weight if is_moe else weight.squeeze(0)

    repacked_weight = _ops().repack_weight(
        inputs=weight,
        zero_point=zero_point,
        weight_bits=b_dtype.num_bits,
        activation_bits=a_dtype.num_bits,
        is_weight_packed=packed,
        should_preprocess_for_int2fp=should_preprocess_for_int2fp,
        should_preprocess_with_zp=should_preprocess_with_zp,
        use_wgmma=use_wgmma,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        use_fused_e4m3_scale=use_fused_e4m3_scale,
        group_size_zp=group_size_zp,
    )

    if use_nvfp4_raw_s2r_deint:
        assert a_dtype == dtypes.float8e4m3
        assert b_dtype == dtypes.float4e2m1
        repacked_weight = deinterleave_nvfp4_raw_s2r_weight(
            repacked_weight,
            padded_shape_n,
        )

    if use_nvfp4_snc:
        assert use_fused_e4m3_scale
        assert a_dtype == dtypes.float8e4m3
        assert b_dtype == dtypes.float4e2m1
        repacked_weight = remap_nvfp4_e2m1_for_fpma(repacked_weight)

    return repacked_weight if is_moe else repacked_weight.squeeze(0)


def prepare_tensorbridge_weight_scale(
    weight_scale: torch.Tensor,
    to_apply_on_c: bool = False,
    is_blockwise: bool = False,
    use_nvfp4_swizzle64_raw: bool = False,
) -> torch.Tensor:
    if use_nvfp4_swizzle64_raw:
        # Swizzle64_raw path: transpose to (..., K_groups, N) so the existing
        # cp.async 2D loader (loader_bs.cuh) can copy BlockShape::N contiguous
        # bytes per row into SMEM. Skip the 8-stride N permutation that the
        # default path applies; the swizzle BS S2R formula reads scales by
        # natural N index. The kUseNvfp4TmaSwizzle64 S2R branch reads from
        # SMEM at byte offset = k_group * BlockShape::N + n_row.
        weight_scale = weight_scale.transpose(-1, -2).contiguous()
        return weight_scale

    if is_blockwise:
        return weight_scale.transpose(-1, -2).contiguous()

    if to_apply_on_c:
        perm = [0, 1, 8, 9, 16, 17, 24, 25]
    else:
        perm = [0, 8, 16, 24, 32, 40, 48, 56]

    count = sum(x < 8 for x in perm)
    perm_new = []
    for i in range(8 // count):
        perm_new += [x + count * i for x in perm]

    perm_tensor = torch.tensor(perm_new, dtype=torch.int32, device=weight_scale.device)
    weight_scale = weight_scale.transpose(-1, -2).contiguous()
    orig_shape = weight_scale.shape
    weight_scale = weight_scale.view(-1, len(perm_tensor))[:, perm_tensor]
    weight_scale = weight_scale.contiguous().view(orig_shape)

    return weight_scale


def prepare_tensorbridge_zero_point(
    zero_point: torch.Tensor,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor | None:
    if zero_point.dtype.is_floating_point:
        return prepare_tensorbridge_weight_scale(zero_point, False)

    if packed:
        zero_point = zero_point.transpose(-1, -2).contiguous()
        zero_point = zero_point.squeeze().view(*zero_point.shape)
        zero_point = _ops().unpack_weight(zero_point, dtype.num_bits)
        zero_point = zero_point.transpose(-1, -2).contiguous()

    assert zero_point is not None
    num_zp_bits = 4 if dtype.num_bits <= 4 else 8
    shape_n = zero_point.size(-2)
    zero_point = prepare_tensorbridge_weight_scale(zero_point)
    assert zero_point is not None
    zero_point = zero_point.to(torch.uint8)
    zero_point = zero_point.view(-1)
    if num_zp_bits == 4:
        zero_point = zero_point[..., 1::2] * 16 + zero_point[..., ::2]
    return zero_point.view(torch.int32).view(-1, shape_n * num_zp_bits // 32)


def prepare_tensorbridge_bias(bias: torch.Tensor) -> torch.Tensor:
    bias = prepare_tensorbridge_weight_scale(bias.unsqueeze(-1), True)
    assert bias is not None
    return bias.squeeze(-2)
