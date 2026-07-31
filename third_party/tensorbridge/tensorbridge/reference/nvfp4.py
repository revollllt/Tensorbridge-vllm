"""Reference construction for ModelOpt NVFP4 and the TensorBridge FPMA bridge."""

from __future__ import annotations

import torch


FPMA_PREFOLD_DELTA = 0x1C
FPMA_ANALYTIC_ALPHA_V1_SCALE_MIN = 0x39
FPMA_ANALYTIC_ALPHA_V1_SCALE_MAX = 0x7E
# First-order debias: 1 - (5/16 eligible cases) * (1/8 E4M3 binade ULP).
FPMA_ANALYTIC_ALPHA_V1_UNROUNDED = 1.0 - (5.0 / 16.0) * (1.0 / 8.0)
FPMA_ANALYTIC_ALPHA_V1 = round(FPMA_ANALYTIC_ALPHA_V1_UNROUNDED, 3)


def default_fpma_alpha(
    *,
    prefold_selector: str = "none",
    ulp_correction: bool = False,
) -> float:
    """Choose the forward alpha for plain FPMA while preserving legacy alternatives."""
    if prefold_selector not in {"none", "normal_b8_sse"}:
        raise ValueError(f"unknown NVFP4 prefold selector: {prefold_selector!r}")
    if prefold_selector != "none" or ulp_correction:
        return 1.0
    return FPMA_ANALYTIC_ALPHA_V1


def validate_analytic_fpma_scale_domain(weight_scale_e4m3: torch.Tensor) -> None:
    """Require the E4M3 scale domain used to derive and evaluate analytic_v1."""
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn:
        raise TypeError("weight_scale_e4m3 must have dtype torch.float8_e4m3fn")
    raw = weight_scale_e4m3.contiguous().view(torch.uint8)
    invalid = (raw != 0) & (
        (raw < FPMA_ANALYTIC_ALPHA_V1_SCALE_MIN)
        | (raw > FPMA_ANALYTIC_ALPHA_V1_SCALE_MAX)
    )
    if invalid.any().item():
        first = int(raw[invalid][0].item())
        raise ValueError(
            "FPMA analytic_v1 requires nonzero raw E4M3 scales in "
            f"[0x{FPMA_ANALYTIC_ALPHA_V1_SCALE_MIN:02x}, "
            f"0x{FPMA_ANALYTIC_ALPHA_V1_SCALE_MAX:02x}]; found 0x{first:02x}. "
            "Set TENSORBRIDGE_NVFP4_FPMA_ALPHA explicitly for an out-of-domain "
            "checkpoint."
        )


def unpack_nvfp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """Unpack ModelOpt's low-nibble-first E2M1 bytes into uint8 codes."""
    if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
        raise TypeError("weight_packed must be a 2D uint8 tensor")
    codes = torch.empty(
        (weight_packed.shape[0], weight_packed.shape[1] * 2),
        dtype=torch.uint8,
        device=weight_packed.device,
    )
    codes[:, 0::2] = weight_packed & 0x0F
    codes[:, 1::2] = weight_packed >> 4
    return codes


def decode_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibbles to FP32 without materializing int64 LUT indices."""
    if codes.dtype != torch.uint8:
        raise TypeError("codes must have dtype torch.uint8")
    magnitude = codes & 0x07
    values = magnitude.float() * 0.5
    values = torch.where(magnitude == 5, 3.0, values)
    values = torch.where(magnitude == 6, 4.0, values)
    values = torch.where(magnitude == 7, 6.0, values)
    return torch.where((codes & 0x08) != 0, -values, values)


def prefold_nvfp4_scale(weight_scale_e4m3: torch.Tensor) -> torch.Tensor:
    """Return the FPMA scale byte raw(s)-0x1c used by the device bridge."""
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn:
        raise TypeError("weight_scale_e4m3 must have dtype torch.float8_e4m3fn")
    raw = weight_scale_e4m3.contiguous().view(torch.uint8).to(torch.int16)
    invalid = (raw != 0) & ((raw < 0x1C) | (raw > 0x7E))
    if invalid.any().item():
        first = int(raw[invalid][0].item())
        raise ValueError(f"cannot prefold out-of-domain NVFP4 scale byte 0x{first:02x}")
    return (raw - FPMA_PREFOLD_DELTA).clamp(min=0, max=255).to(torch.uint8)


def validate_nvfp4_scale_domain(
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    group_size: int = 16,
) -> None:
    """Reject scales outside FPMA's safe byte-add domain before any prefold clamp."""
    if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
        raise TypeError("weight_packed must be a 2D uint8 tensor")
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn or weight_scale_e4m3.ndim != 2:
        raise TypeError("weight_scale_e4m3 must be a 2D E4M3 tensor")
    if group_size % 2 != 0:
        raise ValueError("packed NVFP4 validation requires an even group_size")
    n, k_half = weight_packed.shape
    expected = (n, k_half * 2 // group_size)
    if k_half * 2 % group_size != 0 or weight_scale_e4m3.shape != expected:
        raise ValueError(f"expected weight_scale shape {expected}, got {weight_scale_e4m3.shape}")

    bytes_per_group = group_size // 2
    magnitude_nonzero = ((weight_packed & 0x07) != 0) | (((weight_packed >> 4) & 0x07) != 0)
    group_nonzero = magnitude_nonzero.view(n, -1, bytes_per_group).any(dim=-1)
    raw = weight_scale_e4m3.contiguous().view(torch.uint8)
    legal_nonzero = (raw >= 0x1C) & (raw <= 0x7E)
    legal_zero_group = (raw == 0) & ~group_nonzero
    invalid = ~(legal_nonzero | legal_zero_group)
    if invalid.any().item():
        first = int(raw[invalid][0].item())
        count = int(invalid.sum().item())
        raise ValueError(
            "NVFP4 FPMA requires raw E4M3 scales in [0x1c, 0x7e], except raw 0 "
            f"for all-zero groups; found 0x{first:02x} in {count} group(s)"
        )


def normal_nvfp4_fp8(
    codes: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Construct B8=E4M3(BF16(q*s/6)); the FP32 6*g scale stays outside."""
    if codes.ndim != 2 or weight_scale_e4m3.ndim != 2:
        raise ValueError("codes and weight_scale_e4m3 must both be 2D")
    if codes.shape[-1] % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    expected = (codes.shape[0], codes.shape[1] // group_size)
    if weight_scale_e4m3.shape != expected:
        raise ValueError(f"expected weight_scale shape {expected}, got {weight_scale_e4m3.shape}")
    scales = weight_scale_e4m3.float().repeat_interleave(group_size, dim=-1)
    normalized = decode_e2m1_codes(codes) * scales / 6.0
    return normalized.to(torch.bfloat16).to(torch.float8_e4m3fn)


def build_normal_nvfp4_fp8_weight(
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    *,
    group_size: int = 16,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Expand a packed NVFP4 matrix into the normal E4M3 W8A8 operand."""
    if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
        raise TypeError("weight_packed must be a 2D uint8 tensor")
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn or weight_scale_e4m3.ndim != 2:
        raise TypeError("weight_scale_e4m3 must be a 2D E4M3 tensor")
    if weight_packed.device != weight_scale_e4m3.device:
        raise ValueError("weight and weight_scale must be on the same device")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    validate_nvfp4_scale_domain(weight_packed, weight_scale_e4m3, group_size)
    n, k_half = weight_packed.shape
    k = k_half * 2
    output = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight_packed.device)
    for row_start in range(0, n, chunk_rows):
        row_end = min(row_start + chunk_rows, n)
        codes = unpack_nvfp4_weight(weight_packed[row_start:row_end])
        converted = normal_nvfp4_fp8(
            codes,
            weight_scale_e4m3[row_start:row_end],
            group_size,
        )
        invalid = (converted.view(torch.uint8) & 0x7F) == 0x7F
        if invalid.any().item():
            raise ValueError(
                "normal NVFP4-to-E4M3 conversion produced "
                f"{int(invalid.sum().item())} non-finite byte(s)"
            )
        output[row_start:row_end] = converted
    return output


def fpma_snc_fp8_bytes(
    codes: torch.Tensor,
    prefolded_scale: torch.Tensor,
    group_size: int = 16,
    *,
    exact_ulp_correction: bool = False,
) -> torch.Tensor:
    """Reproduce the production SNC remap, mask, and byte-wise FPMA add."""
    if codes.dtype != torch.uint8 or prefolded_scale.dtype != torch.uint8:
        raise TypeError("codes and prefolded_scale must have dtype torch.uint8")
    if codes.ndim != 2 or prefolded_scale.ndim != 2:
        raise ValueError("codes and prefolded_scale must both be 2D")
    expected = (codes.shape[0], codes.shape[1] // group_size)
    if codes.shape[-1] % group_size != 0 or prefolded_scale.shape != expected:
        raise ValueError(f"expected prefolded_scale shape {expected}, got {prefolded_scale.shape}")

    original_mag = codes & 0x07
    original_sign = codes & 0x08
    remapped_mag = torch.where(
        original_mag == 0,
        torch.ones_like(original_mag),
        torch.where(original_mag == 1, torch.zeros_like(original_mag), original_mag),
    )
    # Offline SNC drops the sign on E2M1 +/-0, but preserves it on +/-0.5.
    remapped_sign = torch.where(
        original_mag == 0,
        torch.zeros_like(original_sign),
        original_sign,
    )
    addend = (remapped_sign.to(torch.int16) << 4) | (remapped_mag.to(torch.int16) << 2)
    prefolded = prefolded_scale.to(torch.int16).repeat_interleave(group_size, dim=-1)
    summed = prefolded + addend
    if summed.numel() and int(summed.max().item()) > 0xFF:
        raise ValueError("FPMA byte add overflowed; the NVFP4 scale domain is invalid")

    if exact_ulp_correction:
        invalid = (prefolded_scale != 0) & (prefolded_scale < 0x1D)
        if invalid.any().item():
            raise ValueError("exact FPMA ULP correction requires raw E4M3 scales >= 0x39")
        scale_residue = prefolded_scale.repeat_interleave(group_size, dim=-1) & 0x07
        scale_eligible = (scale_residue >= 2) & (scale_residue <= 6)
        magnitude_eligible = (remapped_mag & 0x01) == 0
        summed = summed - (scale_eligible & magnitude_eligible).to(torch.int16)

    # The SNC mask treats remapped magnitude 1 as original FP4 zero, while
    # remapped magnitude 0 (original +/-0.5) remains a real value.
    return torch.where(original_mag == 0, 0, summed).to(torch.uint8)


def build_nvfp4_reference_weights(
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    group_size: int = 16,
    chunk_rows: int = 256,
    prefold_selector: str = "none",
    exact_ulp_correction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build W4A16-exact, normal B8, SNC-FPMA B8, and the external 6*g scale."""
    if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
        raise TypeError("weight_packed must be a 2D uint8 tensor")
    if weight_scale_e4m3.dtype != torch.float8_e4m3fn or weight_scale_e4m3.ndim != 2:
        raise TypeError("weight_scale_e4m3 must be a 2D E4M3 tensor")
    if weight_packed.device != weight_scale_e4m3.device:
        raise ValueError("weight and weight_scale must be on the same device")
    if global_scale.numel() != 1:
        raise ValueError("ModelOpt NVFP4 weight_scale_2 must contain one scalar")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    validate_nvfp4_scale_domain(weight_packed, weight_scale_e4m3, group_size)
    if exact_ulp_correction:
        raw_scale = weight_scale_e4m3.contiguous().view(torch.uint8)
        invalid = (raw_scale != 0) & (raw_scale < 0x39)
        if invalid.any().item():
            raise ValueError("exact FPMA ULP correction requires raw E4M3 scales >= 0x39")
    if prefold_selector == "none":
        selected_prefold = None
    elif prefold_selector == "normal_b8_sse":
        from tensorbridge.utils.weight import select_nvfp4_prefold_scale

        selected_prefold = select_nvfp4_prefold_scale(
            weight_packed,
            weight_scale_e4m3,
            group_size=group_size,
            chunk_rows=chunk_rows,
        )
    else:
        raise ValueError(f"unknown NVFP4 prefold selector: {prefold_selector!r}")

    n, k_half = weight_packed.shape
    k = k_half * 2
    expected_scale_shape = (n, k // group_size)
    if k % group_size != 0 or weight_scale_e4m3.shape != expected_scale_shape:
        raise ValueError(
            f"expected weight_scale shape {expected_scale_shape}, got {weight_scale_e4m3.shape}"
        )

    exact_bf16 = torch.empty((n, k), dtype=torch.bfloat16, device=weight_packed.device)
    normal_fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight_packed.device)
    fpma_snc_fp8 = torch.empty_like(normal_fp8)
    g = global_scale.float().reshape(())

    for row_start in range(0, n, chunk_rows):
        row_end = min(row_start + chunk_rows, n)
        packed_chunk = weight_packed[row_start:row_end]
        scale_chunk = weight_scale_e4m3[row_start:row_end]
        codes = unpack_nvfp4_weight(packed_chunk)
        q = decode_e2m1_codes(codes)
        scale_fp32 = scale_chunk.float().repeat_interleave(group_size, dim=-1)
        exact_bf16[row_start:row_end] = (q * scale_fp32 * g).to(torch.bfloat16)
        normal_fp8[row_start:row_end] = (q * scale_fp32 / 6.0).to(torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        prefolded = (
            prefold_nvfp4_scale(scale_chunk)
            if selected_prefold is None
            else selected_prefold[row_start:row_end]
        )
        fpma_bytes = fpma_snc_fp8_bytes(
            codes,
            prefolded,
            group_size,
            exact_ulp_correction=exact_ulp_correction,
        )
        fpma_snc_fp8[row_start:row_end] = fpma_bytes.view(torch.float8_e4m3fn)

    return exact_bf16, normal_fp8, fpma_snc_fp8, (global_scale.float().reshape(1) * 6.0)
