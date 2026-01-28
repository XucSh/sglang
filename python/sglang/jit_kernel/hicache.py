from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

DEFAULT_BLOCK_QUOTA = 2


@lru_cache(maxsize=None)
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    num_threads, occupancy = 1024, 1
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        num_threads,
        occupancy,
    )
    return load_jit(
        "hicache",
        *args,
        cuda_files=["hicache.cuh"],
        cuda_wrappers=[
            ("launch_one", f"&HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"&HiCacheKernel<{args}>::run_all"),
            ("launch_one_pf_lf", f"&HiCacheKernel<{args}>::run_one_pf_lf"),
            ("launch_all_lf_pf", f"&HiCacheKernel<{args}>::run_all_lf_pf"),
        ],
    )


def can_use_hicache_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load JIT HiCache kernel: {e}")
        return False


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4

    if element_size <= 1024:
        return 2

    # fallback: no unroll
    return 1


def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_one_layer_pf_lf(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    layer_id: int,
    src_layout_dim: int,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """Transfer one layer from page_first (host) to layer_first (device).

    Args:
        k_cache_dst: Destination K cache for a single layer [token, head*dim]
        v_cache_dst: Destination V cache for a single layer [token, head*dim]
        indices_dst: Destination indices on GPU
        k_cache_src: Source K cache (page_first layout) [token, layer, head, dim]
        v_cache_src: Source V cache (page_first layout) [token, layer, head, dim]
        indices_src: Source indices on GPU
        layer_id: The layer to transfer
        src_layout_dim: Stride per token in bytes (layer_num * head * dim * dtype_size)
        element_dim: Size of one token's data for one layer (head * dim)
        unroll: Unroll factor for kernel
        block_quota: Max number of blocks to use
    """
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one_pf_lf(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
        src_layout_dim,
        layer_id,
    )


def transfer_hicache_all_layer_lf_pf(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    dst_layout_dim: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """Transfer all layers from layer_first (device) to page_first (host).

    Args:
        k_cache_dst: Destination K cache (page_first layout) [token, layer, head, dim]
        v_cache_dst: Destination V cache (page_first layout) [token, layer, head, dim]
        indices_dst: Destination indices on GPU
        k_ptr_src: Tensor of K cache pointers per layer (uint64)
        v_ptr_src: Tensor of V cache pointers per layer (uint64)
        indices_src: Source indices on GPU
        kv_cache_src_stride_bytes: Source stride per token in bytes
        dst_layout_dim: Destination stride per token in bytes (layer_num * element_size)
        element_size: Size of one token's data for one layer in bytes
        unroll: Unroll factor for kernel
        block_quota: Max number of blocks to use
    """
    if element_size is None:
        element_size = kv_cache_src_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    # Flatten destination tensors but keep element_dim for kernel
    element_dim = element_size // k_cache_dst.element_size()
    k_cache_dst_view = k_cache_dst.view(-1, element_dim)
    v_cache_dst_view = v_cache_dst.view(-1, element_dim)
    module.launch_all_lf_pf(
        k_cache_dst_view,
        v_cache_dst_view,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        dst_layout_dim,
    )
