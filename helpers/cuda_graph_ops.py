"""CUDA-graph capture helpers for the optimization loop.

Several blocks of an EPO step are dominated by kernel-launch dispatch
rather than by GPU work: the pose-refinement MLP (7 linear layers over
``(N_img, 12)``), the whole geometry rebuild, and the per-step convergence
metric. Their shapes never change during a run, so each can be captured
once and replayed. A replay runs the identical kernels in the identical
order, so values and gradients stay bit-identical to the eager path.

Two capture styles are provided:

* :func:`capture_forward_backward` — for a differentiable ``nn.Module``,
  wrapping ``torch.cuda.make_graphed_callables`` (forward *and* backward).
* :class:`StaticGraph` — for a plain no-grad function of fixed-shape
  tensors, using a manual capture with static input/output buffers.

Four traps, each of which produced *silently wrong* results before being
found, are handled or documented here:

1. ``make_graphed_callables`` patches ``module.forward`` to a dispatcher
   that ignores the input shape, so a later call with a different batch
   silently gets the captured shape back. :func:`capture_forward_backward`
   returns that dispatcher and restores the eager ``forward``.
2. Modules that override ``parameters()`` (EPO's per-image modules return
   only their own leaf) hide parameters from the capture, and those get no
   gradient at all. Use :func:`leaf_parameters`.
3. The autocast weight cache must be off during capture, or BF16 copies of
   the weights are baked into the graph and never see an optimizer update.
   :func:`capture_forward_backward` passes ``cache_enabled=False``.
4. Outputs alias static buffers that the next replay overwrites. A caller
   that must hold a value across steps has to copy it.

Capture also fails if any backward has already run in the process (stale
AccumulateGrad nodes on the default stream), so capture before the first
optimization step.
"""

import contextlib

import torch
import torch.nn as nn


def leaf_parameters(modules) -> list[torch.Tensor]:
    """Collect the distinct trainable leaves of ``modules``, in order.

    Goes through ``nn.Module.parameters`` explicitly: EPO's per-image
    modules override ``parameters()`` to return only their own leaf tensor,
    which would hide nested parameters (the pose MLP, the translation
    offset) from a graph capture.

    Args:
        modules: Iterable of modules to collect from.

    Returns:
        The parameters with ``requires_grad=True``, de-duplicated by
        identity, in module order.
    """
    seen, out = set(), []
    for module in modules:
        for param in nn.Module.parameters(module):
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                out.append(param)
    return out


def capture_forward_backward(module: nn.Module, sample_args=(), autocast_dtype=None):
    """Capture ``module``'s forward and backward as CUDA graphs.

    Args:
        module: Module to capture. Its parameters must be the ones the
            optimizer updates (a graph reads the same memory, so in-place
            updates are picked up by later replays).
        sample_args: Example inputs, used only for their shape/dtype and
            ``requires_grad``. They are cloned, so the originals are left
            untouched. Empty when the module reads everything from its own
            parameters and buffers.
        autocast_dtype: Run the capture under ``torch.autocast`` with this
            dtype (weight cache off — see the module docstring). ``None``
            captures outside autocast.

    Returns:
        A callable with the same signature as ``module.forward`` that
        replays the graph. ``module.forward`` itself is left eager, so
        calls with other shapes still work.
    """
    samples = tuple(
        t.detach().clone().requires_grad_(t.requires_grad) for t in sample_args
    )
    scope = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype, cache_enabled=False)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with scope:
        eager_forward = module.forward
        torch.cuda.make_graphed_callables(module, samples)
        graphed = module.forward
        module.forward = eager_forward
    return graphed


class StaticGraph:
    """Replay a fixed-shape, gradient-free function from a CUDA graph.

    For blocks that are many tiny kernels behind a few hundred microseconds
    of dispatch. Inputs are copied into static buffers on every call, so
    the caller may pass any tensor of the captured shape.
    """

    def __init__(self, fn, *sample_inputs: torch.Tensor, warmup: int = 3):
        """Capture ``fn`` for the given input shapes.

        Args:
            fn: Callable taking exactly ``sample_inputs`` tensors and
                returning a tensor (or tuple of tensors). Must not sync or
                allocate outside the graph pool.
            *sample_inputs: Tensors giving the shape/dtype/device of each
                argument; their values are not used.
            warmup: Eager iterations run on a side stream before recording
                (lazy init and allocations must not happen inside a
                capture).
        """
        self._inputs = tuple(torch.empty_like(t) for t in sample_inputs)

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(warmup):
                fn(*self._inputs)
        torch.cuda.current_stream().wait_stream(warm)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._output = fn(*self._inputs)

    def __call__(self, *args: torch.Tensor):
        """Replay the graph on ``args`` and return the static output.

        The result aliases a buffer that the next call overwrites — consume
        or copy it before replaying again.
        """
        for static, arg in zip(self._inputs, args, strict=False):
            static.copy_(arg)
        self._graph.replay()
        return self._output
