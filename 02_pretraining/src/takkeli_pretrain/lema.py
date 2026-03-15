"""LEMA: Layer-wise Efficient Memory Abstraction for weight streaming.

Implements the LEMA framework's triple-buffer weight streaming protocol
for VRAM-constrained training. LEMA uses three buffer slots per transformer
layer to overlap data transfer with computation:

1. **Active buffer**: Weights currently being used for forward/backward pass
2. **Prefetch buffer**: Next layer's weights being loaded asynchronously
3. **Offload buffer**: Previously used weights being offloaded to CPU

This enables training of models that exceed VRAM capacity by streaming
layer weights between CPU RAM and GPU VRAM on demand.

Reference: GitHub:Pomilon/LEMA-llama — "A Proof of Concept for the LEMA
Framework for Efficient Memory Abstraction in LLM Fine-Tuning"
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Buffer Slot
# ---------------------------------------------------------------------------


@dataclass
class BufferSlot:
    """A single buffer slot in the triple-buffer scheme.

    Attributes:
        layer_idx: Which layer this buffer holds weights for.
        data: The weight data (parameter state dict fragment).
        device: Device where the data currently resides.
        is_ready: Whether the data has been loaded and is available.
    """

    layer_idx: int = -1
    data: dict[str, torch.Tensor] = field(default_factory=dict)
    device: torch.device = torch.device("cpu")
    is_ready: bool = False

    def clear(self) -> None:
        """Clear the buffer and free memory."""
        self.data = {}
        self.layer_idx = -1
        self.is_ready = False

    def load_weights(
        self,
        layer_idx: int,
        weight_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Load weights into this buffer.

        Args:
            layer_idx: Layer index these weights belong to.
            weight_dict: Dictionary of parameter name to tensor.
            device: Target device for the weights.
        """
        self.clear()
        self.layer_idx = layer_idx
        self.device = device
        self.data = {name: param.detach().to(device) for name, param in weight_dict.items()}
        self.is_ready = True

    def transfer_to(self, target_device: torch.device) -> None:
        """Transfer buffer contents to a different device.

        Args:
            target_device: Target device to move tensors to.
        """
        self.data = {name: param.detach().to(target_device) for name, param in self.data.items()}
        self.device = target_device


# ---------------------------------------------------------------------------
# LEMA Configuration
# ---------------------------------------------------------------------------


@dataclass
class LEMAConfig:
    """Configuration for LEMA weight streaming.

    Attributes:
        num_layers: Total number of transformer layers to stream.
        compute_device: Device where computation happens (e.g., 'cuda').
        storage_device: Device where inactive weights are stored (e.g., 'cpu').
        num_buffers: Number of buffer slots. Default 3 (triple-buffer).
        num_prefetch_threads: Number of threads for async prefetching. Default 1.
    """

    num_layers: int = 24
    compute_device: str | torch.device = "cpu"
    storage_device: str | torch.device = "cpu"
    num_buffers: int = 3
    num_prefetch_threads: int = 1


# ---------------------------------------------------------------------------
# Layer Weight Extractor
# ---------------------------------------------------------------------------


def get_layer_params(
    model: nn.Module,
    layer_idx: int,
    layer_prefix: str = "blocks",
) -> dict[str, torch.Tensor]:
    """Extract parameters for a specific transformer layer.

    Args:
        model: The full model.
        layer_idx: Index of the layer to extract.
        layer_prefix: Prefix for layer modules (e.g., 'blocks' for
            model.blocks[i]).

    Returns:
        Dictionary mapping parameter names to tensors for the given layer.
    """
    layer = getattr(model, layer_prefix)[layer_idx]
    return {name: param.detach().clone() for name, param in layer.named_parameters()}


def set_layer_params(
    model: nn.Module,
    layer_idx: int,
    weight_dict: dict[str, torch.Tensor],
    layer_prefix: str = "blocks",
) -> None:
    """Set parameters for a specific transformer layer from a weight dict.

    Args:
        model: The full model.
        layer_idx: Index of the layer to update.
        weight_dict: Dictionary of parameter name to tensor.
        layer_prefix: Prefix for layer modules.
    """
    layer = getattr(model, layer_prefix)[layer_idx]
    state = layer.state_dict()
    for name, tensor in weight_dict.items():
        # Match by parameter name suffix
        if name in state:
            state[name] = tensor
    layer.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Triple Buffer Weight Streamer
# ---------------------------------------------------------------------------


class TripleBufferStreamer:
    """Triple-buffer weight streaming manager for LEMA.

    Manages three buffer slots per layer group to overlap computation
    with data transfer. The triple-buffer scheme:

    - Buffer 0 (Active): Contains weights for the currently executing layer
    - Buffer 1 (Prefetch): Being filled with next layer's weights
    - Buffer 2 (Offload): Holding previous layer's weights (to be offloaded)

    In the CPU-only training scenario, the streaming is simulated since
    all data is already in RAM. On GPU, the buffers manage CPU↔GPU transfers.

    Args:
        config: LEMA configuration.
    """

    def __init__(self, config: LEMAConfig) -> None:
        self.config = config
        self.compute_device = torch.device(config.compute_device)
        self.storage_device = torch.device(config.storage_device)

        # Create buffer slots
        self.buffers: list[BufferSlot] = [
            BufferSlot(device=self.storage_device) for _ in range(config.num_buffers)
        ]

        # Thread pool for async prefetch
        self._executor = ThreadPoolExecutor(max_workers=config.num_prefetch_threads)
        self._prefetch_future: Future[Any] | None = None

        # Current state tracking
        self._active_buffer_idx: int = 0
        self._current_layer: int = -1
        self._is_initialized: bool = False

    @property
    def num_buffers(self) -> int:
        """Total number of buffer slots."""
        return len(self.buffers)

    def initialize(
        self,
        model: nn.Module,
        layer_prefix: str = "blocks",
    ) -> None:
        """Initialize all buffers by loading layer weights.

        Args:
            model: The model to extract weights from.
            layer_prefix: Prefix for layer modules.
        """
        self.buffers = [
            BufferSlot(device=self.storage_device) for _ in range(self.config.num_buffers)
        ]

        for i in range(self.config.num_layers):
            params = get_layer_params(model, i, layer_prefix)
            # For the triple-buffer, each layer gets its own 3 buffers
            # But in practice, we cycle through 3 global buffers
            # and assign them to layers as needed during streaming.

        # Initially load the first few layers
        for i in range(min(self.config.num_buffers, self.config.num_layers)):
            params = get_layer_params(model, i, layer_prefix)
            self.buffers[i].load_weights(i, params, self.compute_device)

        self._current_layer = 0
        self._active_buffer_idx = 0
        self._is_initialized = True

        # Start prefetching next layer if available
        if self.config.num_layers > self.config.num_buffers:
            next_layer = self.config.num_buffers
            next_params = get_layer_params(model, next_layer, layer_prefix)
            self._start_prefetch(next_layer, next_params, layer_prefix)

    def _start_prefetch(
        self,
        layer_idx: int,
        weight_dict: dict[str, torch.Tensor],
        layer_prefix: str = "blocks",
    ) -> None:
        """Start asynchronous prefetch of layer weights.

        Args:
            layer_idx: Layer to prefetch.
            weight_dict: Weight dictionary for the layer.
            layer_prefix: Prefix for layer modules.
        """
        if self._prefetch_future is not None:
            self._prefetch_future.result()  # Wait for previous prefetch

        def _prefetch_task() -> BufferSlot:
            """Prefetch task executed in thread pool."""
            # Find an available buffer (not the active one)
            available_idx = (self._active_buffer_idx + 1) % self.config.num_buffers
            buffer = self.buffers[available_idx]
            buffer.load_weights(layer_idx, weight_dict, self.compute_device)
            return buffer

        self._prefetch_future = self._executor.submit(_prefetch_task)

    def get_active_buffer(self) -> BufferSlot:
        """Get the currently active buffer.

        Returns:
            The buffer slot containing weights for the current layer.
        """
        return self.buffers[self._active_buffer_idx]

    def advance(self, model: nn.Module, layer_prefix: str = "blocks") -> None:
        """Advance to the next layer, rotating buffers.

        Args:
            model: The model (used for prefetch).
            layer_prefix: Prefix for layer modules.
        """
        if not self._is_initialized:
            return

        self._current_layer += 1

        if self._current_layer >= self.config.num_layers:
            return

        # Rotate active buffer index
        self._active_buffer_idx = (self._active_buffer_idx + 1) % self.config.num_buffers

        # Wait for any pending prefetch
        if self._prefetch_future is not None:
            self._prefetch_future.result()

        # Start prefetching the layer after the next one
        next_prefetch_layer = self._current_layer + self.config.num_buffers
        if next_prefetch_layer < self.config.num_layers:
            next_params = get_layer_params(model, next_prefetch_layer, layer_prefix)
            self._start_prefetch(next_prefetch_layer, next_params, layer_prefix)

    def wait_for_prefetch(self) -> None:
        """Block until the current prefetch operation completes."""
        if self._prefetch_future is not None:
            self._prefetch_future.result()
            self._prefetch_future = None

    def is_prefetch_ready(self) -> bool:
        """Check if the prefetch is complete without blocking.

        Returns:
            True if no prefetch is pending or the pending prefetch is done.
        """
        if self._prefetch_future is None:
            return True
        return self._prefetch_future.done()

    def shutdown(self) -> None:
        """Shut down the thread pool and release resources."""
        if self._prefetch_future is not None:
            self._prefetch_future.result()
            self._prefetch_future = None
        self._executor.shutdown(wait=True)
        # Clear all buffers
        for buf in self.buffers:
            buf.clear()


# ---------------------------------------------------------------------------
# LEMA Training Context Manager
# ---------------------------------------------------------------------------


class LEMATrainingContext:
    """Context manager for LEMA-augmented training.

    Integrates weight streaming with the training loop, managing
    buffer lifecycle across forward and backward passes.

    Args:
        config: LEMA configuration.
    """

    def __init__(self, config: LEMAConfig) -> None:
        self.config = config
        self.streamer = TripleBufferStreamer(config)
        self._model: nn.Module | None = None

    def setup(self, model: nn.Module) -> None:
        """Initialize the streaming context with a model.

        Args:
            model: The model to stream weights for.
        """
        self._model = model
        self.streamer.initialize(model)

    def pre_layer_forward(self, layer_idx: int) -> None:
        """Called before each layer's forward pass.

        Ensures the correct weights are loaded for the given layer.

        Args:
            layer_idx: Index of the layer about to execute.
        """
        # Wait for any pending prefetch to complete
        self.streamer.wait_for_prefetch()

    def post_layer_forward(self, layer_idx: int) -> None:
        """Called after each layer's forward pass.

        Initiates prefetch of the next layer's weights.

        Args:
            layer_idx: Index of the layer that just executed.
        """
        if self._model is not None and layer_idx + 1 < self.config.num_layers:
            next_layer = layer_idx + 1
            next_params = get_layer_params(self._model, next_layer)
            # Find an available buffer
            available_idx = (self.streamer._active_buffer_idx + 1) % self.config.num_buffers
            self.streamer.buffers[available_idx].load_weights(
                next_layer, next_params, self.streamer.compute_device
            )

    def cleanup(self) -> None:
        """Release all resources."""
        self.streamer.shutdown()
        self._model = None
