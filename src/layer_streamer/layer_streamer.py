import math
import torch
import threading
import queue as Q
from typing import Any, Callable, Generator
from tqdm import tqdm

class LayerStreamer:
    _RAW_DTYPE_MAP = { 1: torch.uint8, 2: torch.int16, 4: torch.int32 }

    def __init__(self,
        model_dir: str,
        tensor_info: dict[str, tuple[str, int, tuple[int, ...]]],
        shard_offsets: dict[str, int],
        block_keys: list[str],
        block_sizes: dict[str, int],
        dtype: torch.dtype,
        num_layers: int,
        n_gpu: int,
        n_cpu_pinned: int | None,
        n_cpu: int | None,
        batch_gpu: int | None,
        batch_cpu_pinned: int | None,
        batch_cpu: int | None,
        track_progress: bool = True
    ) -> None:
        self.model_dir = model_dir
        self.tensor_info = tensor_info
        self.shard_offsets = shard_offsets
        self.block_keys = block_keys
        self.block_sizes = block_sizes
        self.dtype = dtype
        self.raw_dtype = self._RAW_DTYPE_MAP[torch.empty(1, dtype=dtype).element_size()]
        self.num_layers = num_layers
        self.n_cpu = n_cpu
        self.n_cpu_pinned = n_cpu_pinned
        self.batch_gpu = batch_gpu
        self.batch_cpu = batch_cpu
        self.batch_cpu_pinned = batch_cpu_pinned
        self.use_cpu = n_cpu is not None and n_cpu > 0
        self.use_pinned = n_cpu_pinned is not None and n_cpu_pinned > 0
        self.cpu_pool: Q.Queue = Q.Queue()
        self.track_progress = track_progress

        if self.use_cpu and n_cpu:
            for _ in range(n_cpu + 1):
                self.cpu_pool.put(self._alloc(False, None))

        self.pinned_pool: Q.Queue = Q.Queue()
        if self.use_pinned and n_cpu_pinned:
            for _ in range(n_cpu_pinned + 1):
                transfer_done = torch.cuda.Event()
                transfer_done.record()
                self.pinned_pool.put((self._alloc(True, None), transfer_done))
            torch.cuda.synchronize()

        num_gpu_slots = n_gpu + 1
        self.compute_done_events = [torch.cuda.Event() for _ in range(num_gpu_slots)]
        for event in self.compute_done_events: event.record()
        torch.cuda.synchronize()

        self.gpu_pool: Q.Queue = Q.Queue()
        for slot_idx in range(num_gpu_slots):
            self.gpu_pool.put((slot_idx, self._alloc(False, 'cuda')))

        self.transfer_stream = torch.cuda.Stream()
        self.compute_stream  = torch.cuda.Stream()
        self.done = object()
        self._cpu_queue: Q.Queue | None = None
        self._pinned_queue: Q.Queue | None = None
        self._gpu_queue: Q.Queue = Q.Queue()
        self._h2d_events: list[torch.cuda.Event] = []

    def run_pass(self,
        x: torch.Tensor,
        layer_fn: Callable[[torch.Tensor, dict[str, torch.Tensor], int], torch.Tensor]
    ) -> torch.Tensor:
        self._cpu_queue = Q.Queue() if self.use_cpu else None
        self._pinned_queue = Q.Queue() if self.use_pinned else None
        self._gpu_queue = Q.Queue()
        self._h2d_events = [torch.cuda.Event() for _ in range(self.num_layers)]

        threads: list[threading.Thread] = []
        if self.use_cpu:
            threads.append(threading.Thread(target=self._stage_disk, daemon=True))
        if self.use_pinned:
            threads.append(threading.Thread(target=self._stage_pin,  daemon=True))
        threads.append(threading.Thread(target=self._stage_h2d, daemon=True))

        for thread in threads:
            thread.start()

        self.compute_stream.wait_stream(torch.cuda.default_stream())
        layer_iterator = self._drain(self._gpu_queue, self.done)
        if self.track_progress:
            layer_iterator = tqdm(layer_iterator, total=self.num_layers, leave=False)
        for layer_idx, slot_idx, gpu_slot in layer_iterator:
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(self._h2d_events[layer_idx])
                x = layer_fn(x, self._as_weights(gpu_slot, layer_idx), layer_idx)
                self.compute_done_events[slot_idx].record()
            self.gpu_pool.put((slot_idx, gpu_slot))

        torch.cuda.current_stream().wait_stream(self.compute_stream)
        for thread in threads:
            thread.join()
        return x

    def _alloc(self, pin: bool, device: str | None) -> dict[str, torch.Tensor]:
        if pin:
            return { key: torch.empty(self.block_sizes[key], dtype=self.raw_dtype).pin_memory() for key in self.block_keys }
        return { key: torch.empty(self.block_sizes[key], dtype=self.raw_dtype, device=device) for key in self.block_keys }

    def _read_block(self, layer_idx: int, slot: dict[str, torch.Tensor]) -> None:
        prefix = f'model.layers.{layer_idx}.'
        for key, buffer in slot.items():
            shard, offset, _ = self.tensor_info[prefix + key]
            with open(f'{self.model_dir}/{shard}', 'rb') as f:
                f.seek(self.shard_offsets[shard] + offset)
                f.readinto(buffer.numpy())

    def _as_weights(self, gpu_slot: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f'model.layers.{layer_idx}.'
        return { key: gpu_slot[key].view(self.dtype).view(self.tensor_info[prefix + key][2]) for key in self.block_keys }

    def load_static(self, name: str) -> torch.Tensor:
        shard, offset, shape = self.tensor_info[name]
        buffer = torch.empty(math.prod(shape), dtype=self.raw_dtype)
        with open(f'{self.model_dir}/{shard}', 'rb') as f:
            f.seek(self.shard_offsets[shard] + offset)
            f.readinto(buffer.numpy())
        return buffer.view(self.dtype).view(shape).cuda()

    @staticmethod
    def _drain(queue: Q.Queue, done: object) -> Generator[Any, None, None]:
        while True:
            item = queue.get()
            if item is done:
                return
            if isinstance(item, list):
                yield from item
            else:
                yield item

    @staticmethod
    def _emit(
        item: Any,
        queue: Q.Queue,
        accumulator: list[Any],
        batch: int | None
    ) -> list[Any]:
        if batch is None:
            queue.put(item)
            return accumulator
        accumulator.append(item)

        if len(accumulator) >= batch:
            queue.put(list(accumulator))
            accumulator.clear()
        return accumulator

    @staticmethod
    def _flush(accumulator: list[Any], queue: Q.Queue) -> None:
        if accumulator:
            queue.put(list(accumulator))

    def _stage_disk(self) -> None:
        if self._cpu_queue is None:
            return
        
        accumulator: list[Any] = []
        for layer_idx in range(self.num_layers):
            cpu_slot = self.cpu_pool.get()
            self._read_block(layer_idx, cpu_slot)
            accumulator = self._emit((layer_idx, cpu_slot), self._cpu_queue, accumulator, self.batch_cpu)

        self._flush(accumulator, self._cpu_queue)
        self._cpu_queue.put(self.done)

    def _stage_pin(self) -> None:
        if self._pinned_queue is None:
            return
        
        accumulator: list[Any] = []
        if self._cpu_queue is not None:
            for layer_idx, cpu_slot in self._drain(self._cpu_queue, self.done):
                pinned_slot, transfer_done = self.pinned_pool.get()
                transfer_done.synchronize()
                for key in self.block_keys:
                    pinned_slot[key].copy_(cpu_slot[key])
                self.cpu_pool.put(cpu_slot)
                accumulator = self._emit((layer_idx, pinned_slot, transfer_done), self._pinned_queue, accumulator, self.batch_cpu_pinned)
        else:
            for layer_idx in range(self.num_layers):
                cpu_slot = self._alloc(False, None)
                self._read_block(layer_idx, cpu_slot)
                pinned_slot, transfer_done = self.pinned_pool.get()
                transfer_done.synchronize()
                for key in self.block_keys:
                    pinned_slot[key].copy_(cpu_slot[key])
                accumulator = self._emit((layer_idx, pinned_slot, transfer_done), self._pinned_queue, accumulator, self.batch_cpu_pinned)

        self._flush(accumulator, self._pinned_queue)
        self._pinned_queue.put(self.done)

    def _stage_h2d(self) -> None:
        accumulator: list[Any] = []
        if self._pinned_queue is not None:
            for layer_idx, src_slot, transfer_done in self._drain(self._pinned_queue, self.done):
                slot_idx, gpu_slot = self.gpu_pool.get()
                with torch.cuda.stream(self.transfer_stream):
                    self.transfer_stream.wait_event(self.compute_done_events[slot_idx])
                    for key in self.block_keys:
                        gpu_slot[key].copy_(src_slot[key], non_blocking=True)
                    transfer_done.record()
                    self._h2d_events[layer_idx].record()

                self.pinned_pool.put((src_slot, transfer_done))
                accumulator = self._emit((layer_idx, slot_idx, gpu_slot), self._gpu_queue, accumulator, self.batch_gpu)
        elif self._cpu_queue is not None:
            for layer_idx, src_slot in self._drain(self._cpu_queue, self.done):
                slot_idx, gpu_slot = self.gpu_pool.get()
                with torch.cuda.stream(self.transfer_stream):
                    self.transfer_stream.wait_event(self.compute_done_events[slot_idx])
                    for key in self.block_keys:
                        gpu_slot[key].copy_(src_slot[key], non_blocking=True)
                    self._h2d_events[layer_idx].record()

                self._h2d_events[layer_idx].synchronize()
                self.cpu_pool.put(src_slot)
                accumulator = self._emit((layer_idx, slot_idx, gpu_slot), self._gpu_queue, accumulator, self.batch_gpu)
        else:
            for layer_idx in range(self.num_layers):
                src_slot = self._alloc(False, None)
                self._read_block(layer_idx, src_slot)
                slot_idx, gpu_slot = self.gpu_pool.get()
                with torch.cuda.stream(self.transfer_stream):
                    self.transfer_stream.wait_event(self.compute_done_events[slot_idx])
                    for key in self.block_keys:
                        gpu_slot[key].copy_(src_slot[key], non_blocking=True)
                    self._h2d_events[layer_idx].record()

                accumulator = self._emit((layer_idx, slot_idx, gpu_slot), self._gpu_queue, accumulator, self.batch_gpu)
        self._flush(accumulator, self._gpu_queue)
        self._gpu_queue.put(self.done)
