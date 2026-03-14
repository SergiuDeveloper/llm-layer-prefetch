import time
import torch
import itertools
import multiprocessing as mp
from typing import Any, Callable

from .layer_streamer import LayerStreamer

def _worker(
    result_queue: mp.Queue[tuple[str, int | None, float | str | None]],
    setup_fn: Callable[['LayerStreamer'], Callable[[], None]],
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
    n_runs: int,
    sleep_seconds: float,
) -> None:
    try:
        streamer = LayerStreamer(
            model_dir=model_dir,
            tensor_info=tensor_info,
            shard_offsets=shard_offsets,
            block_keys=block_keys,
            block_sizes=block_sizes,
            dtype=dtype,
            num_layers=num_layers,
            n_gpu=n_gpu,
            n_cpu_pinned=n_cpu_pinned,
            n_cpu=n_cpu,
            batch_gpu=batch_gpu,
            batch_cpu_pinned=batch_cpu_pinned,
            batch_cpu=batch_cpu,
            track_progress=False,
        )
        run_fn = setup_fn(streamer)

        for run_idx in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            result_queue.put(('run', run_idx, elapsed))
            if run_idx < n_runs - 1:
                time.sleep(sleep_seconds)
        result_queue.put(('done', None, None))
    except Exception as e:
        result_queue.put(('error', None, str(e)))

class LayerStreamerTuner:
    def __init__(self,
        setup_fn: Callable[['LayerStreamer'], Callable[[], None]],
        model_dir: str,
        tensor_info: dict[str, tuple[str, int, tuple[int, ...]]],
        shard_offsets: dict[str, int],
        block_keys: list[str],
        block_sizes: dict[str, int],
        dtype: torch.dtype,
        num_layers: int,
        n_gpu_values: list[int],
        n_cpu_pinned_values: list[int | None],
        n_cpu_values: list[int | None],
        batch_gpu_values: list[int | None],
        batch_cpu_pinned_values: list[int | None],
        batch_cpu_values: list[int | None],
        n_runs: int,
        sleep_seconds: float,
    ) -> None:
        self.setup_fn = setup_fn
        self.model_dir = model_dir
        self.tensor_info = tensor_info
        self.shard_offsets = shard_offsets
        self.block_keys = block_keys
        self.block_sizes = block_sizes
        self.dtype = dtype
        self.num_layers = num_layers
        self.n_gpu_values = n_gpu_values
        self.n_cpu_pinned_values = n_cpu_pinned_values
        self.n_cpu_values = n_cpu_values
        self.batch_gpu_values = batch_gpu_values
        self.batch_cpu_pinned_values = batch_cpu_pinned_values
        self.batch_cpu_values = batch_cpu_values
        self.n_runs = n_runs
        self.sleep_seconds = sleep_seconds

    def tune(self) -> dict[str, int | None]:
        best_time: float = float('inf')
        best_config: dict[str, int | None] = {}

        for n_gpu, n_cpu_pinned, n_cpu, batch_gpu, batch_cpu_pinned, batch_cpu in self._configs():
            config: dict[str, int | None] = {
                'n_gpu': n_gpu,
                'n_cpu_pinned': n_cpu_pinned,
                'n_cpu': n_cpu,
                'batch_gpu': batch_gpu,
                'batch_cpu_pinned': batch_cpu_pinned,
                'batch_cpu': batch_cpu,
            }
            avg_time = self._run_config(config, best_time)
            if avg_time is not None and avg_time < best_time:
                best_time = avg_time
                best_config = config
                print(f'new best {avg_time:.3f}s - {config}')

        return best_config

    def _configs(self) -> itertools.product[tuple[Any, ...]]:
        return itertools.product(
            self.n_gpu_values,
            self.n_cpu_pinned_values,
            self.n_cpu_values,
            self.batch_gpu_values,
            self.batch_cpu_pinned_values,
            self.batch_cpu_values,
        )

    def _run_config(self,
        config: dict[str, int | None],
        best_time: float,
    ) -> float | None:
        ctx = mp.get_context('spawn')
        result_queue: mp.Queue = ctx.Queue()
        process = ctx.Process(
            target=_worker,
            args=(
                result_queue,
                self.setup_fn,
                self.model_dir,
                self.tensor_info,
                self.shard_offsets,
                self.block_keys,
                self.block_sizes,
                self.dtype,
                self.num_layers,
                config['n_gpu'],
                config['n_cpu_pinned'],
                config['n_cpu'],
                config['batch_gpu'],
                config['batch_cpu_pinned'],
                config['batch_cpu'],
                self.n_runs,
                self.sleep_seconds,
            ),
            daemon=True,
        )
        process.start()

        timings: list[float] = []
        cold_timeout = self.sleep_seconds + 120.0
        warm_timeout = self.sleep_seconds + (best_time * 2.0 if best_time < float('inf') else 120.0)

        for expected_run in range(self.n_runs):
            timeout = cold_timeout if expected_run == 0 else warm_timeout
            try:
                kind, run_idx, value = result_queue.get(timeout=timeout)
            except Exception:
                process.terminate()
                process.join()
                return None

            if kind == 'error':
                process.join()
                return None

            if kind == 'done':
                break

            if kind == 'run':
                if run_idx == 0:
                    continue
                elapsed: float = value
                if best_time < float('inf') and elapsed > best_time:
                    process.terminate()
                    process.join()
                    return None
                timings.append(elapsed)

        try:
            result_queue.get(timeout=5.0)
        except Exception:
            pass

        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()

        if not timings:
            return None
        return sum(timings) / len(timings)
