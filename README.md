# llm-layer-prefetch

Stream transformer blocks one layer at a time from disk to GPU using a three-stage pipelined prefetch queue. Runs full LLM inference on a GPU with far less VRAM than the model's total weight size. Only one (or a few) layers need to be resident on the GPU at once.

## How it works

Each forward pass pipelines three concurrent stages:

```
disk  ──►  CPU RAM  ──►  pinned RAM  ──►  GPU
  (disk thread)  (pin thread)  (H2D thread)  (compute stream)
```

- **Disk thread** reads raw safetensors bytes directly into CPU buffers (zero-copy via `readinto`)
- **Pin thread** copies CPU buffers into pinned (page-locked) memory for fast DMA
- **H2D thread** issues non-blocking `copy_` on a dedicated CUDA transfer stream
- **Compute stream** waits on CUDA events from the H2D thread and runs the layer forward pass

Buffer pools at each stage allow the pipeline to stay full without extra allocation per layer. CUDA events synchronize the transfer and compute streams so a GPU slot is never overwritten while it's still being used.

## Installation

```bash
pip install llm-layer-prefetch
```

> **Note:** `torch` must already be installed with the CUDA variant of your choice before installing this package. See [pytorch.org](https://pytorch.org) for installation instructions.

## Usage

```python
from layer_streamer import LayerStreamer

streamer = LayerStreamer(
    model_dir='./my-model',
    tensor_info=tensor_info,       # dict: tensor_name → (shard_filename, byte_offset, shape)
    shard_offsets=shard_offsets,   # dict: shard_filename → data section byte offset
    block_keys=block_keys,         # list of per-layer tensor keys, e.g. ['self_attn.q_proj.weight', ...]
    block_sizes=block_sizes,       # dict: key → flat element count
    dtype=torch.bfloat16,
    num_layers=28,
    n_gpu=2,                       # number of layers to buffer on GPU ahead of compute
    n_cpu_pinned=4,                # number of layers to buffer in pinned RAM (None to skip stage)
    n_cpu=4,                       # number of layers to buffer in CPU RAM (None to skip stage)
    batch_gpu=1,                   # items per batch between H2D and compute (None = stream one by one)
    batch_cpu_pinned=1,
    batch_cpu=2,
    track_progress=True,           # show tqdm progress bar per forward pass
)

# Load tensors that stay on GPU for the whole session (embeddings, norm, lm_head, etc.)
embed = streamer.load_static('model.embed_tokens.weight')

# Run one full forward pass through all layers
hidden = streamer.run_pass(input_embeds, layer_fn)
```

### `run_pass`

```python
streamer.run_pass(
    x: torch.Tensor,
    layer_fn: Callable[[torch.Tensor, dict[str, torch.Tensor], int], torch.Tensor]
) -> torch.Tensor
```

`layer_fn(x, weights, layer_idx)` receives the current hidden state, a dict of weight tensors for that layer (already on GPU, already cast to `dtype`), and the layer index. It must return the new hidden state.

### Pipeline stages

| Parameter                               | Stage skipped when                                                  |
| --------------------------------------- | ------------------------------------------------------------------- |
| `n_cpu=None` or `n_cpu=0`               | No CPU RAM buffering; layers read directly by the pin or H2D thread |
| `n_cpu_pinned=None` or `n_cpu_pinned=0` | No pinned RAM buffering; H2D reads from CPU RAM or disk directly    |

All three stages can be disabled independently. With all disabled (`n_cpu=None`, `n_cpu_pinned=None`), layers are read from disk inline in the H2D thread with no prefetch.

## Tuning

The right values for `n_gpu`, `n_cpu_pinned`, `n_cpu`, and the `batch_*` params depend on your specific hardware (disk speed, RAM bandwidth, PCIe bandwidth, GPU). `LayerStreamerTuner` does a grid search to find the fastest combination automatically.

```python
from layer_streamer import LayerStreamerTuner

def my_setup(streamer):
    # load static weights, tokenize, etc.
    # return a zero-arg callable that runs one full forward pass
    def run():
        ...
    return run

tuner = LayerStreamerTuner(
    setup_fn=my_setup,
    model_dir='./my-model',
    tensor_info=tensor_info,
    shard_offsets=shard_offsets,
    block_keys=block_keys,
    block_sizes=block_sizes,
    dtype=torch.bfloat16,
    num_layers=28,
    n_gpu_values=[1, 2, 4],
    n_cpu_pinned_values=[None, 2, 4],
    n_cpu_values=[None, 2, 4],
    batch_gpu_values=[None, 1, 2],
    batch_cpu_pinned_values=[None, 1, 2],
    batch_cpu_values=[None, 1, 2],
    n_runs=5,
    sleep_seconds=3.0,
)

best = tuner.tune()
print(best)
# {'n_gpu': 2, 'n_cpu_pinned': 4, 'n_cpu': 4, 'batch_gpu': 1, ...}
```

Each configuration runs in a **separate process**, so a CUDA OOM or driver crash in one config doesn't affect the search. For each config:

- **1 warmup run** is performed and discarded (cold caches, JIT, etc.)
- **`n_runs - 1` measured runs** are averaged
- A `sleep_seconds` gap between runs lets the GPU and memory bus settle
- If any measured run exceeds the current best time, the process is **immediately killed** - no time wasted on obviously slow configs

`setup_fn(streamer)` must be a **top-level picklable function** (not a lambda or closure). It is called once inside the subprocess after the streamer is constructed. It should load static weights, prepare the input, and return a zero-arg `run()` callable. `run()` is called for each timing iteration and should reset any stateful objects (like KV cache) so every run is identical.

## Examples

Install example dependencies first:

```bash
pip install llm-layer-prefetch[examples]
```

### Inference - `examples/example_prefetch_qwen.py`

End-to-end autoregressive generation with Qwen2.5-Coder-7B-Instruct. Downloads the model automatically on first run (safetensors only). Implements tokenization, KV cache, RoPE, GQA attention, and SwiGLU MLP on top of `LayerStreamer`.

```bash
python examples/example_prefetch_qwen.py
```

Uses `n_gpu=2, n_cpu_pinned=4, n_cpu=4` by default. Only ~2 transformer layers resident on the GPU at once. A 7B bfloat16 model (~14 GB of weights) can run on a GPU with 4–6 GB of VRAM.

### Tuning - `examples/example_tune_qwen.py`

Runs the grid search for Qwen2.5-Coder-7B-Instruct and prints the optimal config. Each config is benchmarked by running a single full prefill forward pass (one token), which is the most representative single-step cost.

```bash
python examples/example_tune_qwen.py
```

Prints a line each time a new best is found, and outputs the final optimal config at the end.
