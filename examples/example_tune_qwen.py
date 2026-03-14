import json, math, os, struct
import torch, torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from typing import Any, Callable

from layer_streamer import LayerStreamer, LayerStreamerTuner

def parse_header(path: str) -> tuple[dict[str, Any], int]:
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header, 8 + header_size

def _rotate_half(t: torch.Tensor) -> torch.Tensor:
    half = t.shape[-1] // 2
    return torch.cat([-t[..., half:], t[..., :half]], -1)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_slice = cos[:, :, past_length:past_length + q.shape[2]]
    sin_slice = sin[:, :, past_length:past_length + q.shape[2]]
    return (
        q * cos_slice + _rotate_half(q) * sin_slice,
        k * cos_slice + _rotate_half(k) * sin_slice
    )

def make_rope(
    length: int,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(length).float(), inv_freq)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos().to(dtype).to(device)[None, None], emb.sin().to(dtype).to(device)[None, None]

def forward_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    past_kv = kv_cache[layer_idx]
    past_length = past_kv[0].shape[2] if past_kv is not None else 0
    residual = x
    x = rms_norm(x, weights['input_layernorm.weight'], eps)
    q = F.linear(x, weights['self_attn.q_proj.weight'], weights.get('self_attn.q_proj.bias')).view(batch_size, seq_len, num_heads,    head_dim).transpose(1, 2)
    k = F.linear(x, weights['self_attn.k_proj.weight'], weights.get('self_attn.k_proj.bias')).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(x, weights['self_attn.v_proj.weight'], weights.get('self_attn.v_proj.bias')).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin, past_length)

    if past_kv is not None:
        k, v = torch.cat([past_kv[0], k], 2), torch.cat([past_kv[1], v], 2)
    kv_cache[layer_idx] = (k, v)
    kv_repeat = num_heads // num_kv_heads

    x = F.scaled_dot_product_attention(q, k.repeat_interleave(kv_repeat, 1), v.repeat_interleave(kv_repeat, 1), is_causal=(past_length == 0 and seq_len > 1))
    x = residual + F.linear(x.transpose(1, 2).reshape(batch_size, seq_len, -1), weights['self_attn.o_proj.weight'])
    residual = x
    x = rms_norm(x, weights['post_attention_layernorm.weight'], eps)
    gate = F.linear(x, weights['mlp.gate_proj.weight'])
    up = F.linear(x, weights['mlp.up_proj.weight'])
    return residual + F.linear(F.silu(gate) * up, weights['mlp.down_proj.weight'])

def qwen_setup(streamer: LayerStreamer) -> Callable[[], None]:
    model_dir = streamer.model_dir
    dtype = streamer.dtype

    config = json.load(open(f'{model_dir}/config.json'))
    num_layers = config['num_hidden_layers']
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    head_dim = config.get('head_dim', config['hidden_size'] // num_heads)
    eps = config['rms_norm_eps']
    rope_theta = config.get('rope_theta', 10000.)

    embed_weight = streamer.load_static('model.embed_tokens.weight')
    norm_weight = streamer.load_static('model.norm.weight')
    lm_head = streamer.load_static('lm_head.weight')

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = tokenizer.apply_chat_template(
        [{ 'role': 'user', 'content': 'Write a fibonacci function in Python.' }],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].cuda()
    cos, sin = make_rope(input_ids.shape[1] + 1, head_dim, rope_theta, dtype, 'cuda')
    input_embeds = embed_weight[input_ids[0]].unsqueeze(0)

    def run() -> None:
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        layer_fn = lambda x, w, i: forward_block(
            x=x, weights=w, kv_cache=kv_cache, layer_idx=i,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, eps=eps, cos=cos, sin=sin,
        )
        with torch.no_grad():
            hidden = streamer.run_pass(input_embeds, layer_fn)
            F.linear(rms_norm(hidden[:, -1], norm_weight, eps), lm_head)

    return run

if __name__ == '__main__':
    model_dir = './qwen2.5-coder-7b'
    if not os.path.exists(os.path.join(model_dir, 'model.safetensors.index.json')):
        snapshot_download(
            repo_id='Qwen/Qwen2.5-Coder-7B-Instruct',
            local_dir=model_dir,
            ignore_patterns=['*.bin', '*.pt', '*.gguf', 'original/'],
        )

    config     = json.load(open(f'{model_dir}/config.json'))
    weight_map = json.load(open(f'{model_dir}/model.safetensors.index.json'))['weight_map']
    num_layers = config['num_hidden_layers']

    shards: list[str] = sorted(set(weight_map.values()))
    shard_headers: dict[str, Any] = {}
    shard_offsets: dict[str, int] = {}
    for shard in shards:
        shard_headers[shard], shard_offsets[shard] = parse_header(f'{model_dir}/{shard}')
    tensor_info = {
        name: (shard, shard_headers[shard][name]['data_offsets'][0], tuple(shard_headers[shard][name]['shape']))
        for name, shard in weight_map.items()
    }
    layer0_prefix = 'model.layers.0.'
    block_keys = sorted(k[len(layer0_prefix):] for k in weight_map if k.startswith(layer0_prefix))
    block_sizes = {key: math.prod(tensor_info[layer0_prefix + key][2]) for key in block_keys}

    tuner = LayerStreamerTuner(
        setup_fn=qwen_setup,
        model_dir=model_dir,
        tensor_info=tensor_info,
        shard_offsets=shard_offsets,
        block_keys=block_keys,
        block_sizes=block_sizes,
        dtype=torch.bfloat16,
        num_layers=num_layers,
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
    print('optimal config:', best)
