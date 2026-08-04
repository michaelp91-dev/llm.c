# RoPE wiring for train_gpt2.cu

Kernel support is already on master:
- `llmc/rope.cuh` — precompute + forward/backward
- `llmc/encoder.cuh` — `wpe == NULL` means token-only embeddings
- `llmc/attention.cuh` — `attention_forward` / `attention_backward` take `rope_cos`, `rope_sin`

`train_gpt2.cu` still needs the call-site updates below (file is large; apply these edits).

## 1. Include RoPE

After the attention include block, ensure rope is available (non-cuDNN path already pulls it via attention.cuh).

## 2. Add RoPE tables to GPT2 struct

Inside `typedef struct { ... } GPT2;`, add:

```c
    // RoPE tables (NULL = legacy absolute positions via wpe)
    float* rope_cos;  // (maxT, HD/2)
    float* rope_sin;  // (maxT, HD/2)
    int use_rope;     // 1 = RoPE on, 0 = classic wpe
```

In `gpt2_init_common`:

```c
    model->rope_cos = NULL;
    model->rope_sin = NULL;
    model->use_rope = 1; // default ON for modern 124M
```

## 3. Allocate + precompute in `gpt2_allocate_state`

At the end of `gpt2_allocate_state`, after other allocations:

```c
    // RoPE cos/sin tables
    if (model->use_rope) {
        int HD = model->config.channels / model->config.num_heads;
        int half = HD / 2;
        int maxT = model->config.max_seq_len;
        size_t table_elems = (size_t)maxT * half;
        float* cos_cpu = (float*)mallocCheck(table_elems * sizeof(float));
        float* sin_cpu = (float*)mallocCheck(table_elems * sizeof(float));
        rope_precompute_cpu(cos_cpu, sin_cpu, maxT, HD, 10000.0f);
        cudaCheck(cudaMalloc((void**)&model->rope_cos, table_elems * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->rope_sin, table_elems * sizeof(float)));
        cudaCheck(cudaMemcpy(model->rope_cos, cos_cpu, table_elems * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(model->rope_sin, sin_cpu, table_elems * sizeof(float), cudaMemcpyHostToDevice));
        free(cos_cpu);
        free(sin_cpu);
        printf0("RoPE enabled: theta=10000, head_dim=%d, maxT=%d\n", HD, maxT);
    }
```

You need `#include "llmc/rope.cuh"` at the top of `train_gpt2.cu` for `rope_precompute_cpu`.

## 4. Forward: token-only encoder + RoPE attention

Replace:

```c
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream);
```

with:

```c
    encoder_forward(acts.encoded, model->inputs, params.wte,
                    model->use_rope ? NULL : params.wpe, B, T, C, main_stream);
```

Replace non-cuDNN attention call:

```c
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
```

with:

```c
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH,
                          model->use_rope ? model->rope_cos : NULL,
                          model->use_rope ? model->rope_sin : NULL,
                          main_stream);
```

## 5. Backward: matching signatures

Replace:

```c
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
```

with:

```c
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH,
                           model->use_rope ? model->rope_cos : NULL,
                           model->use_rope ? model->rope_sin : NULL,
                           main_stream);
```

Replace:

```c
    encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);
```

with:

```c
    encoder_backward(grads.wte, model->use_rope ? NULL : grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);
```

## 6. Free

In `gpt2_free`:

```c
    cudaFreeCheck(&model->rope_cos);
    cudaFreeCheck(&model->rope_sin);
```

## Notes

- `wpe` stays in the parameter layout for checkpoint compatibility but is unused when `use_rope=1`.
- cuDNN attention path is unchanged; use non-cuDNN build for RoPE (`make` without `ENABLE_CUDNN`).
- SwiGLU is still TODO (next phase).
