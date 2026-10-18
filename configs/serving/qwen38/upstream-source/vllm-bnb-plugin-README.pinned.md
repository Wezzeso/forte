# vllm-bnb-plugin

Out-of-tree [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
quantization plugin for vLLM.

## Install

```bash
uv pip install -e /path/to/vllm-bnb-plugin
```

The package registers itself through the `vllm.general_plugins` entry-point
group, so `quantization="bitsandbytes"` and `load_format="bitsandbytes"`
remain available after installation.

## What it provides

- in-flight 4-bit bitsandbytes quantization
- pre-quantized 4-bit and 8-bit bitsandbytes checkpoint loading
- bitsandbytes linear and MoE quantization methods

## Development

Install and enable the lint hooks:

```bash
uv pip install -r requirements/lint.txt
pre-commit install
```

Run all hooks:

```bash
pre-commit run --all-files
```
