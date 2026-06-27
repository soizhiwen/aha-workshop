# AHa Workshop — Mitigating Hallucinations in Large Audio-Language Models

A hands-on tutorial showing how Large Audio-Language Models (LALMs) **hallucinate**
sounds that aren't actually present — e.g. reporting "thunder" just because a speaker
*says* the word — and how to mitigate it at decoding time with **AAD (Audio-Aware
Decoding)**.

Everything lives in [`tutorial.ipynb`](tutorial.ipynb). In it you will:

1. Run an open LALM — **Qwen2.5-Omni-3B** — on a handful of audio clips.
2. Watch it hallucinate in the **vanilla** (no-mitigation) setting.
3. Apply **AAD**, a training-free contrastive-decoding method, and write its core formula yourself.
4. Score the free-text answers with an **LLM-as-a-judge** (Qwen3.5-9B) and compare against ground truth.

## Requirements

- A CUDA GPU with **~18 GB** of memory (the notebook loads weights onto `cuda`).
- **Python 3.12** (pinned in [`.python-version`](.python-version)).
- A few GB of disk for the model weights and dataset, downloaded on first run from the Hugging Face Hub.

> **No GPU?** You can run the whole tutorial for free on
> [**Google Colab**](https://colab.research.google.com/drive/1fnZid_LGmpk-v_sjHyFGGlyavLXvN0DO?usp=sharing)
> (select a GPU runtime under *Runtime → Change runtime type*). The Colab is
> self-contained, so you can skip the local Setup steps below.
>
> Because Colab has limited GPU memory, the Colab version uses the smaller
> **Qwen3.5-4B** judge instead of Qwen3.5-9B. The results will therefore differ from,
> and be somewhat more degraded than, the local run.

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or `source ~/.bashrc`) so the `uv` command is on your `PATH`.
See the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/) for
other installation methods.

### 2. Sync the environment

From the repo root, this creates a `.venv` and installs every dependency pinned in
[`uv.lock`](uv.lock):

```bash
uv sync
```

### 3. Run the tutorial

Open [`tutorial.ipynb`](tutorial.ipynb) in your editor (VS Code, JupyterLab, etc.),
select the `.venv` created by `uv sync` as the notebook kernel, then read through and
run the cells one by one from top to bottom.

> **One thing you'll write yourself:** in the AAD cell, fill in the contrastive
> decoding formula where marked:
> `final = (1 + alpha) * positive - alpha * negative`.

## What gets downloaded automatically

When you run the notebook, it pulls everything it needs — no manual data download:

- **Dataset** — `soizhiwen/aha-workshop` from the Hugging Face Hub into
  `.cache/aha_workshop/` (6 questions over 5 short clips, schema-compatible with the
  full AHa-Bench).
- **Models** — `Qwen/Qwen2.5-Omni-3B` (the LALM) and `Qwen/Qwen3.5-9B` (the judge),
  cached locally after the first run.

## Project layout

| Path | Description |
|------|-------------|
| [`tutorial.ipynb`](tutorial.ipynb) | The full, self-contained tutorial. |
| [`pyproject.toml`](pyproject.toml) | Project metadata and dependencies. |
| [`uv.lock`](uv.lock) | Pinned, reproducible dependency versions. |
| `.cache/aha_workshop/` | Downloaded dataset (created on first run). |
