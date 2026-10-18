# Integration decisions

| Date (Asia/Qyzylorda) | Decision | Evidence / scope |
|---|---|---|
| 2026-10-17 | Pin current stable direct Python dependencies and commit the uv lock. | Public PyPI JSON checked on the same day; `dependency-versions.md`. Installation validates this CPU environment only. |
| 2026-10-17 | Keep CPU API/worker dependencies separate from GPU training and vLLM. | CPU tests use Python 3.14; the optional training extra targets Linux/Python 3.12. No CUDA claim follows from dependency resolution. |
| 2026-10-17 | Use Qwen3-VL-8B-Instruct baseline until real validation justifies an adapter. | Fixed revision is supplied by A07; no measured adapter improvement yet. |
| 2026-10-17 | A17 has exclusive shared-PC GUI input; A19 uses public fetch/search. | `pc-control.md`; observed target and capability remain A17's handoff. |
| 2026-10-17 | Synthetic integration fixtures cannot satisfy real quality or public deployment gates. | Acceptance rows remain UNMEASURED until the specified proof exists. |
| 2026-10-17 | Match the current stable vLLM release's CUDA variant to the actual host driver. | Initial CUDA 13 tensor check failed on driver 565.57.01. The same vLLM 0.31.0 release's CUDA 12.9 variant passed an actual BF16 matrix operation at 15:19:28+05 (Torch 2.13.0+cu129); model warmup/inference remain separate gates. |
| 2026-10-17 | Align the optional training lock to the actual compatible CUDA 12.9 stack. | Explicit official PyTorch index pins Torch 2.13.0+cu129 / vision 0.28.0+cu129; Transformers 5.17.0 matches serving. CPU API/Pillow pins remain unchanged; A09 maintains an isolated training image for exact GPU-library parity. No review/split gate is bypassed. |
| 2026-10-17 | Select reversible short frame-ID aliases and two clip requests per job in deployment. | Eight synthetic 19-fact responses pass strict validation. With two 16-frame/672px requests, observed pair elapsed time was 14.547 s for compact/two-per-job versus 53.565 s for raw/one-per-job. One ordered trial per condition; grammar caching differs and this is not whole-video capacity or accuracy evidence. Code defaults remain raw/one-per-job. |

Original organizer media remains local. Production inference must not import the label workbook, annotation labels, split labels, or evaluation truth. The baseline, public schema, rules, and deployment versions freeze only against actual owner handoffs.
