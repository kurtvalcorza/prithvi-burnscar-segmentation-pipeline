# Fleet-sweep fixes: `prithvi_burnscar_segmentation_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `7361e28`. Changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`) and, for SWP-F, in the carried `pipeline.py`; the notebook is regenerated. Status and release
labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → `build_notebook.py/2.2` (fleet shared isolated runtime); the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (131 packages, the same pins and lock as prithvi-eo-feature-extraction; `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. Section 8 imports `importlib.metadata` itself. | Section 1, Section 8; `tools/build_notebook.py`, `tools/notebook_template.py`, `tools/validate_release_assets.py`, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–7), What to notice + Check your reasoning after Sections 4–8 quoting the recorded Kaggle T4 run of 2026-09-25 (frozen burn-scar IoU 0.9251 → adapted 0.9263, baseline accuracy 0.7869, validation loss 0.1072 → 0.0984, validation IoU 0.7627 → 0.7701), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. The literal `{{id, image, label}}` in the Prerequisites now renders as `{id, image, label}`. | opening, Sections 4–8 markdown, closing, Prerequisites | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Fixed (no direction was asserted) | The sweep flagged Section 7's `assert kept-epoch val_loss <= frozen val_loss`. On inspection it is a procedure invariant (epoch 0 is a selection candidate), not a quality comparison; it and the validation re-scoring check are now explicit `RuntimeError` contract checks. A recorded verdict (`improved` / `no change` / `worse`, with the delta) is added to the report and `result.json`. | Section 7, Section 8 | `test_swp_a_no_model_quality_assert_remains`, `test_swp_a_section_7_records_the_verdict_and_writes_the_report[worse/no change/improved]`, `test_swp_a_contract_checks_still_stop_the_run` |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trained the neck/decoder/head in place starting from whatever weights the model held, so re-running Section 6 (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model", and re-running Section 5 scored the adapted model as frozen. `pipeline.py` now keeps the pinned-base value of every tensor adapt() or load_artifact() changes and restores it first (siglip-v1's `restore_base` pattern); `restore_base()` is public; a failed adapt leaves the weights as before; the result records `started_from`. Section 5 calls `pipe.restore_base()` before its frozen evaluation. | `src/…/pipeline.py`, Sections 5–6 | `test_swp_f_adapt_and_load_artifact_restore_the_pinned_base_first`, `test_swp_f_restore_base_puts_trained_tensors_back_and_detaches_the_adapter`, `test_swp_f_section_5_restores_the_base_before_the_frozen_evaluation` |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (zip or folder; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, non-.zip each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- `PrithviBurnScarPipeline.adapt()` and `load_artifact()` now always start from the pinned base; new `restore_base()`; `adapt()` result has `started_from`. A re-run of Section 5 after Section 6 puts the base back (re-run 6–8 next).
- New `BYOD_PATH` field; Section 7 prints a `verdict`; `result.json` carries `verdict`.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; Sections 4 (BYOD block) and 7 run with stand-in pipelines; `restore_base` runs on a NumPy stand-in model. Plumbing evidence, not model evidence.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- `pytest` with CI's dependencies only (torch absent): 35 passed, 1 skipped before → 53 passed, 1 skipped after.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 8 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
