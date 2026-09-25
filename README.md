# Prithvi Burn-Scar Segmentation Pipeline

DIMER-oriented pipeline for **Prithvi-EO-2.0-300M BurnScars** (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`, the NASA/IBM Earth-observation foundation model fine-tuned for burn-scar mapping), pinned to an immutable Hugging Face revision. The repository exposes burn-scar segmentation of six-band HLS scenes with softmax scores, pixel IoU / F1 / precision / recall against the not-burned baseline, a labelled-chip contract with explicit ceilings, a bounded decoder fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`
- Revision: `a3f2c410e45b8ac7417976614528a872f024d831`
- Source asset: `Prithvi_EO_V2_300M_BurnScars.pt` (1,297,798,380 bytes, SHA-256 `0c5f9334…`) — a PyTorch Lightning checkpoint (a pickle), converted once to safetensors and never served (see below)
- Upstream weight license: Apache-2.0
- Upstream task: burn-scar segmentation of 512 × 512 HLS scenes (blue, green, red, narrow NIR, SWIR 1, SWIR 2) — the Prithvi-EO-2.0 300M encoder with a U-Net decoder, fine-tuned on the HLS Burn Scars dataset with TerraTorch
- Runtime: `terratorch==1.2.13` + `torch==2.14.0` + `torchvision` + `tifffile` — the model class comes from PyPI, **no Hub-hosted code and no served pickle**
- Repository adaptation: **E2E** (bounded fine-tuning of the pyramid neck, U-Net decoder and head — optionally the last encoder block — on labelled scenes, with a portable safetensors adapter)

## Two things to know before you start

**The upstream asset is a pickle, and it is never served.** The checkpoint is a torch archive whose pickle references only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage` and `torch.LongStorage`. `audit_pickle()` lists every global it would import without executing anything and refuses anything outside that allow-list (audit digest pinned); `convert_model()` unpickles it **once** through `torch.load(weights_only=True)`, strips the Lightning `model.` prefix, loads the 355 tensors strictly into the architecture rebuilt from `terratorch`, and writes `prithvi-eo-2.0-300m-burnscars.safetensors` (1,297,682,024 bytes), whose digest is pinned in `pipeline.py`; `from_pretrained()` loads only that.

**The model already trained on the tutorial's dataset, which ships as one 2.6 GB tarball.** The 44 sample scenes are extracted from the digest-pinned HLS Burn Scars tarball (exactly the 88 pinned members, each digest-verified, no `extractall`) with the roles of the non-overlapping splits the upstream authors publish beside the checkpoint, so the frozen model's numbers are in-distribution (test burn-scar IoU about 0.93 on the 12 held-out scenes, validation about 0.76) and the bounded adaptation has little to learn from 24 scenes: it is selected by validation loss with the frozen model as epoch 0 and moved the test burn-scar IoU from 0.925 to 0.926 in the build record — a sample-sanity observation, not a quality claim. The contract exists for *your* labelled scenes from another region, biome or year.

## Quick start

```python
from prithvi_burnscar_segmentation_pipeline import PrithviBurnScarPipeline, fetch_sample_dataset

pipe = PrithviBurnScarPipeline.from_pretrained()       # verifies the snapshot, audits + converts the pickle once, loads safetensors
splits = fetch_sample_dataset()                      # 24 / 8 / 12 pinned HLS Burn Scars scenes, extracted from the digest-verified tarball
print(pipe.evaluate(splits["test"])["model"])        # frozen model: IoU per class, mean IoU, accuracy, precision, recall, F1
pipe.adapt(splits["train"], splits["validation"])    # bounded fine-tuning of neck + decoder + head, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["model"])        # adapted model, same chips
masks = pipe.predict(splits["test"][:2])["predictions"]
pipe.save_artifact("outputs/adapter")
```

`predict()`, `evaluate()` and `adapt()` take records — `{id, image, label}` with a (6, 512, 512) reflectance array (or a GeoTIFF path; 13-band Sentinel-2 L1C files are reduced to the six HLS-equivalent bands) and a (512, 512) mask with 0 / 1 / −1. HLS scenes are already reflectance in [0, 1]; values above 2 are read as reflectance × 10 000 and scaled by 10⁻⁴ (once: a checked record is never rescaled), no-data (0 or −9999) is replaced by 0, and validation is structural: nothing checks that the bands are the right six in the right order or that the reflectance is corrected.

## Weights layout

```
weights/prithvi-eo-2.0-300m-burnscars/   README.md  config.json  burn_scars_config.yaml  splits/*.txt  Prithvi_EO_V2_300M_BurnScars.pt
                                          examples/*.tif  dimer-base-manifest.json
                                          prithvi-eo-2.0-300m-burnscars.safetensors  (git-ignored, converted)
weights/hls-burn-scars/                   hls_burn_scars.tar.gz (2.6 GB, digest-pinned) and chips/ with the 88 extracted members (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of all 10 manifest entries and of the converted file when present), converts the pickle when the safetensors file is absent, and refuses on the first mismatch. With `require_source=False` the digest-verified converted file is accepted without the checkpoint — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audit, the conversion and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` assembles 44 labelled HLS scenes of the HLS Burn Scars dataset — 24 from the training split, 8 from the validation split and 12 from the test split that the upstream authors publish beside the checkpoint, drawn with a fixed seed from the scenes whose mask is at least 60 % valid and 3 % burn scar. `fetch_tarball` downloads the 2.6 GB `hls_burn_scars.tar.gz` from the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at an immutable revision and refuses it on a size or SHA-256 mismatch; `extract_pinned_members` streams through it once and copies out exactly the 88 members pinned by path, byte size and SHA-256 in `SAMPLE_RECORDS` — each refused on its own mismatch, written under its base name, no `extractall`. The dataset is CC BY 4.0 (NASA IMPACT / UAH).

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the neck, decoder and head, about 81 MB; plus encoder block 23 with `trainable="decoder+last_block"`) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digest, the adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `PrithviBurnScarPipeline.from_artifact(dir)` re-verifies the base file, checks the manifest, scope and digest before deserialising, rebuilds the model and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles, temporary manifests, synthetic scenes, a synthetic tarball with a decoy member, an injected fetcher and a stub model, never the weights or `terratorch`; the model-backed smoke is recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/prithvi-burnscar-segmentation-pipeline/blob/main/tutorials/prithvi_burnscar_segmentation_colab.ipynb)

`tutorials/prithvi_burnscar_segmentation_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It needs a GPU runtime. It downloads and converts the pinned checkpoint in the runtime (the audit and conversion records are printed before the model loads), fetches the pinned tarball and extracts the pinned scenes, and runs the sample path: validation, the frozen model against the not-burned baseline, bounded fine-tuning, held-out evaluation, segmentation of the upstream example scenes, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Release-grade** — the `E2E` notebook blob `d8aa27bf` (committed at `f5d329c`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-25 (10/10 ok (1 restart after install cell), 430.8 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Licensing

- Upstream weights: Apache-2.0 (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`; the Prithvi-EO-2.0 code and TerraTorch are Apache-2.0), staged from the pinned revision and converted, not modified, into the served safetensors.
- Tutorial data: HLS Burn Scars (NASA IMPACT / University of Alabama in Huntsville, CC BY 4.0, attribution required); fetched at run time as a digest-pinned tarball, never committed.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
