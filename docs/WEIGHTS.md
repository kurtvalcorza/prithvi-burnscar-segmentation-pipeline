# Weight provenance, the pickle audit, the conversion, the pinned dataset tarball and DIMER hosting

This repository pins **one** model snapshot with its own `dimer-base-manifest.json` and **one** dataset tarball. The checkpoint is a pickle, which this pipeline audits and converts but never serves; the tarball is streamed once for exactly the pinned members and never `extractall`-ed.

## Prithvi-EO-2.0-300M BurnScars weights

- Upstream: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`
- Immutable revision: `a3f2c410e45b8ac7417976614528a872f024d831` (2025-09-29, "Update config.json"); the checkpoint was first published in commit `bf9c5cd9` (2025-02-28, "Upload Prithvi_EO_V2_300M_BurnScars.pt"); the file bytes are identical at both revisions.
- Source format: `Prithvi_EO_V2_300M_BurnScars.pt` — torch zip archive (`archive/data.pkl`, 359 entries) holding a PyTorch Lightning 2.5.0.post0 checkpoint: `state_dict` (355 tensors, float32 and int64, under the `model.` prefix), `epoch` 40, `global_step` 2665, plain-dict `hyper_parameters` (the `terratorch.tasks.SemanticSegmentationTask` arguments) and `loops`.
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter).
- Local layout: `weights/prithvi-eo-2.0-300m-burnscars/` holds the 10 manifest entries (the checkpoint, upstream `README.md`, `config.json`, `burn_scars_config.yaml`, the three split files `splits/{train,val,test}.txt`, three example scenes; 1,316,707,805 bytes total) with byte size and SHA-256 for each, plus the converted file described below. `verify_snapshot()` in `src/prithvi_burnscar_segmentation_pipeline/pipeline.py` checks the manifest entries, asserts the checkpoint's digest against the package constant, and checks the converted file against its pinned digest when present.
- Cross-check: the manifest's checkpoint digest `0c5f9334be9a75c9006387ab8f3dc05a55ea7fb5ef7956717316be57c62954d3` equals the `oid sha256` of the Hub LFS pointer at the pinned revision.

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles the file with `pickletools.genops` — every `.pkl` inside the torch zip archive — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `Prithvi_EO_V2_300M_BurnScars.pt` | `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` | exactly those four | `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256` (the digest covers the sorted set of global names, so it equals the digest of the sibling Sen1Floods11 checkpoint, which names the same four); `convert_model()` refuses a file whose audit digest differs. Tests craft a torch archive carrying `os.system` and a plain pickle of a `complex` number, and assert that the audit refuses each before anything is constructed.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict with a `state_dict` of tensors whose every key starts with `model.`;
- the prefix is stripped and the 355 tensors are loaded with `strict=True` into `EncoderDecoderFactory().build_model(task="segmentation", backbone="prithvi_eo_v2_300", backbone_pretrained=False, backbone_bands=[BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2], decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2, rescale=True, necks=[SelectIndices(5, 11, 17, 23), ReshapeTokensToImage, LearnedInterpolateToPyramidal])` — the pinned `burn_scars_config.yaml`'s model arguments; the model's own state dict is saved as safetensors.

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `prithvi-eo-2.0-300m-burnscars.safetensors` | 1,297,682,024 | 355 (324,411,275 elements; 324,204,674 parameters) | `4209c5a013f90dfe372abb4ade3865e7a283d655ef17b1c7de95d36ecc033c3f` | no (regenerated) |

The conversion is deterministic: the digest was reproduced on the build run, the smoke run and the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count.

Layout note: the backbone is the plain `prithvi_eo_v2_300` (no temporal or location embeddings, unlike the `TL` Sen1Floods11 fine-tune), so the encoder holds 294 tensors; the UNet decoder with channels `[512, 256, 128, 64]` and the 11 `neck.*` tensors of the learned pyramid interpolation make up the rest. The strict load matched every key with 0 missing, 0 unexpected and 0 shape mismatches.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match against the architecture rebuilt from the pinned configuration under `terratorch 1.2.13`, and sane predictions on the upstream example scenes: burn fractions 0.797 on `T10SEH.2018190`, 0.025 on `T10SFF.2018190` and 0.429 on `T10SGF.2020217`; on the 12 pinned test scenes the frozen model reaches a burn-scar IoU of 0.925 and on the 8 validation scenes 0.763 (`MODEL_CARD.md`, *Runtime*). Behaviour under the TerraTorch version that produced the checkpoint was not measured.

## Runtime facts

- The model is float32 as shipped; `predict` runs under `torch.inference_mode()` with float16 autocast on CUDA and moves results to the CPU; `adapt` trains with gradients only on the selected tensors, with BatchNorm running statistics frozen.
- Inputs are standardised with the six-band means and standard deviations of the pinned `burn_scars_config.yaml` after replacing no-data (0 or −9999) by 0; HLS scenes are already surface reflectance in [0, 1], and a chip arriving as reflectance × 10 000 is scaled by `CONSTANT_SCALE = 1e-4` first — the same preprocessing the fine-tune used.
- `terratorch` pulls `torchgeo`, `lightning`, `timm`, `segmentation-models-pytorch`, `albumentations`, `rasterio`, `geopandas`, `lightly`, `h5py`, `tensorboard` and others; the pipeline uses `terratorch.models.EncoderDecoderFactory` only, and reads scenes with `tifffile` (no rasterio, no georeferencing).

## The tutorial data: the pinned HLS Burn Scars tarball

`samples.py` fetches `hls_burn_scars.tar.gz` from the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at the immutable revision `1864285e25010d346a842e4f068b1a1d4248ed6d` (2,645,552,531 bytes, SHA-256 `4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6`, hashed once per `fetch_tarball` call and refused on a mismatch), then streams through it **once** with `extract_pinned_members` and copies out exactly the 88 pinned members — 44 six-band float32 `_merged.tif` scenes (6,295,168 bytes each) and their single-band `.mask.tif` masks (525,272 bytes each; values −1 / 0 / 1), 300,099,360 bytes uncompressed — each pinned by member path, byte size and SHA-256 in `SAMPLE_RECORDS`. No `extractall`, no path taken from the archive: every member is written under its base name in `weights/hls-burn-scars/chips/` (git-ignored). The scenes were chosen on 2026-09-19 with a fixed seed from the model repository's `splits/{train,val,test}.txt` (24 / 8 / 12) among the scenes whose mask is at least 60 % valid and 3 % burn; the roles follow those splits. The HLS Burn Scars dataset (NASA IMPACT / University of Alabama in Huntsville) is CC BY 4.0.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `inference.py` (the authors' rasterio/terratorch inference script); it is neither listed in the manifest nor executed. The pretrained (non-fine-tuned) `Prithvi-EO-2.0-300M` backbone is not fetched: `backbone_pretrained=False` and the fine-tuned checkpoint carries the encoder. Of the tarball, only the 88 pinned members are ever written to disk.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `prithvi-eo-2.0-300m-burnscars.safetensors`. **The `.pt` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class comes from `terratorch==1.2.13` on PyPI, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: a scene needs the 1.30 GB weights and one 512 × 512 six-band array; on an RTX 5070 Ti laptop GPU 12 scenes take 1.5 s in float16 autocast at 1.9 GB; a CPU takes tens of seconds per scene. An adapted profile needs the weights plus an 81 MB adapter (the UNet decoder is larger than the flood row's UPerNet decoder).
- One review item is open: whether the one-time restricted unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
