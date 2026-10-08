"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E burn-scar-segmentation workflow: the pinned Prithvi burn-scar checkpoint (a
Lightning pickle) is digest-verified, statically audited and converted once into safetensors, 44 labelled HLS
scenes are extracted from the digest-pinned HLS Burn Scars tarball, validated and assigned the model repository's
roles, the frozen model is scored against the not-burned baseline, a bounded decoder fine-tuning runs in the
kernel, the held-out chips are scored again, new chips are segmented, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "prithvi-burnscar-segmentation-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/prithvi_burnscar_segmentation_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ibm--nasa--geospatial%2FPrithvi--EO--2.0--300M--BurnScars-ffcc4d?style=flat",
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-NASA--IMPACT%2FPrithvi--EO--2.0-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/NASA-IMPACT/Prithvi-EO-2.0",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2412.02732-b31b1b.svg", "https://arxiv.org/abs/2412.02732"),
]

TEMPLATE = {
    "package": "prithvi_burnscar_segmentation_pipeline",
    "repo_name": REPO,
    "stem": "prithvi_burnscar_segmentation",
    "notebook_name": "prithvi_burnscar_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime builds an isolated environment from the hash-locked pins (torch, torchvision, "
        "terratorch and its stack, tifffile, numpy, safetensors, huggingface-hub); nothing is installed into the notebook's own Python, so "
        "no restart is needed and Run all completes in one pass. It then stages and digest-verifies the pinned Prithvi burn-scar checkpoint "
        "(1.30 GB) from the Hub, statically audits the Lightning pickle against an allow-list, converts it once into safetensors "
        "with a pinned digest, rebuilds the architecture from the installed `terratorch` package and loads it strictly, fetches the "
        "digest-pinned HLS Burn Scars tarball (2.6 GB, no credential) and extracts exactly the 44 pinned scenes and masks, validates "
        "them and assigns the model repository's roles (24 training, 8 validation, 12 test), segments the held-out scenes with the "
        "frozen model and scores them against the not-burned baseline, runs a bounded fine-tuning of the neck, U-Net decoder and "
        "head, scores the same scenes again, segments the three example scenes shipped with the upstream repository, exports the "
        "adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The "
        "default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes a few minutes of model time after the downloads; building "
        "the isolated environment and the tarball are the slowest steps."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled scenes — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, through "
        "the Colab upload dialog — as a zip (or folder) holding `pairs.csv` (columns `id`, `image`, `label`) beside six-band 512 × 512 GeoTIFF chips "
        "(blue, green, red, narrow NIR, SWIR 1, SWIR 2 — surface reflectance in [0, 1] or × 10 000) and single-band label rasters "
        "(0 = not burned, 1 = burn scar, −1 = no data); at least **7** labelled chips with some burn scar (the seeded 25 % test / 20 % validation split must "
        "leave the 4 training chips adaptation needs; Section 4 prints the minimum and refuses a smaller set by name). An optional `group` column (a fire or HLS "
        "tile id) keeps every chip of one group in one role, so neighbouring scenes of one fire cannot sit on both sides of the split. Your chips flow through the "
        "same contract — validation, frozen baseline, adaptation, held-out evaluation, inference, artifact export and reload parity; Section 5 puts the model back "
        "to the pinned base first, so the frozen numbers on your scenes are the packaged model's. The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional "
        "and never part of the default path."
    ),
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a geospatial '
            'foundation model fine-tuned for one task is evaluated and adapted honestly: how its burn-scar maps are scored against a baseline that predicts *not '
            'burned* everywhere, what a bounded fine-tuning of the decoder does to a model that already trained on this dataset, and how the change is exported and '
            'reloaded. No prior experience with Prithvi, TerraTorch or remote sensing models is assumed; terms are explained where they first matter and again in '
            'the **Glossary** at the end. A GPU runtime (T4 or better) is expected.\n\n**Input → Model → Output.**\n\n| | Segmentation | Bounded fine-tuning |\n|---|---|---|\n| '
            'Input | six-band 512 × 512 HLS reflectance chips (blue, green, red, narrow NIR, SWIR 1, SWIR 2) | labelled chips with 0 / 1 / −1 masks (24 training, 8 '
            'validation, 12 test in the sample) |\n| Model | the Prithvi-EO-2.0 ViT-L encoder, pyramid neck, U-Net decoder and two-class head, loaded from audited, '
            'converted safetensors | the same model; only the neck, decoder and head (20.3 M parameters) are trained, the encoder and BatchNorm statistics stay '
            "frozen |\n| Output | a per-pixel burn-scar mask, softmax scores, burn fraction; pixel IoU / F1 beside the not-burned baseline | the adapted model's "
            "numbers beside the frozen model's on the same held-out scenes, and an 81 MB safetensors adapter that reloads with parity |\n\n**How to use this "
            'notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run all**. Run all completes in one pass: Section 1 '
            "installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried "
            'package and the audited model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. '
            'Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the '
            'notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded '
            'run (the Kaggle Tesla T4 run of 25 September 2026 recorded in `docs/release-verification.md`). Every adaptation starts from the pinned base, so '
            're-running Section 6 with other settings is a fresh experiment. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. '
            'Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the labelled scenes, validation and refusals *(evaluation practice)* → '
            '5 the frozen model against the not-burned baseline *(core concept)* → 6 bounded fine-tuning of the neck, decoder and head *(core concept)* → 7 the '
            'held-out paired comparison with per-scene numbers and a figure *(evaluation practice)* → 8 the upstream example scenes and their provenance, export and fresh reload *(engineering)* → interpretation, an optional experiment, troubleshooting, glossary '
            'and your conclusion.'
        ],
    },
    "pipeline_class": "PrithviBurnScarPipeline",
    "model_load": "PrithviBurnScarPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "prithvi-eo-2.0-300m-burnscars",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "timm", "lightning", "tifffile"],
    "title": "Prithvi-EO-2.0 burn-scar mapping — DIMER E2E segmentation fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "burn-scar segmentation of six-band HLS scenes with a Prithvi-EO-2.0 ViT-L encoder and U-Net decoder, held-out IoU/F1 against a not-burned baseline, and bounded fine-tuning of the decoder to labelled scenes",
    "intro": (
        "Prithvi-EO-2.0 (Szwarcman et al., 2024) is NASA and IBM's foundation model for Harmonized Landsat Sentinel-2 imagery: a "
        "ViT-L masked autoencoder pretrained on 4.2 M global multispectral samples. The checkpoint packaged here is the upstream "
        "authors' fine-tune for burn-scar mapping — the 300 M-parameter encoder, a learned pyramid neck over four encoder depths, a "
        "U-Net decoder and a two-class head — trained on the 804 labelled HLS scenes of the HLS Burn Scars dataset (2018–2021, "
        "contiguous United States) with TerraTorch, on the non-overlapping splits the authors publish beside the checkpoint.\n\n"
        "Two things about this row are handled in the open. **The upstream asset is a pickle** — a PyTorch Lightning checkpoint. "
        "Section 3 downloads and digest-verifies it, statically lists every global the pickle would import (a state dict of tensors "
        "and nothing else), refuses anything outside that allow-list, unpickles it exactly once through torch's weights-only loader, "
        "and writes a safetensors file whose digest is pinned in the carried module; the model you run is rebuilt from the installed "
        "`terratorch` package and loads that file strictly. **The dataset ships as one 2.6 GB tarball**, so Section 4 pins the "
        "tarball by size and digest, streams through it once and copies out exactly the 88 pinned members (each pinned again by "
        "size and digest, no `extractall`, no paths taken from the archive), and leaves the other 1,522 alone. **The model is "
        "already fine-tuned on this dataset**, so the bounded adaptation in Section 6 is a demonstration of the contract, selected by validation "
        "loss with the frozen model as epoch 0; the point of the contract is the same recipe applied to *your* labelled scenes."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify a pickled "
        "checkpoint, read its static audit and see it converted into safetensors; extract pinned members from a digest-verified "
        "tarball and validate real labelled multispectral scenes with an ignore class; read pixel IoU, F1, precision and recall "
        "against a not-burned baseline; run a bounded decoder fine-tuning with explicit hyperparameters and frozen BatchNorm "
        "statistics; compare the adapted and frozen models on the same held-out scenes, per scene and pooled, and look at the masks; segment the upstream example scenes; and export a safetensors "
        "adapter that reloads against the pinned base with verified parity. The three example scenes are not new to the checkpoint — the notebook shows where each one comes from."
    ),
    "exclusions": (
        "burn severity or dNBR estimation, active-fire detection, temporal stacks (the packaged fine-tune is single-date), tiling of "
        "scenes larger than 512 × 512, atmospheric correction, cloud masking, the published benchmark scores, and any claim that a "
        "44-scene sample stands in for an operational evaluation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the ViT-L encoder runs in float16 autocast and the default adaptation needs about 2.5 GB of GPU memory; on CPU one 512 × 512 scene takes tens of seconds and the adaptation would take an hour. About 6 GB of disk is needed for the checkpoint, its conversion and the tarball; building the isolated environment (terratorch pulls torchgeo, lightning and their dependencies) takes several minutes the first time and is reused on a re-run.",
        "- **Knowledge:** what a multispectral surface-reflectance scene is (bands, scaling, no-data), what a pixel-wise segmentation mask and an ignore class are, and how IoU, precision and recall are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `terratorch` is installed from PyPI at a pinned version.",
        "- **Data contract:** a record is `{id, image, label}` — a (6, 512, 512) reflectance array (or a GeoTIFF; 13-band Sentinel-2 L1C files are reduced to the six HLS-equivalent bands) with values in [0, 1] or × 10 000, no-data 0 or −9999, and a (512, 512) mask with 0 / 1 / −1. Validation is structural: nothing checks that the bands are the right six in the right order, that the reflectance is corrected, or that the label belongs to the scene.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — commercial imagery under licence or unreleased fire-damage assessments are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches one pinned object — the 2.6 GB `hls_burn_scars.tar.gz` of the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at an immutable revision — over HTTPS, digest-verified before any member is read; the dataset is CC BY 4.0 (NASA IMPACT / University of Alabama in Huntsville).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample scenes, validation and roles\n\n"
                "The default dataset is 44 labelled HLS scenes of the HLS Burn Scars dataset — 24 from the training split, 8 from "
                "the validation split and 12 from the test split that the upstream authors publish beside the checkpoint, drawn "
                "with a fixed seed from the scenes whose mask is at least 60 % valid and 3 % burn scar. `fetch_corpus` downloads "
                "the dataset tarball from the Hub at its immutable revision, refuses it on any size or SHA-256 mismatch, streams "
                "through it once and copies out exactly the 88 pinned members — each refused on its own size or digest mismatch "
                "and written under its base name, never at a path taken from the archive — then reads the six-band float32 "
                "scenes (already surface reflectance in [0, 1]) and the masks, which keep −1 for no data. `dataset_manifest` "
                "validates every split, checks that no scene appears twice and records a digest.\n\n"
                "Look for: 24 / 8 / 12 scenes with burn fractions around 0.12..0.21, tile ids (UTM zone and grid square) per "
                "split, a written sample pair (`outputs/{stem}_sample_chip.tif` + `_sample_label.tif`, the BYOD shape), a complete BYOD example "
                "(`outputs/{stem}_byod_example/`: `pairs.csv` beside the first 7 held-out scenes under the names the table lists — zip that folder and it loads "
                "through `BYOD_PATH` unchanged; 7 is the smallest dataset the split accepts, and the cell prints it), and three "
                "refusal probes — a five-band scene, a mask with an unknown class, a scene with reflectance far outside range — "
                "each rejected before the model runs. The tarball takes about a minute to fetch and a minute to stream.\n\n"
                "One caveat in the sample itself: 2 of the 12 test scenes (`T10TFQ.2018245`, `T10TGS.2018245`) share an HLS tile with training scenes "
                "(`T10TFQ.2019245`, `T10TGS.2018190`) — an upstream split choice kept here so the roles match the model repository's, and the reason the "
                "interpretation says to split by fire or tile. For your own scenes the `group` column of `pairs.csv` does exactly that: `split_dataset` keeps "
                "every chip of one group in one role, and the cell reports whether the split was grouped.\n\n"
                "*Evaluation practice.* **Predict before running:** about one pixel in six is burned in these scenes. What accuracy will a "
                "model get that never predicts a burn?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "# A .zip or a folder already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding pairs.csv and the GeoTIFF files.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding pairs.csv and the GeoTIFF files.')\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    byod_records = load_byod_dataset(byod_path)\n"
                "    byod_grouped = bool(byod_records) and all(r.get('group') for r in byod_records)\n"
                "    splits = split_dataset(byod_records, seed=0)\n"
                "    print({{'byod_chips': len(byod_records), 'minimum_chips': byod_minimum_records(), 'grouped_split': byod_grouped, 'groups': sorted({{r['group'] for r in byod_records}}) if byod_grouped else 'no group column: scenes of one fire or tile may land in different roles'}})\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/hls-burn-scars')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    print({{name: {{'burn_fraction': part['class_pixel_fraction']['burn scar'], 'ignored_pixels': part['ignored_pixels'], 'tiles': part['regions']}}}})\n"
                "print({{'first_test_scene': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_chip.tif', 'outputs/{stem}_sample_label.tif')\n"
                "byod_example = write_byod_example((test_records + val_records + train_records)[:byod_minimum_records()], 'outputs/{stem}_byod_example')\n"
                "print({{'sample_pair': sample_pair, 'byod_example': {{k: byod_example[k] for k in ('folder', 'pairs_csv', 'n_pairs', 'minimum_chips')}}, 'how_to_reuse': 'zip the folder and set BYOD_PATH to the zip'}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'five-band scene': [{{**test_records[0], 'image': test_records[0]['image'][:5]}}, *test_records[1:4]],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 7, test_records[0]['label'])}}, *test_records[1:4]],\n"
                "    'reflectance out of range': [{{**test_records[0], 'image': test_records[0]['image'] * 50000.0}}, *test_records[1:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the split sizes, `burn_fraction` and `ignored_pixels` per split, the tile ids, and the three refusals.\n\n<details><summary>Check '
                'your reasoning</summary>Roughly 80 %. In the recorded run the not-burned baseline scored accuracy 0.7869 on the test pixels with a burn-scar IoU of 0 '
                '— high accuracy for doing nothing, which is why the burn-scar IoU, precision and recall are the numbers to read. No-data pixels (−1) are excluded from '
                'every count rather than treated as unburned. The refusals (five bands, an unknown label class, reflectance far out of range) stop before any model '
                'call and name the rule. Note what validation cannot catch: six bands in the wrong order.</details>'
            ),
        },
        {
            "md": (
                "## 5. The frozen model against the not-burned baseline\n\n"
                "`pipe.predict` standardises each scene with the band statistics of the upstream training configuration, runs the "
                "encoder, neck, decoder and head in float16 autocast, and returns the argmax mask, the softmax scores (the model's "
                "outputs, not calibrated probabilities) and the burn fraction per scene. `pipe.evaluate` pools the labelled pixels "
                "of every held-out scene into one confusion matrix (−1 pixels excluded) and reports the per-class IoU, mean IoU, "
                "accuracy, and the burn-scar class's precision, recall and F1; the **not-burned baseline** — every pixel predicted "
                "as unburned — is scored on the same pixels, so its accuracy is exactly the unburned fraction and its burn IoU is 0. "
                "The decision rule is the argmax over the two class scores — equivalent to a 0.5 threshold on the burn-scar score — and it is a default, not a tuned "
                "operating point: a deployment chooses its own threshold on its own validation data, weighing a missed burned pixel against a false one, and owns "
                "the calibration of the scores; this notebook sets neither.\n\n"
                "Look for: a burn-scar IoU above 0.9 on the test scenes (in the build record 0.925 with F1 0.961 — this fine-tune is "
                "strong on its own test split, from which these scenes were drawn) and a lower validation IoU (about 0.76: a few "
                "validation scenes are hard). These are sample-sanity numbers on 12 and 8 scenes, not the benchmark. If you re-run this "
                "cell after Section 6, it first puts the adapted tensors back to the pinned base, so *frozen* always means the packaged "
                "model.\n\n"
                "**Predict before running:** the packaged model was fine-tuned on this dataset's training split. How close to perfect "
                "will its test-scene burn-scar IoU be, and will validation be easier or harder?"
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "# SWP-F: the frozen numbers are always the pinned base. On a re-run after Section 6 the adapted tensors are put back\n"
                "# to the base first (then re-run Sections 6-8 in order); adapt() itself also starts from the base on every call.\n"
                "restored_tensors = pipe.restore_base()\n"
                "if restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'adapted tensors put back to the pinned base; re-run Sections 6-8 in order'}})\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_not_burned_test': {{k: frozen_test['baseline_not_burned'][k] for k in ('iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('iou', 'mean_iou', 'accuracy', 'precision', 'recall', 'f1')}}}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('iou', 'f1')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "for record, pred in list(zip(test_records, frozen_predictions['predictions']))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'scene': record['source_id'], 'burn_label': round(float((record['label'] == 1).sum() / labelled.sum()), 3), 'burn_predicted': pred['class_fraction']['burn scar'], 'ignored': int((~labelled).sum())}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'scores_shape': frozen_predictions['predictions'][0]['scores'].shape}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `baseline_not_burned_test` beside `frozen_test`, precision against recall, the validation IoU, and the per-scene burn fractions.\n\n<details><summary>Check '
                'your reasoning</summary>Close, not perfect. In the recorded run the frozen test burn-scar IoU was 0.9251 (F1 0.9611, accuracy 0.9833 against the '
                "baseline's 0.7869), while validation was clearly harder at 0.7627 — a few validation scenes have ambiguous scar edges. The scores are softmax outputs, "
                'not calibrated probabilities; the mask is their argmax.</details>'
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the neck, decoder and head\n\n"
                "`pipe.adapt` trains the 34 tensors of the pyramid neck, the U-Net decoder and the head (20.3 M parameters — 6.3 % of "
                "the model) and nothing else: the ViT-L encoder is frozen (no gradient is stored for it), and every BatchNorm layer "
                "keeps its running statistics, because batches of two scenes would corrupt them. Each step takes two scenes with a "
                "seeded horizontal or vertical flip, computes the cross-entropy over the labelled pixels (−1 ignored) and takes an "
                "AdamW step at a small fixed learning rate with gradient-norm clipping and float16 loss scaling. Epoch 0 records the "
                "frozen model's validation loss and metrics; the epoch with the lowest validation loss is kept — which can be "
                "epoch 0, since the packaged model already trained on this dataset.\n\n"
                "Watch the validation loss: in the build record it dipped at epoch 1 and drifted up afterwards — the sign that a "
                "small learning rate and validation selection are doing their job on a model that has little left to learn from "
                "24 scenes of a dataset it trained on. Four epochs (48 steps) take under a minute on a T4; the cell prints the peak GPU memory it used. "
                "`TRAINABLE = 'decoder+last_block'` also unfreezes the last encoder block (32.9 M parameters). Every call starts "
                "from the pinned base (`started_from` in the printed result), so a re-run with other settings is a fresh experiment, "
                "not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** the model has already seen these 24 training scenes' dataset. Will validation loss improve "
                "at all, and if so, at which epoch?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 2  # @param {{type:\"integer\"}}\n"
                "TRAINABLE = 'decoder'  # @param [\"decoder\", \"decoder+last_block\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_burn_iou'] = entry['val']['iou']['burn scar']\n"
                "        row['val_f1'] = entry['val']['f1']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "adapt_result['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds, 'gpu_peak_gb': adapt_result['gpu_peak_gb']}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the validation loss per epoch, `best_epoch`, and the trainable share of the parameters.\n\n<details><summary>Check '
                'your reasoning</summary>A little, once. In the recorded run validation loss went from 0.1072 (frozen) to 0.0984 at epoch 1 and then drifted up, so '
                'epoch 1 was kept (validation burn-scar IoU 0.7627 → 0.7701). A small learning rate and validation-loss selection keep a model with little left to '
                'learn from getting worse. What a ten-times larger rate does is the first optional experiment at the end — predict it before you run it.</details>'
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test scenes were never used for training or epoch selection (they come from the model repository's test "
                "split). The adapted model is scored exactly as the frozen model was in Section 5, and the table puts the baseline, "
                "the frozen and the adapted numbers side by side. The cell stops only on what the procedure guarantees — the kept "
                "epoch's validation loss is no higher than the frozen model's (epoch 0 is a candidate), and re-scoring the validation "
                "scenes reproduces the kept epoch's burn-scar IoU within 0.01 (float16 kernels are not bit-reproducible across batch "
                "sizes) — and records the test direction as a **verdict** (`improved`, `no change` or `worse`) instead of asserting one: on this sample the burn-scar IoU moved from 0.925 to 0.926 in the "
                "build record, a sample-sanity observation on 12 scenes with no dispersion estimate, not a quality claim. With your "
                "own scenes from another region or year, the gap between frozen and adapted is the number to watch — and in a BYOD run *frozen* is the packaged "
                "model, because Section 5 restored the pinned base before scoring it.\n\n"
                "The pooled numbers let large burns dominate, so the cell also prints the burn-scar IoU **per scene** for both models and the range across the "
                "12 scenes, and it draws the first two held-out scenes: a SWIR 2 / NIR / red composite (burn scars are bright and reddish, healthy vegetation "
                "dark), the label, the frozen and adapted masks, and the adapted model's errors (red = burned pixels it missed, blue = unburned pixels it "
                "called burned; grey = no data). The figure is written to `outputs/{stem}_test_scenes.png`.\n\n"
                "*Evaluation practice.* **Predict before running:** after fine-tuning on 24 scenes, will the test burn-scar IoU move by "
                "more than 0.01 — and if precision and recall move, will they move the same way?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('mean_iou', 'accuracy', 'precision', 'recall', 'f1'):\n"
                "    comparison[key] = {{'baseline_not_burned': frozen_test['baseline_not_burned'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "comparison['burn_iou'] = {{'baseline_not_burned': frozen_test['baseline_not_burned']['iou']['burn scar'], 'frozen': frozen_test['model']['iou']['burn scar'], 'adapted': adapted_test['model']['iou']['burn scar']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "delta_burn_iou = adapted_test['model']['iou'][CLASS_NAMES[1]] - frozen_test['model']['iou'][CLASS_NAMES[1]]\n"
                "# SWP-A: the direction is a recorded verdict, never an assert, so a BYOD run always reaches export and result.json.\n"
                "comparison['verdict'] = {{'adapted_vs_frozen_burn_iou': 'improved' if delta_burn_iou > 0 else ('no change' if delta_burn_iou == 0 else 'worse'), 'delta': round(delta_burn_iou, 4)}}\n"
                "print({{'verdict': comparison['verdict']}})\n"
                "print({{'validation_burn_iou': {{'frozen': frozen_val['model']['iou']['burn scar'], 'adapted': adapted_val['model']['iou']['burn scar']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "# BS-S4: per-scene burn-scar IoU (the pooled metric lets large burns dominate).\n"
                "adapted_predictions = pipe.predict(test_records)\n"
                "test_labels, test_ids = [r['label'] for r in test_records], [r.get('source_id', r['id']) for r in test_records]\n"
                "per_scene = {{'frozen': per_chip_burn_iou([p['mask'] for p in frozen_predictions['predictions']], test_labels, test_ids), 'adapted': per_chip_burn_iou([p['mask'] for p in adapted_predictions['predictions']], test_labels, test_ids)}}\n"
                "for f_row, a_row in zip(per_scene['frozen'], per_scene['adapted']):\n"
                "    print({{'scene': f_row['id'], 'burn_label': f_row['positive_fraction'], 'frozen_burn_iou': f_row['iou'], 'adapted_burn_iou': a_row['iou']}})\n"
                "comparison['per_scene_burn_iou_range'] = {{name: [min(v), max(v)] if (v := [row['iou'] for row in rows if row['iou'] is not None]) else None for name, rows in per_scene.items()}}\n"
                "print({{'per_scene_burn_iou_range': comparison['per_scene_burn_iou_range']}})\n"
                "# BS-m6: see the scenes, labels, masks and errors, not only the numbers.\n"
                "import matplotlib.pyplot as plt\n\n"
                "shown = list(zip(test_records, frozen_predictions['predictions'], adapted_predictions['predictions']))[:2]\n"
                "fig, axes = plt.subplots(len(shown), 5, figsize=(17, 3.6 * len(shown)), squeeze=False)\n"
                "for row, (record, frozen_pred, adapted_pred) in zip(axes, shown):\n"
                "    label, composite = record['label'], false_colour_composite(record['image'])\n"
                "    errors = composite * 0.35\n"
                "    errors[(label == 1) & (adapted_pred['mask'] == 0)] = (1.0, 0.1, 0.1)\n"
                "    errors[(label == 0) & (adapted_pred['mask'] == 1)] = (0.2, 0.4, 1.0)\n"
                "    errors[label < 0] = 0.5\n"
                "    panels = [('SWIR 2 / NIR / red', composite, None), ('label (white = burn, grey = no data)', np.where(label < 0, 0.5, label.astype(np.float32)), 'gray'), ('frozen mask', frozen_pred['mask'], 'gray'), ('adapted mask', adapted_pred['mask'], 'gray'), ('adapted errors (red missed, blue false)', errors, None)]\n"
                "    for ax, (title, image, cmap) in zip(row, panels):\n"
                "        ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')\n"
                "        ax.set_title(record.get('source_id', record['id']) + ': ' + title, fontsize=9)\n"
                "        ax.set_axis_off()\n"
                "fig.tight_layout()\n"
                "fig.savefig('outputs/{stem}_test_scenes.png', dpi=80)\n"
                "plt.show()\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'per_scene_burn_iou': per_scene,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "# Contract integrity (not model quality): epoch 0 is a selection candidate, and re-scoring reproduces the kept epoch.\n"
                "if adapt_result['history'][adapt_result['best_epoch']]['val_loss'] > adapt_result['history'][0]['val_loss']:\n"
                "    raise RuntimeError('contract: the kept epoch has a higher validation loss than the frozen model, which epoch selection cannot produce')\n"
                "if abs(adapted_val['model']['iou'][CLASS_NAMES[1]] - adapt_result['history'][adapt_result['best_epoch']]['val']['iou'][CLASS_NAMES[1]]) >= 1e-2:\n"
                "    raise RuntimeError('contract: re-scoring the validation scenes does not reproduce the kept epoch')\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the `burn_iou` row (baseline, frozen, adapted), the `precision` and `recall` rows, the `verdict` line with its `delta`, the per-scene range, and in the figure where the red and blue pixels sit.\n\n<details><summary>Check your '
                'reasoning</summary>No, and not the same way. In the recorded run the test burn-scar IoU moved from 0.9251 to 0.9263 (F1 0.9611 → 0.9618, accuracy 0.9833 → 0.9839): the '
                'verdict was *improved* by about 0.001, far inside what one seed and 12 scenes can resolve. Underneath that flat IoU, precision rose from 0.9528 to 0.9741 while recall fell '
                'from 0.9695 to 0.9498: the adapted model draws tighter scars — fewer unburned pixels called burned, more burned pixels missed — a trade that matters for '
                'burned-area mapping (a conservative map under-reports area) even when the IoU does not move. Read the verdict as *the contract ran and did no harm*, not as a '
                'gain, and read the per-scene range to see which scenes the pooled number hides. On scenes from another region or year the same delta is the number that matters.</details>'
            ),
        },
        {
            "md": (
                "## 8. The upstream example scenes, artifact export and fresh reload\n\n"
                "The adapted model segments the three example scenes that ship with the upstream model repository (three HLS tiles over "
                "California, 2018 and 2020). **None of them is new to the checkpoint**, and the cell says so for each scene from the files themselves: "
                "`example_provenance` hashes the scene, looks it up in the checkpoint's own `splits/train.txt`, `val.txt` and `test.txt`, and compares it with the 44 "
                "pinned sample records. `T10SEH.2018190.v1` is byte-identical to sample test scene `test-001`, which Section 7 has already scored — so here its "
                "mask is compared with that label (burn-scar IoU printed as `label_agreement`); `T10SFF.2018190.v1` and `T10SGF.2020217.v1` are in the "
                "checkpoint's training split, so their burn fractions show what the model reproduces on scenes it trained on, not generalisation. They are used "
                "because they are the only unlabelled-looking inputs the model repository ships; a real sanity check on *new* scenes needs scenes from outside this "
                "dataset, which is what BYOD is for.\n\n"
                "`pipe.save_artifact` writes the trained tensors (about 81 MB) as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the converted base file, the "
                "adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `PrithviBurnScarPipeline.from_artifact` re-verifies the base file, checks the artifact manifest, scope and "
                "digest **before** deserialising, rebuilds the model and overlays the tensors — a fresh object from files, not the "
                "in-memory model (VER2). The cell asserts the same held-out burn-scar IoU within 0.001 and score maps within 0.01 "
                "(VER4: float16 tolerances; on one device they are usually identical)."
            ),
            "code": (
                "import importlib.metadata\n"
                "import platform\n"
                "import shutil\n\n"
                "import tifffile\n\n"
                "example_dir = WEIGHTS_DIR / 'examples'\n"
                "new_records = [{{'id': path.stem.replace('subsetted_512x512_HLS.S30.', ''), 'image': read_chip(path), 'source': str(path.name)}} for path in sorted(example_dir.glob('*.tif'))]\n"
                "new_predictions = pipe.predict(new_records)\n"
                "# BS-m2: say what each example scene is (checkpoint split files + pinned sample digests), never call it new.\n"
                "provenance = {{}}\n"
                "for record, pred in zip(new_records, new_predictions['predictions']):\n"
                "    tifffile.imwrite(f'outputs/{stem}_mask_' + record['id'] + '.tif', pred['mask'])\n"
                "    prov = example_provenance(example_dir / record['source'], WEIGHTS_DIR / 'splits')\n"
                "    row = {{'scene': record['id'], 'burn_fraction': pred['class_fraction']['burn scar'], 'checkpoint_split': prov['upstream_split'], 'sample_record': prov['sample_record']}}\n"
                "    labelled = [r for r in test_records if r.get('source_id') == (prov['sample_record'] or {{}}).get('key')]\n"
                "    if labelled:\n"
                "        row['label_agreement'] = per_chip_burn_iou([pred['mask']], [labelled[0]['label']], [labelled[0]['id']])[0]\n"
                "        row['note'] = 'a sample test scene already scored in Section 7 (byte-identical); compared with its label here'\n"
                "    elif prov['upstream_split']:\n"
                "        row['note'] = 'in the checkpoint ' + '/'.join(prov['upstream_split']) + ' split: reproduction on a training scene, not generalisation'\n"
                "    else:\n"
                "        row['note'] = 'not in the checkpoint split files or the sample: a sanity check without a label'\n"
                "    provenance[record['id']] = row\n"
                "    print(row)\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': new_predictions['model'], 'classes': new_predictions['classes'], 'decision_rule': new_predictions['decision_rule'], 'predictions': [{{'id': p['id'], 'class_fraction': p['class_fraction'], 'provenance': provenance[p['id']]}} for p in new_predictions['predictions']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PrithviBurnScarPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'positive_iou_diff': round(abs(reloaded_test['model']['iou'][CLASS_NAMES[1]] - adapted_test['model']['iou'][CLASS_NAMES[1]]), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_score_diff': max(float(np.abs(a['scores'] - b['scores']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['positive_iou_diff'] < 1e-3 and parity['max_abs_score_diff'] < 1e-2\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'data_tarball': {{'name': TAR_NAME, 'sha256': TAR_SHA256, 'pinned_members': 2 * len(SAMPLE_RECORDS)}},\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'lightning': lightning.__version__, 'tifffile': tifffile.__version__, 'terratorch': importlib.metadata.version('terratorch')}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'verdict': comparison['verdict'],\n"
                "    'example_scenes': provenance,\n"
                "    'adaptation_gpu_peak_gb': adapt_result['gpu_peak_gb'],\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                "**What to notice:** each example scene's `checkpoint_split` and `sample_record`, the `label_agreement` of the one that is a sample test scene, the artifact's size and tensor count, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>One example scene is sample test scene `test-001` (in the recorded run its label had burn fraction 0.791 and the adapted model predicted 0.7891), '
                'the other two are in the checkpoint\'s training split, so none of the three burn fractions says anything about new scenes. In the recorded run the adapter (34 '
                'tensors, about 81 MB) reloaded into a fresh pipeline with identical held-out metrics (`positive_iou_diff` 0.0, `metrics_identical` True, '
                '`max_abs_score_diff` 0.0): the adapter plus the pinned, re-verified base is the whole adapted model.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 12 held-out scenes from the model repository's test split the packaged burn-scar model finds burn scars with an IoU "
        "above 0.9, against a not-burned baseline that scores 0; a bounded fine-tuning of its neck, decoder and head on 24 scenes, "
        "selected by validation loss with the frozen model as a candidate, leaves the IoU where it was (0.9251 → 0.9263 in the build record) while trading "
        "recall for precision underneath it (precision 0.9528 → 0.9741, recall 0.9695 → 0.9498): a tighter, more conservative scar map. That is the "
        "claim: the adaptation contract runs end to end on real labelled multispectral scenes drawn from a digest-verified "
        "tarball, the pickle is audited and converted rather than served, and the artifact that carries the change is 81 MB and "
        "reloads with the same outputs. It is not a claim that this sample improves the model — the model already trained on "
        "this dataset — nor that 12 scenes measure its skill.\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 12 test scenes from a handful of HLS tiles, no dispersion "
        "estimate, pixel-pooled metrics that let large burns dominate, and labels derived from a burned-area product with their "
        "own uncertainty at scar edges and under smoke. Nothing here measures the model outside the contiguous United States, "
        "outside 2018–2021, on scenes larger than a chip, or on burn severity.\n\n"
        "Three things to carry to real data. **The six bands and their scaling are the contract:** blue, green, red, narrow "
        "NIR, SWIR 1, SWIR 2 in that order, surface reflectance in [0, 1]; a different band order or an uncorrected product is "
        "segmented without complaint and silently wrong. **Split by fire or tile, not by scene:** neighbouring scenes of one "
        "fire are near-duplicates, and a random split makes memorisation look like skill. **Read the baseline first:** on a "
        "scene with 5 % burn the not-burned baseline is 95 % accurate; only the burn-scar IoU, precision and recall say whether "
        "the model did anything.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without "
        "executing anything outside the audited allow-list, rebuild the model from the installed package, fetch a digest-pinned "
        "tarball and extract exactly the pinned labelled scenes, execute bounded fine-tuning, evaluate against a baseline and the "
        "frozen model on held-out scenes, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or burn-mapping skill beyond the checks "
        "shown.\n\n"
        "## Optional experiment: Predict → Change → Run → Observe → Explain\n\n"
        "None of this affects the default path, and every adaptation starts from the pinned base, so each run is a fresh experiment rather than continued "
        "training. **Scope of a re-run:** change the form field in Section 6, then run Sections 6, 7 and 8 in that order (Section 5 need not be re-run; if you do, "
        "it restores the base and prints `restored_pinned_base`). Pick one:\n\n"
        "1. **Learning rate.** *Predict:* with `LEARNING_RATE = 1e-4` (ten times the default) on a model that already trained on this dataset, will any epoch beat "
        "the frozen model's validation loss (0.1072 in the record), or will epoch 0 be kept? *Change* the field, *run* 6–8, *observe* `best_epoch`, the per-epoch "
        "`val_loss` and the verdict, and *explain* the result in terms of what a larger step does to a model near a minimum. There is no recorded outcome for "
        "this setting; your run is the evidence.\n"
        "2. **Scope.** *Predict:* does `TRAINABLE = 'decoder+last_block'` (32.9 M parameters) change the validation loss curve or the precision/recall trade? "
        "*Observe* `trainable_parameters`, `gpu_peak_gb` and the comparison rows.\n"
        "3. **Epochs.** *Predict:* with `EPOCHS = 10`, where does the validation loss bottom out, and does the kept epoch change? *Observe* the drift after the "
        "minimum and whether the verdict moves.\n"
        "4. **Your own scenes.** Set `USE_BYOD = True` with `BYOD_PATH` (a `group` column keeps one fire in one role) and re-run from Section 4: read the "
        "not-burned baseline before the adapted number, and compare the frozen column — the packaged model — with the adapted one.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 3 stops during the pickle audit or '
        'conversion** — the audit refuses any global outside the allow-list and names it; the downloaded checkpoint is not the pinned one. Delete the snapshot '
        "folder and run Section 3 again.\n- **Section 4 stops on the tarball's size or SHA-256** — the 2.6 GB download was cut short or altered. Run Section 4 "
        'again: a truncated download is fetched again, and scenes already extracted and verified are reused. If the digest still fails, delete '
        '`weights/hls-burn-scars/` and run it once more.\n- **CUDA out of memory in Section 6** — another notebook holds the GPU, or `TRAINABLE = '
        "'decoder+last_block'` with a larger `BATCH_SIZE` exceeds a T4. Restart the session, keep `BATCH_SIZE = 2`, and choose **Run all**.\n- **Section 7 shows no figure** — the figure is also written to `outputs/{stem}_test_scenes.png`; on Jupyter make sure the notebook is trusted.\n- **Section 5 "
        'prints `restored_pinned_base`** — you re-ran it after Section 6; the adapted tensors were put back to the base. Re-run Sections 6–8 in order.\n- '
        '**BYOD: a band, shape or label refusal** — the message names the rule; chips must be six-band 512 × 512 GeoTIFFs in the documented band order with '
        'masks of 0 / 1 / −1, and the dataset needs some burn-scar pixels.\n- **BYOD: "… bring at least 7 chips"** — the seeded split takes 25 % for test and 20 % '
        'for validation and must leave 4 training chips, so 7 distinct labelled chips is the minimum (more if a `group` column puts many chips in one group).\n- '
        '**BYOD: "pairs.csv row … is listed but not in the zip"** or **"… is not a readable GeoTIFF"** — the message names the row and the file; fix the name in '
        '`pairs.csv` or replace the file. `outputs/{stem}_byod_example/` is a complete, loadable example of the layout.\n- **BYOD: "chips have no \'group\'"** — '
        'either give every row a `group` value (a fire or tile id) or leave the column out.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working '
        'directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip (or folder) in the '
        'runtime and set `BYOD_PATH` to its path.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the '
        "cell again.\n\n## Glossary\n\n- **HLS (Harmonized Landsat Sentinel-2):** NASA's surface-reflectance product that puts Landsat 8/9 and Sentinel-2 on one 30 "
        'm grid; the six bands here are blue, green, red, narrow NIR, SWIR 1 and SWIR 2.\n- **Surface reflectance:** the fraction of sunlight a surface reflects '
        'in a band, after atmospheric correction; stored in [0, 1] or × 10 000.\n- **Burn scar:** ground whose vegetation was removed or charred by fire; it '
        'darkens NIR and brightens SWIR, which is what the model keys on.\n- **Segmentation mask / ignore class:** one class per pixel; −1 marks no-data pixels '
        'that are excluded from training and scoring.\n- **Not-burned baseline:** predicting *not burned* for every pixel; its accuracy equals the unburned '
        'fraction and its burn-scar IoU is 0.\n- **IoU, precision, recall, F1:** overlap of predicted and true burn pixels; the share of predicted burn that is '
        'real; the share of real burn found; their harmonic mean.\n- **Encoder, neck, decoder, head:** the ViT-L backbone that turns patches into features; the '
        'pyramid that rescales them; the U-Net that upsamples them to pixels; the final two-class layer.\n- **Frozen / adapted:** the packaged model as '
        'downloaded / after Section 6 trained the neck, decoder and head.\n- **Pinned base:** the verified packaged weights; every adaptation starts from them (`restore_base`).\n- '
        '**Group (fire or tile) split:** assigning every chip of one fire or HLS tile to one role, so near-duplicate neighbours cannot sit in both training and test; the `group` column of `pairs.csv`.\n- **Precision / recall trade:** a model can raise precision (fewer false burned pixels) while lowering recall (more missed burned pixels) with the IoU barely moving; Section 7 shows both.\n- '
        '**BatchNorm statistics:** running means and variances inside the decoder; kept fixed because two-scene batches would corrupt them.\n- **Validation-loss '
        'selection:** keeping the epoch with the lowest validation loss, with the frozen model as epoch 0.\n- **Pickle audit / safetensors:** the upstream '
        'checkpoint is a pickle that could run code when loaded; it is statically checked against an allow-list and converted once to safetensors, a format '
        'that stores only tensors.\n- **Isolated environment:** the separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## '
        "Conclusion (your notes)\n\nComplete these in your own words; the recorded run's values are in the **Check your reasoning** answers above.\n\n- The frozen "
        "model's test burn-scar IoU was ___ against the not-burned baseline's ___.\n- Bounded fine-tuning moved it to ___ (verdict: ___) and moved precision ___ and recall ___, which I read as ___.\n- "
        'The number I would not trust on its own is ___, because ___.\n- Before adapting on my own scenes I would check the band order and scaling, split by '
        '___, and compare against ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/prithvi-burnscar-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/prithvi-burnscar-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/prithvi-burnscar-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars (revision `{MODEL_REVISION}`)\n"
        "- HLS Burn Scars dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars (CC BY 4.0)\n"
        "- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732: https://arxiv.org/abs/2412.02732\n"
        "- TerraTorch: https://github.com/IBM/terratorch\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
