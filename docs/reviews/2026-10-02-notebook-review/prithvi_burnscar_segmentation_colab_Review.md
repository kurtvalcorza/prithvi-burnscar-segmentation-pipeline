# Prithvi-EO-2.0 Burn-Scar Segmentation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/prithvi-burnscar-segmentation-pipeline`  
**Notebook:** `tutorials/prithvi_burnscar_segmentation_colab.ipynb`  
**Reviewed commit:** `7361e2823c1ee02f7777934b2ddd9e958132f237` (`main`, confirmed with `gh api repos/kurtvalcorza/prithvi-burnscar-segmentation-pipeline/commits/main`)  
**Notebook Git blob:** `d8aa27bf67202be8c252afc0c4c83b968b0207ea`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-25 (commit `f5d329c`); the only later commit (`84d4dc2`) changes documentation, not the notebook or the carried modules.  
**Finding prefix:** `BS`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The engineering is careful and the prose is unusually honest. The notebook statically audits the pickled Lightning checkpoint against a four-global allow-list, unpickles it once through torch's weights-only loader into a digest-pinned safetensors file, streams exactly 88 digest-pinned members out of a digest-pinned 2.6 GB tarball without `extractall`, scores every model number against a not-burned baseline on the same pixels, labels the softmax scores uncalibrated, says plainly that the model already trained on this dataset, selects the adapted epoch on validation loss with the frozen model as epoch 0, asserts only what the procedure guarantees, and reloads the exported adapter into a fresh pipeline with exact parity. The validation-loss story in Section 6 matches the record (0.1072 → 0.0984 at epoch 1, then 0.1034 / 0.1034 / 0.1051).

| Measure | This review (CPU, no weights) | Kaggle T4 record (blob `d8aa27bf`) |
|---|---|---|
| Code cells completed | none of the model cells; carried modules exercised through the offline suite (41 passed) | 10/10 on pass 2; pass 1 stopped at the install guard |
| Test burn-scar IoU, baseline / frozen / adapted | not run | 0 / 0.9251 / 0.9263 (precision 0.9528 → 0.9741, recall 0.9695 → 0.9498) |
| "New scenes" in Section 8 that are actually new | **0 of 3**: one is byte-identical to sample test scene `T10SEH.2018190.v1`, two are in the checkpoint's training split (digest + split-file check) | burn fraction on that scene 0.7891; its label says 0.791 |
| Sample test scenes sharing an HLS tile with sample training scenes | **2 of 12** (`T10TFQ`, `T10TGS`) | tile lists printed, overlap not commented |
| BYOD smallest dataset accepted | **7 chips** (stated minimum of 4 refused; 5 and 6 refused too) | not run |
| Model state when BYOD is re-run as instructed | **not reset** (source + stub probe): Section 5 "frozen" and epoch 0 "frozen model" are the sample-adapted model | not run |

Three problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (BS-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart. `docs/release-verification.md` calls the restart "expected", and the repository marks the blob `Release-grade` on that run.
2. **BYOD, followed as written, compares against a model that is not frozen (BS-M2).** `adapt()` trains from whatever state the model is in and always labels epoch 0 "frozen model". The documented BYOD route ("set `USE_BYOD = True` in Section 4 and re-run from that cell") never reloads the base model, so Section 5's "frozen" numbers, Section 6's epoch 0 and Section 7's frozen column are all the sample-adapted model. Section 7 calls that frozen-versus-adapted gap "the number to watch" for the learner's own scenes.
3. **Guided layer largely absent (BS-M3).** The notebook is declared `GUIDED`, but it has no audience statement, how-to-use section, roadmap, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template. 1,794 lines of carried modules sit in three unlabelled, uncollapsed cells.

The pattern matches the sibling E2E notebooks generated from the same template family (pixart PX-M1/M3/M4, flux FS-M1/M4). There is no inverted "ceiling" here and no loss-trend error.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. The Prerequisites (cell 1, "Knowledge") assume the reader knows multispectral surface reflectance, bands, scaling and no-data, pixel-wise masks with an ignore class, and how IoU / precision / recall are read against a majority baseline |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12)"; about 6 GB of disk; "about 2.5 GB of GPU memory" for the default adaptation |
| Promised outcomes | Pinned install; carried package; snapshot staged and digest-verified; pickle audited and converted once; 44 pinned scenes extracted, validated and assigned the upstream roles (24 / 8 / 12) with three refusals; frozen model vs not-burned baseline; bounded fine-tuning of neck, decoder and head (20.3 M parameters); paired held-out comparison; segmentation of "new scenes"; safetensors adapter export and fresh reload with parity; BYOD zip through the same contract |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `3a58f89` |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 23 cells (10 code). Cells 5, 7 and 9 carry `pipeline.py` (972 lines), `metrics.py` (76 lines) and `samples.py` (746 lines). Also read: `pipeline.py` (`_check_record`, `validate_dataset`, `adapt`, `predict`, `evaluate`), `samples.py` (`SAMPLE_RECORDS`, `fetch_corpus`, `read_corpus`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`), the upstream `splits/*.txt` and `dimer-base-manifest.json`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` and `STATUS.md`. The repository has no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`, row 2026-09-25, and the workspace archive it cites (`.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-burnscar-segmentation/v2/evidence/`: `run_summary.json`, `executed.ipynb`, `executed-pass1.ipynb`, `outputs/`). The run was on a Kaggle Tesla T4 with **the reviewed blob** (SHA-1 verified before execution) and a clean Hugging Face cache. Pass 1 (22:38:07Z) raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 (22:41:39Z) completed 10/10; per-cell times 30 / 66 / 71 / 5 / 19 / 4 / 18 s for the install, model, data, frozen, adapt, evaluate and export cells.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), the shared `eo-notebook-test` conda env (Python 3.12.14, torch 2.13.0+cpu, numpy 2.5.3, tifffile 2026.9.20; no `terratorch`; nothing installed). No Prithvi weights and no tarball were downloaded: the 1.3 GB checkpoint and 2.6 GB tarball are out of scope for a CPU review host.
  - **Probes (about 47 s in total):**
    - P1: notebook parse, every code cell compiles, guided-layer markers, display calls.
    - P2: `tools/build_notebook.py --check` (OK, byte-identical), `tools/validate_release_assets.py` (PASS), offline suite with `PYTHONPATH=src` (exit 0, 41 passed, `test_adaptation.py` included with its stub model).
    - P3: sample roles vs HLS tiles; the three upstream example scenes vs the upstream split files and the pinned sample digests.
    - P4: the carried `load_byod_dataset` + `split_dataset(seed=0)`, exactly as Section 4 calls them, on 11 synthetic zips.
    - P5: the exported `…_sample_pairs.csv` file names vs the files the notebook writes or extracts.
    - P6: `adapt()` called twice on a stub torch model (CPU, synthetic chips) to see what epoch 0 of the second call is.
    - P7: identity, passes, pass-1 error and Section 5/7/8 outputs from the archived run.
- **Not verified:** any notebook cell executed end to end here; any Colab run (the stated runtime has no record); the real upload dialog; BYOD beyond the load/split stage on real weights; the optional experiments (`decoder+last_block`, more epochs, `LEARNING_RATE = 1e-4`); the "about 2.5 GB of GPU memory" figure.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| **Technical correctness** | Strong on the default path: provenance, conversion, extraction, metrics, adaptation selection and reload are sound and recorded. Two state defects: the install needs a restart (BS-M1), and a rerun of Section 4 or 6 adapts from the already-adapted model while labelling it frozen (BS-M2, BS-m5). BYOD error handling is partly unactionable (BS-m1). |
| **Promise fulfilment** | The default promises are delivered on the recorded run, except that the "new scenes" of Section 8 are not new (BS-m2) and the BYOD comparison is not frozen-vs-adapted (BS-M2). |
| **Learner experience** | The prose explains what each stage does and has "Look for" notes with recorded values, but there is no guided layer (BS-M3), no image or mask is ever shown in a segmentation tutorial (BS-m6), and the interpretation glosses over a precision/recall trade (BS-m5). |
| **Spec conformance** | Open `MUST`s: RUN1, RUN10, ENV6, REL2/REL11 (BS-M1); DAT14 and REL12 (BS-M2, for the BYOD comparison); DAT12, DAT19 (BS-m1, BS-m4); SPL5 (BS-m3, BYOD); UNC4 (BS-m7). Open `SHOULD`s: GDL1–GDL14, UX8 (BS-M3); UX3, UX11 (BS-m6); SPL10 (BS-m3); EXE5 (BS-S3). |

## 3. Findings

### BS-M1 — Major: `Run all` needs a manual restart after the install cell

**Location:** Section 1, install cell (cell 3); `docs/release-verification.md` procedure step 4 and Current status; generator `tools/notebook_template.py` (install cell).

**Observed issue:** The install cell `pip install`s the pins into the running kernel and then raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` whenever a pin replaced an already-imported distribution. In a fresh Kaggle T4 image that always happens (`numpy` 2.0.2 preloaded vs pin 2.5.3, `cuda-bindings` 12.9.4 vs 13.4.3).

**Consequence:** A learner who chooses `Run all` stops at cell 3 after about 3.5 minutes of installation and must restart and run again. The release record treats this as expected and marks the blob `Release-grade` on a two-pass run, which RUN10 and ENV6 forbid.

**Evidence:** Documented execution evidence: `executed-pass1.ipynb` cell 3 error text above; `run_summary.json` passes `[{attempt 1, ok false}, {attempt 2, ok true}]`. Source inspection: the guard in cell 3. Release record: "10/10 ok (1 restart after install cell)"; procedure step 4 "an interpreter restart after the install is expected".

**Recommended correction:** Replace the in-kernel install with the uv isolated-environment pattern: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Make the change in `tools/notebook_template.py` and regenerate; remove "restart is expected" from the release procedure and return the status to `Candidate` until a one-pass run is recorded.

**Acceptance check:** On a fresh Colab or Kaggle GPU runtime, `Run all` on the regenerated blob completes every code cell in one pass with no error output and no restart; the release record shows one pass for that blob.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### BS-M2 — Major: the documented BYOD rerun compares against the sample-adapted model labelled "frozen"

**Location:** Opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Section 4 (cell 13) BYOD branch; Section 5 (cell 15); Section 6 (cell 17) and `PrithviBurnScarPipeline.adapt` in `pipeline.py` (epoch 0 note `"frozen model"`); Section 7 (cell 19) markdown "the gap between frozen and adapted is the number to watch".

**Observed issue:** `adapt()` trains the neck/decoder/head from their current values and keeps the best epoch in place; nothing restores the base tensors, and Section 4 does not reload the model. After the default run, the BYOD rerun therefore (a) scores the sample-adapted model in Section 5 as `frozen_test` / `frozen_val`, (b) starts Section 6 from it and labels its epoch 0 "frozen model", and (c) reports Section 7's `frozen` column and the evaluation-report `frozen` block from it. The exported adapter's history repeats the label.

**Consequence:** On the learner's own scenes, the comparison the notebook tells them to read is between two adapted models. The recorded sample adaptation was small (1 kept epoch at 1e-5), so the gap is likely understated rather than reversed, but if the learner first tried the optional `LEARNING_RATE = 1e-4` or more epochs, the "frozen" baseline can be substantially different from the packaged model. The learner cannot tell.

**Evidence:** Source inspection of `adapt()` (no reset; `initial_state` is restored only on an exception). Direct execution (P6, stub torch model on CPU, synthetic chips): after one `adapt()`, a second `adapt()` reports epoch 0 note `"frozen model"` with val loss 0.3179, equal to the first call's kept epoch and not the true frozen 0.6931; a Section-5-style `evaluate` between the calls returns burn IoU 0.886 where the true frozen stub scores 0.0.

**Recommended correction:** Snapshot the trainable tensors right after `from_pretrained` and restore them at the start of the BYOD branch (or give `adapt()` a `reset_to_base=True` default that reloads them from the verified safetensors), and make epoch 0's note state what it is ("frozen base" only when the tensors equal the base). Alternatively, instruct "Runtime → Restart and run all with `USE_BYOD = True`". Fix in `pipeline.py` and `tools/notebook_template.py`, then regenerate.

**Acceptance check:** After a complete default run, setting `USE_BYOD = True` and re-running from Section 4 as documented gives Section 5 frozen metrics on the BYOD test split identical to a fresh-runtime BYOD run, and Section 6 epoch 0's validation loss equals that fresh run's epoch 0.

**Spec:** DAT14, REL12 (and GDL10 for the rerun instruction).

### BS-M3 — Major (learner-facing): the declared `GUIDED` layer is largely absent

**Location:** Whole notebook; carried module cells 5, 7, 9; generator `tools/notebook_template.py`.

**Observed issue:** The notebook declares mode `GUIDED`, but has no intended-audience statement (GDL1), no **How to use this notebook** (GDL2), no roadmap (GDL3), no Input → Model → Output contract near the opening (GDL4; the data contract in the Prerequisites is the closest), no glossary although IoU, ignore class, surface reflectance, HLS, pyramid neck, BatchNorm statistics, float16 autocast, GradScaler and safetensors appear (GDL6), no prediction before any principal result (GDL7), no interpretation checkpoints or sample answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10; the optional experiments are one sentence), no Infrastructure labelling or collapsed carrier cells (GDL11; 1,794 lines in three cells with empty metadata), no troubleshooting (GDL13) and no conclusion template (GDL14). Sections end without a synthesis (UX8). "Look for" notes exist for Sections 4–6 (GDL8 partly met).

**Consequence:** A self-paced learner meets 1,800 lines of carrier code before any lesson, gets no help separating essentials from infrastructure, and is never asked to predict, check or explain anything, so the objectives "read pixel IoU … against a not-burned baseline" and "compare the adapted and frozen models" are exercised only by reading printed dictionaries.

**Evidence:** Source inspection; P1 markers (`How to use`, `Glossary`, `Troubleshoot`, `Infrastructure`, `Check your reasoning`, `audience` all absent; `cellView` unset on all 10 code cells).

**Recommended correction:** Add the NOTEBOOK_SPEC 2.2 guided layer in `tools/notebook_template.py`: audience, how-to-use, roadmap, task contract, collapsible glossary, a prediction before Sections 5 and 7 (for example "will the adapted model beat a frozen model that already trained on this dataset?"), "What to notice" notes, collapsible "Check your reasoning" answers, one bounded PCROE activity (for example `TRAINABLE = 'decoder+last_block'`), `# @title Infrastructure: …` with `cellView: form` on the carrier cells, a troubleshooting section (install/restart, Hub download, disk, GPU memory, BYOD errors), and a conclusion template.

**Acceptance check:** The GDL1–GDL14 checklist in NOTEBOOK_SPEC 2.2 passes item by item on the regenerated notebook, and the three carrier cells open collapsed in Colab.

**Spec:** GDL1–GDL14, UX8.

### BS-m1 — Minor: BYOD stated minimum is wrong and several failures are not actionable

**Location:** Opening cell and Section 4 ("at least four chips with some burn scar"); `split_dataset` and `load_byod_dataset` in `samples.py`; cell 13 upload branch.

**Observed issue:** `split_dataset(seed=0)` needs at least 4 training chips after taking 25 % for test and 20 % for validation, so 4, 5 and 6 chips are refused and 7 is the true minimum. A `pairs.csv` row naming a file missing from the zip raises a bare `KeyError: 'c0.tif'`; a non-TIFF file raises `TiffFileError: not a TIFF file`; cancelling the upload dialog raises `StopIteration` from `next(iter(uploaded.items()))` (source). The structural refusals (band count, label values, no burn pixels, no `pairs.csv`, non-zip) are clear.

**Consequence:** A learner following the stated contract with 4–6 chips is refused after uploading, and three common mistakes give errors that do not name the failed contract or the fix.

**Evidence:** Direct execution (P4): `valid_n4/5/6` → `ValueError: split leaves 2/3/3 training chips; at least 4 are required`; `valid_n7` → 4/1/2; `csv_names_missing_member` → `KeyError`; `non_tiff_image` → `TiffFileError`. Source inspection for the cancel path.

**Recommended correction:** State "at least 7 labelled chips" (or derive the minimum from the fractions and print it), and wrap member lookup and TIFF decode in `load_byod_dataset` with `ValueError`s naming the row id, file and expected format; handle an empty upload with a message.

**Acceptance check:** A 7-chip zip is accepted and a 6-chip zip is refused with a message giving the minimum; a missing member, a non-TIFF image and a cancelled upload each produce a `ValueError` naming the row/file and the corrective action.

**Spec:** DAT12, DAT19, UX10.

### BS-m2 — Minor: the "new scenes" of Section 8 are not new

**Location:** Section 8 heading and markdown (cell 20), cell 21; objective "segment new scenes" (cell 0).

**Observed issue:** The three upstream example scenes are presented as new scenes that "carry no labels here". By digest, `T10SEH.2018190.v1` is byte-identical to sample test scene `test-001` (labelled; burn label 0.791, frozen predicted 0.7972, adapted 0.7891 in the record). The other two, `T10SFF.2018190.v1` and `T10SGF.2020217.v1`, are listed in the checkpoint's own `splits/train.txt`. None is in the 24 tutorial adaptation scenes, so INF2 is met in the letter.

**Consequence:** A learner reading the predicted burn fractions as a generalisation sanity check is looking at a scene already scored in Section 7 and two scenes the packaged model was trained on.

**Evidence:** Direct execution (P3, digest and split-file comparison); documented execution evidence (Section 5 and Section 8 outputs).

**Recommended correction:** Either say exactly what these scenes are (one test scene, two from the checkpoint's training split, all from the same dataset) or segment scenes that are outside the checkpoint's training and the sample's test split (for example pinned `val` scenes not used in the sample, or a pinned out-of-dataset HLS chip); and for the labelled one, show its label agreement.

**Acceptance check:** The Section 8 prose names the provenance of each segmented scene, and no scene described as new appears in `splits/train.txt` or among the 44 sample records.

**Spec:** INF2 (spirit), GDL8.

### BS-m3 — Minor: spatial split boundary not preserved in BYOD; sample tile overlap unstated

**Location:** `split_dataset` (`samples.py`), Section 4 BYOD branch; Section 4 output tile lists; Interpretation ("Split by fire or tile, not by scene").

**Observed issue:** The BYOD path is a seeded shuffle by scene with no grouping key; `pairs.csv` has no group or split column, so a learner cannot follow the notebook's own advice inside the notebook. In the sample, 2 of 12 test scenes (`T10TFQ.2018245`, `T10TGS.2018245`) share an HLS tile with training scenes (`T10TFQ.2019245`, `T10TGS.2018190`); the notebook prints the tile lists but does not mention the overlap.

**Consequence:** BYOD held-out numbers on scenes cut from one fire or tile will look better than they are, which is the exact failure the interpretation warns about. Spec conformance: SPL5 is a `MUST` for spatial data.

**Evidence:** Source inspection (`split_dataset`, `load_byod_dataset` read only `id,image,label`); direct execution (P3).

**Recommended correction:** Accept an optional `group` (or `split`) column in `pairs.csv` and split by group when present (default to group = tile/fire, else warn); add one sentence to Section 4 stating that 2 of the 12 sample test scenes share a tile with training scenes (an upstream split choice).

**Acceptance check:** A BYOD zip with a `group` column never puts one group in two roles; the Section 4 markdown states the sample's tile overlap.

**Spec:** SPL5, SPL10 (SPL2 for a `split` column).

### BS-m4 — Minor: the exported sample pairs table does not match any written file

**Location:** Section 4 (cell 13) `write_sample_pair` and `write_dataset_csv`; `samples.py`.

**Observed issue:** The notebook says the written sample pair and `outputs/prithvi_burnscar_segmentation_sample_pairs.csv` show "the BYOD shape". The CSV names files `T10SDH.2020248.v1_merged.tif` / `T10SDH.2020248.v1.mask.tif`, which match neither the extracted chips (`subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif`) nor the written pair (`prithvi_burnscar_segmentation_sample_chip.tif` / `_sample_label.tif`). Only one pair is written, below the 7-chip BYOD minimum.

**Consequence:** A learner who zips the outputs as a template gets a dataset that `load_byod_dataset` cannot read (`KeyError`), so the example does not demonstrate a working BYOD zip.

**Evidence:** Direct execution (P5): all three name checks `false`.

**Recommended correction:** Make `write_dataset_csv` name the files that are actually written (or write the 12 test pairs under the CSV's names), and add one sentence on how to build a valid zip from them.

**Acceptance check:** Zipping `outputs/` sample files with the exported CSV yields a zip that `load_byod_dataset` reads without error.

**Spec:** DAT12.

### BS-m5 — Minor: interpretation and optional experiments overstate or pre-state results

**Location:** Interpretation (cell 22); Section 6/7 markdown; Prerequisites (cell 1).

**Observed issue:** (a) The interpretation says adaptation "leaves those numbers where they were", while the record shows precision 0.9528 → 0.9741 and recall 0.9695 → 0.9498, a trade the notebook never points out. (b) "try `LEARNING_RATE = 1e-4` to see the frozen model win every epoch" states a result with no record, and on a rerun of Section 6 epoch 0 is no longer the frozen model (BS-M2 mechanism). (c) The optional experiments give no rerun scope (which cells, in which order). (d) The data contract reads `{{id, image, label}}` (template brace escape leaked into markdown).

**Consequence:** The learner is told nothing moved while two of the five printed metrics moved by two points in opposite directions, and is promised an experimental outcome that is not established.

**Evidence:** Documented execution evidence (Section 7 output); source inspection; P1 (`brace_typo_present: true`).

**Recommended correction:** Name the precision/recall trade and what it means for burned-area mapping; phrase the learning-rate experiment as a prediction to test; state the rerun scope (Restart and run all, or reset then Sections 6–8); fix the brace escape in the template.

**Acceptance check:** The interpretation mentions the precision and recall change; no optional experiment states its outcome as fact; each experiment names the cells to rerun; the rendered markdown shows `{id, image, label}`.

**Spec:** GDL8, GDL10, GDL14.

### BS-m6 — Minor: a segmentation tutorial that never shows an image or a mask

**Location:** Sections 4, 5, 7 and 8.

**Observed issue:** No code cell displays a scene, a label, a predicted mask or an error map. Masks are written to GeoTIFFs and summarised as burn fractions.

**Consequence:** The learner cannot see what a burn scar looks like in six-band data, where the model disagrees with the label, or what the ignore class covers, so IoU and recall stay abstract.

**Evidence:** Source inspection; P1 (`image_display_calls: 0`).

**Recommended correction:** Add a small figure: false-colour composite (SWIR 2 / NIR / red), label, frozen and adapted masks for two test scenes, and an error overlay.

**Acceptance check:** Running the default path displays at least one composite + label + prediction figure inline.

**Spec:** UX3, UX11.

### BS-m7 — Minor: threshold/calibration ownership is not stated, though the docs say it is

**Location:** Section 5 markdown (cell 14); `tutorials/README.md` conformance note "Score semantics (UNC1–UNC4)".

**Observed issue:** The notebook states the scores are uncalibrated and prints `decision_rule: argmax … (no threshold)`, but never says that a burned-area threshold and its cost trade-off belong to the deployment, or who owns calibration. `tutorials/README.md` claims "the notebook says the threshold and its cost trade-off are the deployment's to set".

**Consequence:** A learner reusing the masks operationally is not told that argmax at 0.5 is a default, not a tuned operating point; the conformance note overstates the notebook.

**Evidence:** Source inspection (no markdown mention of threshold or calibration ownership; P1 `threshold: false`).

**Recommended correction:** Add one sentence to Section 5: the decision rule is argmax; deployments choose a threshold on their own validation data and own calibration.

**Acceptance check:** Section 5 markdown states the default rule and that threshold choice and calibration are the deployment's; the README note then matches.

**Spec:** UNC3, UNC4.

### Suggestions

- **BS-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** The notebook, `metadata.dimer` and the docs declare 2.0. Acceptance: metadata and opening cell declare 2.2 and the validator checks 2.2.
- **BS-S2 — Record a Colab run.** Only Kaggle T4 runs exist although Colab is the stated runtime. Acceptance: a Colab row for the regenerated blob in `docs/release-verification.md`.
- **BS-S3 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it silently. Acceptance: a sentence in Section 1 names it and says it only skips the install. (EXE5)
- **BS-S4 — Show per-scene burn IoU for the 12 test scenes.** The prose rightly says pooled metrics let large burns dominate; a per-scene column (and its range) lets the learner see it. Acceptance: Section 5/7 print per-scene IoU for frozen and adapted. (EVAL6, ENV8)
- **BS-S5 — Back the GPU-memory figure.** Print `torch.cuda.max_memory_allocated()` after adaptation so "about 2.5 GB" is evidenced in each run. Acceptance: the value appears in the adapt output and the result JSON.

## 4. Readiness

**Needs revision.** No Blocker. Three Majors are open (BS-M1 one-pass `Run all`; BS-M2 BYOD frozen baseline; BS-M3 guided layer), and the open `MUST`s RUN1, RUN10, ENV6, REL2/REL11, DAT12, DAT14, DAT19, SPL5, UNC4 and REL12 fail release under the spec regardless of severity. The repository's `Release-grade` status rests on a two-pass run and should return to `Candidate`. Remaining gates after fixes: a one-pass hosted run of the regenerated blob (Colab preferred, BS-S2), and a BYOD run that reaches export and reload (REL12).

## 5. Verified vs inferred

- **Verified by direct execution (CPU, synthetic stand-ins):** generator parity, validator PASS, 41 offline tests; BYOD minimum of 7 and the unactionable errors; sample tile overlap and the example-scene identities (digests and split files); the exported CSV name mismatch; `adapt()` re-labelling an adapted start as "frozen model" (stub model).
- **Verified from documented execution evidence (Kaggle T4, this blob):** pass-1 restart error; default-path metrics, history and reload parity; Section 8 burn fraction on the scene identical to `test-001`.
- **Inferred, not executed:** the size of the BYOD frozen-baseline error on real weights (BS-M2); the behaviour of the optional experiments; the cancel-upload `StopIteration`; GPU memory.
- **Most likely to be wrong:** BS-M2's severity. The mechanism is certain from source and the stub probe, but the recorded sample adaptation moved test IoU by only 0.001, so on real data the mislabelled baseline may differ little from the true frozen model; a reader could rate it Minor. It stays Major because the notebook names this gap as the number to watch and the optional experiments can make the pre-BYOD state arbitrary.
