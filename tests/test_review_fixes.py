"""Regression tests for the Notebook Review Framework v1 findings on prithvi_burnscar_segmentation_colab.ipynb
(review of 2026-10-02, prefix BS): BS-M1..M3 were fixed by the 2026-10-05 sweep (tests in test_sweep_fixes.py); this
file covers BS-m1..m7 and the trivial suggestions BS-S4/S5 taken on the way.

Every test needs only CI's dependencies (numpy, tifffile): synthetic 32..512-pixel chips stand in for HLS scenes and
no model runs. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_burnscar_segmentation_pipeline import (
    SAMPLE_RECORDS,
    byod_file_names,
    byod_minimum_records,
    example_provenance,
    false_colour_composite,
    load_byod_dataset,
    per_chip_burn_iou,
    split_dataset,
    write_byod_example,
    write_dataset_csv,
    write_sample_pair,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_burnscar_segmentation_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def matplotlib_stub() -> types.ModuleType:
    """A `matplotlib.pyplot` stand-in that records imshow calls (matplotlib is not a CI dependency). It rejects what
    the real API rejects on the calls the notebook makes: an image that is not 2-D or (H, W, 3/4)."""
    calls: list[dict] = []

    class Axis:
        def imshow(self, image, cmap=None, vmin=None, vmax=None, interpolation=None):
            array = np.asarray(image)
            if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)):
                raise TypeError(f"Invalid shape {array.shape} for image data")
            calls.append({"shape": array.shape, "cmap": cmap})

        def set_title(self, text, fontsize=None):
            calls.append({"title": str(text)})

        def set_axis_off(self):
            pass

    class Figure:
        def tight_layout(self):
            pass

        def savefig(self, path, dpi=None):
            Path(path).write_bytes(b"png-stand-in")
            calls.append({"saved": str(path)})

    def subplots(rows, cols, figsize=None, squeeze=False):
        return Figure(), np.array([[Axis() for _ in range(cols)] for _ in range(rows)], dtype=object)

    stub = types.ModuleType("matplotlib.pyplot")
    stub.subplots = subplots
    stub.show = lambda: calls.append({"shown": True})
    stub.calls = calls
    return stub


def _write_zip(path: Path, records, *, group=None, missing_member=False, bad_image=False, extra_csv_columns=()):
    rows = ["id,image,label" + ("".join(f",{c}" for c in extra_csv_columns))]
    with zipfile.ZipFile(path, "w") as archive:
        for i, record in enumerate(records):
            image_name, label_name = byod_file_names(record)
            rows.append(f"{record['id']},{image_name},{label_name}" + "".join(f",{group(i) if c == 'group' else ''}" for c in extra_csv_columns))
            if missing_member and i == 0:
                continue
            folder = path.parent / "chips"
            folder.mkdir(exist_ok=True)
            write_sample_pair(record, folder / image_name, folder / label_name)
            archive.writestr(image_name, b"not a tiff" if (bad_image and i == 0) else (folder / image_name).read_bytes())
            archive.writestr(label_name, (folder / label_name).read_bytes())
        archive.writestr("pairs.csv", "\n".join(rows) + "\n")
    return path


# --- BS-m1: the true minimum is stated and refused by name; failures name the row, the file and the fix ---------------


def test_bs_m1_minimum_is_seven_and_a_six_chip_set_is_refused_with_the_minimum():
    assert byod_minimum_records() == 7
    records = synthetic_records(7)
    splits = split_dataset(records, seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"test": 2, "validation": 1, "train": 4}
    with pytest.raises(ValueError, match=r"6 distinct labelled chips split into .* bring at least 7 chips"):
        split_dataset(records[:6], seed=0)
    with pytest.raises(ValueError, match="bring at least 7 chips"):
        split_dataset(records[:4], seed=0)


def test_bs_m1_missing_member_and_non_tiff_name_the_row_file_and_fix(tmp_path):
    records = synthetic_records(2)
    with pytest.raises(ValueError, match=r"pairs.csv row 'chip-000': image file 'chip-000_merged.tif' is listed but not in the zip; add the file"):
        load_byod_dataset(_write_zip(tmp_path / "missing.zip", records, missing_member=True))
    with pytest.raises(ValueError, match=r"pairs.csv row 'chip-000': image file 'chip-000_merged.tif' is not a readable GeoTIFF .*six-band 512 × 512"):
        load_byod_dataset(_write_zip(tmp_path / "bad.zip", records, bad_image=True))
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "pairs.csv").write_text("id,image,label\nx,x.tif,x_mask.tif\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is listed but not in the folder"):
        load_byod_dataset(folder)


def test_bs_m1_notebook_states_the_true_minimum(notebook):
    md = _markdown(notebook)
    assert "at least four chips" not in md and "at least **7** labelled chips" in md
    assert "bring at least 7 chips" in md
    assert "'minimum_chips': byod_minimum_records()" in _cell(notebook, "if USE_BYOD:")


# --- BS-m3: a `group` column never puts one group in two roles; the sample's tile overlap is stated -----------------


def test_bs_m3_group_column_keeps_every_group_in_one_role(tmp_path):
    records = synthetic_records(12)
    archive = _write_zip(tmp_path / "grouped.zip", records, group=lambda i: f"fire-{i % 4}", extra_csv_columns=("group",))
    loaded = load_byod_dataset(archive)
    assert all(r["group"] == r["region"] == f"fire-{i % 4}" for i, r in enumerate(loaded))
    for seed in range(5):
        splits = split_dataset(loaded, seed=seed)
        roles = {}
        for role, members in splits.items():
            for record in members:
                assert roles.setdefault(record["group"], role) == role, f"group {record['group']} in two roles"
        assert set(roles) == {f"fire-{i}" for i in range(4)} and len(splits["train"]) >= 4
    # ungrouped records still shuffle by chip, and a half-grouped table is refused
    assert sum(len(v) for v in split_dataset(records, seed=0).values()) == 12
    with pytest.raises(ValueError, match="chips have no 'group'"):
        split_dataset([{**r, "group": "a"} if i % 2 else r for i, r in enumerate(records)], seed=0)
    with pytest.raises(ValueError, match="at least 3 groups"):
        split_dataset([{**r, "group": "a" if i < 6 else "b"} for i, r in enumerate(records)], seed=0)


def test_bs_m3_section_4_states_the_sample_tile_overlap_and_reports_grouping(notebook):
    md = _markdown(notebook)
    assert "2 of the 12 test scenes (`T10TFQ.2018245`, `T10TGS.2018245`) share an HLS tile with training scenes" in md
    tiles = {name.split(".")[0] for name, role, *_ in SAMPLE_RECORDS if role == "train"} & {name.split(".")[0] for name, role, *_ in SAMPLE_RECORDS if role == "test"}
    assert tiles == {"T10TFQ", "T10TGS"}
    assert "'grouped_split': byod_grouped" in _cell(notebook, "if USE_BYOD:")


# --- BS-m4: the exported BYOD example loads back unchanged, as a folder and as a zip ----------------------------------


def test_bs_m4_written_byod_example_is_loadable_as_folder_and_zip(tmp_path):
    records = [{**r, "source_id": f"T10XXX.2020{i:03d}.v1", "region": "T10XXX", "source": "tar#member"} for i, r in enumerate(synthetic_records(7))]
    example = write_byod_example(records, tmp_path / "example")
    assert example["n_pairs"] == 7 and example["minimum_chips"] == 7
    names = {p.name for p in (tmp_path / "example").iterdir()}
    rows = (tmp_path / "example" / "pairs.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "id,image,label,group,source"
    assert all(row.split(",")[1] in names and row.split(",")[2] in names for row in rows[1:])
    loaded = load_byod_dataset(tmp_path / "example")
    assert [r["id"] for r in loaded] == [r["id"] for r in records] and all(r["group"] == "T10XXX" for r in loaded)
    assert np.array_equal(loaded[3]["label"], records[3]["label"]) and np.allclose(loaded[3]["image"], records[3]["image"], atol=1e-6)
    archive = tmp_path / "example.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in (tmp_path / "example").iterdir():
            zf.write(path, path.name)
    assert len(load_byod_dataset(archive)) == 7
    with pytest.raises(ValueError, match="at least 3 groups"):  # one tile for all 7: the grouped split refuses, naming the rule
        split_dataset(loaded, seed=0)
    assert write_dataset_csv(records[:1], tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()[1].startswith("chip-000,chip-000_merged.tif,chip-000.mask.tif,T10XXX,")


def test_bs_m4_section_4_writes_the_example_from_the_loaded_records(notebook):
    source = _cell(notebook, "if USE_BYOD:")
    assert "write_byod_example((test_records + val_records + train_records)[:byod_minimum_records()], 'outputs/prithvi_burnscar_segmentation_byod_example')" in source
    assert "sample_pairs.csv" not in source


# --- BS-m2: the example scenes are named for what they are -----------------------------------------------------------


def test_bs_m2_example_provenance_from_split_files_and_sample_digests(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("T10SFF.2018190.v1\nT10SGF.2020217.v1\n", encoding="utf-8")
    (splits / "val.txt").write_text("", encoding="utf-8")
    (splits / "test.txt").write_text("T10SEH.2018190.v1\n", encoding="utf-8")
    scene = tmp_path / "subsetted_512x512_HLS.S30.T10SFF.2018190.v1.4_merged.tif"
    scene.write_bytes(b"example bytes")
    prov = example_provenance(scene, splits)
    assert prov == {"scene": "T10SFF.2018190.v1", "sha256": hashlib.sha256(b"example bytes").hexdigest(), "upstream_split": ["train"], "sample_record": None, "new_to_the_checkpoint": False}
    new = tmp_path / "subsetted_512x512_HLS.S30.T99ZZZ.2030001.v1.4_merged.tif"
    new.write_bytes(b"other")
    assert example_provenance(new, splits)["new_to_the_checkpoint"] is True
    # the pinned sample digest of test scene T10SEH.2018190.v1 is what the byte-identical example resolves to
    record = next(r for r in SAMPLE_RECORDS if r[0] == "T10SEH.2018190.v1")
    assert record[1] == "test"
    monkey = {**prov, "sha256": record[4]}
    assert [r for r in SAMPLE_RECORDS if r[4] == monkey["sha256"]][0][0] == "T10SEH.2018190.v1"


def test_bs_m2_section_8_never_calls_the_example_scenes_new(notebook):
    md = _markdown(notebook)
    assert "## 8. New scenes" not in md and "segment new scenes" not in md
    assert "**None of them is new to the checkpoint**" in md and "byte-identical to sample test scene `test-001`" in md
    source = _cell(notebook, "new_predictions = pipe.predict(new_records)")
    assert "example_provenance(example_dir / record['source'], WEIGHTS_DIR / 'splits')" in source
    assert "'sanity check, no label'" not in source and "row['label_agreement'] = per_chip_burn_iou(" in source
    assert "'provenance': provenance[p['id']]" in source


# --- BS-m5: interpretation names the precision/recall trade; experiments predict, never pre-state; rerun scope ------


def test_bs_m5_interpretation_and_experiments(notebook):
    md = _markdown(notebook)
    assert "leaves those numbers where they were" not in md
    assert "precision 0.9528 → 0.9741, recall 0.9695 → 0.9498" in md
    assert "to see the frozen model win every epoch" not in md and "the frozen model (epoch 0) would win" not in md
    assert "## Optional experiment: Predict → Change → Run → Observe → Explain" in md
    assert "run Sections 6, 7 and 8 in that order" in md and "There is no recorded outcome for this setting" in md
    assert "{{id, image, label}}" not in md and "`{id, image, label}`" in md


# --- BS-m6: a figure of composite, label, masks and errors; BS-S4 per-scene IoU; BS-S5 GPU peak -----------------------


def test_bs_m6_false_colour_composite_is_swir2_nir_red_in_unit_range():
    image, _ = synthetic_chip(seed=3, size=32)
    composite = false_colour_composite(image)
    assert composite.shape == (32, 32, 3) and composite.dtype == np.float32
    assert composite.min() >= 0.0 and composite.max() <= 1.0 and composite.max() > 0.9
    scaled = false_colour_composite(image * 10000.0)
    assert np.allclose(scaled, composite, atol=1e-4)
    with pytest.raises(ValueError, match=r"expected a \(6, H, W\) chip"):
        false_colour_composite(image[:5])


def test_bs_s4_per_chip_burn_iou_matches_the_pooled_metric_on_one_chip():
    _, label = synthetic_chip(seed=0, size=32)
    pred = np.clip(label, 0, 1)
    rows = per_chip_burn_iou([pred, np.zeros_like(pred)], [label, label], ["a", "b"])
    assert rows[0]["id"] == "a" and rows[0]["iou"] == 1.0 and rows[1]["iou"] == 0.0
    assert rows[0]["labelled_pixels"] == 32 * 32 - 64
    assert per_chip_burn_iou([np.zeros_like(pred)], [np.zeros_like(label)])[0]["iou"] is None


def test_bs_m6_section_7_draws_the_scenes_and_prints_per_scene_iou(notebook, tmp_path, monkeypatch, capsys):
    from test_sweep_fixes import _section_7_namespace

    source = _cell(notebook, "delta_burn_iou = ")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    namespace = _section_7_namespace(0.8, 0.4)
    exec(compile(source, "<section 7>", "exec"), namespace)
    plt = sys.modules["matplotlib.pyplot"]
    shapes = [c["shape"] for c in plt.calls if "shape" in c]
    assert len(shapes) == 10 and shapes[0] == (32, 32, 3) and shapes[1] == (32, 32) and shapes[4] == (32, 32, 3)
    assert {"shown": True} in plt.calls and (tmp_path / "outputs" / "prithvi_burnscar_segmentation_test_scenes.png").is_file()
    out = capsys.readouterr().out
    assert "'frozen_burn_iou': 1.0, 'adapted_burn_iou': 1.0" in out and "'per_scene_burn_iou_range': {'frozen': [1.0, 1.0], 'adapted': [1.0, 1.0]}" in out
    report = json.loads((tmp_path / "outputs" / "prithvi_burnscar_segmentation_evaluation_report.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in report["per_scene_burn_iou"]["adapted"]] == ["chip-000", "chip-001"]


def test_bs_s5_adapt_cell_records_gpu_peak(notebook):
    source = _cell(notebook, "adapt_result = pipe.adapt(")
    assert "adapt_result['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None" in source
    assert "'adaptation_gpu_peak_gb': adapt_result['gpu_peak_gb']" in _cell(notebook, "result_payload = {")


# --- BS-m7: threshold and calibration ownership ---------------------------------------------------------------------


def test_bs_m7_section_5_states_the_default_rule_and_who_owns_the_threshold(notebook):
    md = _markdown(notebook)
    section_5 = md[md.index("## 5. The frozen model") : md.index("## 6. Bounded fine-tuning")]
    assert "equivalent to a 0.5 threshold on the burn-scar score" in section_5
    assert "a deployment chooses its own threshold on its own validation data" in section_5 and "owns the calibration" in section_5
