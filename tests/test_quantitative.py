"""라이브러리 비교, 참조 없음, 이어서 실행하는 캐시를 검증한다."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("matplotlib")

from motion_blur.classical import linear_psf, wiener_restore
from motion_blur.image_io import save_image
from motion_blur.quantitative import evaluate, library_restore, main, read_case


def test_wiener_reference_uses_same_transfer_and_quantization():
    image = np.random.default_rng(8).integers(0, 256, (32, 36, 3), dtype=np.uint8)
    psf = linear_psf(9, 30)
    direct = wiener_restore(image, psf, boundary="wrap")
    library = library_restore(image, psf, k=.005, iterations=20)
    np.testing.assert_array_equal(direct, library)


def test_real_case_has_no_reference_metrics():
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    rows, panels = evaluate(image, {"data_type": "captured_blurred", "angle": None, "length": None},
                            sweep=False, nafnet=None, naf_device="")
    assert all("psnr_db" not in row and "ssim" not in row for row in rows)
    assert [row["status"] for row in rows] == ["ok", "skipped", "skipped"]
    assert "Reference" not in panels


def test_partial_cache_resume_and_stale_configuration(tmp_path):
    inputs = tmp_path / "input"
    image = np.random.default_rng(9).integers(25, 225, (64, 64, 3), dtype=np.uint8)
    save_image(inputs / "public" / "source.png", image)
    output = tmp_path / "results"
    args = [str(inputs), str(output), "--quick"]
    assert main(args) == 0
    all_cases = list((output / "cases").iterdir())
    assert len(all_cases) == 2
    keep, remove = all_cases
    before = (keep / "result.json").read_bytes()
    shutil.rmtree(remove)
    partial = output / "cases" / ".partial-incomplete"
    partial.mkdir()
    (partial / "result.json").write_text("{", encoding="utf-8")
    assert read_case(output / "cases", partial.name) is None
    assert main([*args, "--resume"]) == 0
    assert (keep / "result.json").read_bytes() == before
    assert (remove / "result.json").exists()
    with (output / "results.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert {row["sigma"] for row in rows} == {"0", "0.01"}
    assert main([*args, "--resume", "--max-size", "63"]) == 1
    assert json.loads((output / "run.json").read_text())["max_size"] == 0
    source = inputs / "public" / "source.png"
    source.write_bytes(source.read_bytes() + b"changed bytes")
    assert main([*args, "--resume"]) == 1
