"""소규모 로컬 쌍, 후보 순위와 참조 정보 분리를 검증한다."""

import numpy as np
import pytest

from motion_blur.classical import linear_psf, synthesize_motion_blur
from motion_blur.image_io import save_image
from motion_blur.paired_data import load_manifest
from motion_blur.psf_search import search_psf

from tests.validate_motion_direction import textures


def test_local_manifest_paths_and_shape_validation(tmp_path):
    images = tmp_path / "scene"
    ref = np.full((24, 30, 3), 128, dtype=np.uint8)
    save_image(images / "blur.png", ref)
    save_image(images / "sharp.png", ref[:-1])
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("dataset,id,blur,sharp\nGoPro,a,scene/blur.png,scene/sharp.png\n"
                        "RealBlur-J,b,scene/blur.png,scene/sharp.png\n", encoding="utf-8")
    gopro, realblur = load_manifest(manifest)
    assert (gopro.dataset, realblur.dataset) == ("gopro", "realblur-j")
    assert gopro.blur == images / "blur.png"
    with pytest.raises(ValueError, match="해상도"):
        gopro.load()


def test_search_has_multiple_axes_and_lengths_and_supports_rl():
    observed = synthesize_motion_blur(textures()["grain"][:96, :128], linear_psf(15, 30))
    for method in ("wiener", "rl"):
        result = search_psf(observed, method=method, iterations=2)
        assert len({row.candidate.angle for row in result.scores}) > 1
        assert len({row.candidate.length for row in result.scores}) > 1
        assert result.selected.score == min(row.score for row in result.scores)
        assert result.restored.shape == observed.shape
        assert all(np.isfinite(row.score) for row in result.scores)


def test_evaluation_references_and_truth_labels_do_not_change_selected_candidate():
    pytest.importorskip("matplotlib")
    from motion_blur.candidate_experiment import evaluate_case

    reference = textures()["grain"][:96, :128]
    observed = synthesize_motion_blur(reference, linear_psf(15, 30))
    results = []
    for target, labels in (
        (reference, dict(dataset="synthetic", true_angle=30, true_length=15)),
        (255 - reference, dict(dataset="synthetic", true_angle=71, true_length=42)),
        (reference, dict(dataset="realblur-j")),
        (None, dict(dataset="unpaired")),
    ):
        rows, _ = evaluate_case(observed, target, labels, method="wiener", k=.005, iterations=2)
        selected = next(row for row in rows if row["role"] == "candidate" and row["selected"])
        results.append(selected)
        if "true_angle" not in labels:
            assert all("angle_error" not in row and "length_error" not in row for row in rows)
        if target is None:
            assert all("psnr_db" not in row and "ssim" not in row for row in rows)
    assert len({(row["angle"], row["length"], row["score"]) for row in results}) == 1
    assert results[0]["psnr_db"] != results[1]["psnr_db"]


def test_uniform_image_without_direction_abstains():
    flat = np.full((100, 128, 3), 120, dtype=np.uint8)
    result = search_psf(flat)
    assert result.selected is None and result.restored is None and not result.scores


def test_gopro_native_pairing_respects_sequence_filename_and_gamma(tmp_path):
    from motion_blur.paired_data import discover_gopro

    test = tmp_path / "GOPRO_Large" / "test"
    image = np.full((16, 24, 3), 128, dtype=np.uint8)
    for path in ("a/blur/001.png", "a/blur_gamma/001.png", "a/sharp/001.png",
                 "a/blur/002.png", "b/sharp/002.png",
                 "b/blur/003.png", "b/sharp/003.png"):
        save_image(test / path, image)
    pairs = discover_gopro(tmp_path, conditions=("blur", "blur_gamma"))
    assert [(p.condition, p.pair_id) for p in pairs] == [
        ("blur", "a/001.png"), ("blur", "b/003.png"), ("blur_gamma", "a/001.png"),
    ]
    assert pairs[0].sharp == pairs[-1].sharp == test / "a/sharp/001.png"
    assert discover_gopro(test.parent, conditions=("blur", "blur_gamma"), limit=1) == [pairs[0], pairs[-1]]
    assert discover_gopro(test) == pairs[:2]


def test_gopro_manifest_accepts_same_id_for_distinct_conditions(tmp_path):
    image = np.full((16, 24, 3), 128, dtype=np.uint8)
    for path in ("blur/001.png", "blur_gamma/001.png", "sharp/001.png"):
        save_image(tmp_path / path, image)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("dataset,id,blur,sharp\n"
                        "gopro,scene/001.png,blur/001.png,sharp/001.png\n"
                        "gopro,scene/001.png,blur_gamma/001.png,sharp/001.png\n", encoding="utf-8")
    assert [pair.condition for pair in load_manifest(manifest)] == ["blur", "blur_gamma"]
