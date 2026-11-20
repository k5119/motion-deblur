"""자동 PSF의 좌표계, 보류 경로 및 복원 간 공유를 검증한다."""

import json

import numpy as np
import pytest

from motion_blur import auto_restore
from motion_blur.classical import apply_motion_blur, linear_psf
from motion_blur.image_io import save_image
from motion_blur.psf_estimator import estimate_psf, to_original_axis
from tests.validate_motion_direction import axial_error, textures


def test_axis_scaling_uses_endpoint_span_and_both_resize_ratios():
    angle, length = to_original_axis(45, 16, (512, 366), (1024, 733))
    assert 45 < angle < 45.1
    assert length == pytest.approx(31.0205, abs=0.001)
    assert to_original_axis(0, 16, (512, 366), (1024, 733)) == (0, 31)


def test_fft_recovers_known_large_image_psf_in_original_pixels():
    image = np.random.default_rng(260).integers(16, 240, (733, 1024, 3), dtype=np.uint8)
    blurred = apply_motion_blur(image, linear_psf(31, 45))
    result = estimate_psf(blurred)
    assert result.analysis_size == (512, 366)
    assert result.original_size == (1024, 733)
    assert result.length_analysis_px == 16
    assert result.length_original_px == pytest.approx(31, abs=1)
    assert axial_error(45, result.angle_deg) < 2


def test_directions_with_texture_and_flat_controls():
    images = textures()
    blurred = apply_motion_blur(images["grain"], linear_psf(15, 45))
    result = estimate_psf(blurred, method="gradient")
    assert axial_error(45, result.angle_deg) < 2
    assert result.length_analysis_px == 15
    flat = estimate_psf(images["flat"])
    assert flat.reason == "no_direction"
    assert flat.length_original_px is None


def test_cli_abstains_and_shares_one_manual_psf(monkeypatch, tmp_path):
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    source = save_image(tmp_path / "flat.png", image)
    automatic = tmp_path / "auto"
    assert auto_restore.main([str(source), str(automatic)]) == 1
    report = json.loads((automatic / "psf_estimate.json").read_text())
    assert report["status"] == "skipped" and report["estimate"]["reason"] == "no_direction"
    assert not (automatic / "wiener.png").exists()
    psfs = []

    def record(observed, psf, *args, **kwargs):
        psfs.append(psf)
        return observed

    monkeypatch.setattr(auto_restore, "wiener_restore", record)
    monkeypatch.setattr(auto_restore, "richardson_lucy_restore", record)
    manual = tmp_path / "manual"
    command = [str(source), str(manual), "--angle", "45", "--length", "9", "--k", "0.01", "--iterations", "5"]
    assert auto_restore.main(command) == 0
    assert len(psfs) == 2 and psfs[0] is psfs[1]
    report = json.loads((manual / "psf_estimate.json").read_text())
    assert report["applied"]["manual_angle"] and report["applied"]["manual_length"]
    assert auto_restore.main(command) == 1
