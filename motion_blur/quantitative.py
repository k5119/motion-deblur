"""동일 PSF의 직접 구현·scikit-image 복원 및 재개 가능한 정량 실험."""

import argparse
import csv
import hashlib
import importlib.metadata
from itertools import product
import json
import logging
import os
from pathlib import Path
import platform
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from skimage.restoration import richardson_lucy, wiener

from . import DEFAULT_WEIGHTS
from .batch import find_inputs, make_nafnet
from .classical import (
    compare_quality, linear_psf, richardson_lucy_restore, synthesize_motion_blur, wiener_restore,
)
from .image_io import Image, limit_image_size, load_image
from .psf_estimator import estimate_psf
from .wiener_sweep import K_VALUES

LOGGER = logging.getLogger(__name__)
LENGTHS = (7, 15, 25, 35)
ANGLES = (0, 30, 60, 135)
NOISE = (0, 0.005, 0.01, 0.02)
ITERATIONS = (5, 10, 20, 40)
SEED = 702
REPRESENTATIVE = (15, 30, 0.01)
ERRORS = (OSError, ValueError, RuntimeError, MemoryError, ImportError, cv2.error)
FIELDS = (
    "image", "data_type", "kind", "width", "height", "analysis_width", "analysis_height",
    "length", "angle", "sigma", "seed", "method", "psf", "boundary", "k", "iterations",
    "estimated_angle", "estimated_length", "angle_error", "length_error", "direction_score",
    "length_prominence", "status", "detail", "psnr_db", "ssim", "elapsed_ms", "device",
    "wrap_wiener_max_pixel_delta",
)


def fingerprint(data: dict) -> str:
    """결정적 JSON 표현에서 캐시 키를 만든다."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """전체 파일을 메모리에 올리지 않고 내용 해시를 구한다."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def version_info(include_nafnet: bool) -> dict:
    """소스, 수치 라이브러리, 장치 환경을 결과 키에 포함한다."""
    sources = {str(path.relative_to(Path(__file__).parent)): file_digest(path)
               for path in sorted(Path(__file__).parent.rglob("*.py"))}
    packages = {}
    for name in ("scikit-image", "scipy", "numpy", "torch", "opencv-python-headless", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = None
    if include_nafnet and packages["torch"] is not None:
        try:
            import torch

            cuda = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except (ImportError, OSError, RuntimeError):
            cuda = None
    return {"source": fingerprint(sources), "packages": packages, "host": platform.node(),
            "platform": platform.platform(), "python": platform.python_version(), "cuda_devices": cuda,
            "threads": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def read_case(cache: Path, key: str) -> list[dict] | None:
    """완료 메타데이터가 유효하고 결과가 온전히 게시된 경우에만 재사용한다."""
    folder = cache / key
    try:
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if data["key"] == key and all(
            file_digest(folder / name) == digest for name, digest in data["files"].items()
        ):
            return data["rows"]
    except (ValueError, OSError, KeyError, TypeError):
        pass
    return None


def library_restore(image: Image, psf: np.ndarray, *, k: float | None, iterations: int) -> Image:
    """동일한 정규화 PSF를 BGR 각 채널에 적용하고 같은 방식으로 uint8화한다."""
    channels = []
    for channel in range(3):
        plane = image[:, :, channel].astype(np.float64) / 255
        if k is not None:
            # 항등 정규화는 분모를 |H|² + K로 맞춘다.
            result = wiener(plane, psf, balance=k, reg=np.ones((1, 1)), clip=False)
        else:
            result = richardson_lucy(plane, psf, num_iter=iterations, clip=False)
        if not np.isfinite(result).all():
            raise ValueError("scikit-image 복원에 비유한 픽셀이 있습니다.")
        channels.append(np.clip(np.rint(result * 255), 0, 255).astype(np.uint8))
    return np.stack(channels, axis=-1)


def quality(reference: Image | None, image: Image) -> dict:
    """정답이 있는 경우에만 기존 PSNR/SSIM 함수를 호출한다."""
    if reference is None:
        return {}
    psnr, ssim = compare_quality(reference, image)
    return {"psnr_db": psnr if np.isfinite(psnr) else "inf", "ssim": ssim}


def evaluate(source: Image, info: dict, *, sweep: bool, nafnet, naf_device: str) -> tuple[list[dict], dict]:
    """동일 관측 영상에서 정답·추정 PSF와 라이브러리 복원을 비교한다."""
    real = info["data_type"] == "captured_blurred"
    true_psf = None if real else linear_psf(info["length"], info["angle"])
    observed = source if real else synthesize_motion_blur(
        source, true_psf, noise_sigma=info["sigma"], seed=info["seed"],
    )
    reference = None if real else source
    estimate = estimate_psf(observed)
    angle = estimate.angle_deg
    length = round(estimate.length_original_px) if estimate.length_original_px is not None else None
    estimated = linear_psf(length, angle) if length is not None and angle is not None else None
    info = {**info, "analysis_width": estimate.analysis_size[0], "analysis_height": estimate.analysis_size[1],
            "estimated_angle": angle, "estimated_length": length, "direction_score": estimate.fft.confidence,
            "length_prominence": estimate.length_prominence}
    if not real and angle is not None:
        info["angle_error"] = abs((angle - info["angle"] + 90) % 180 - 90)
    if not real and length is not None:
        info["length_error"] = abs(length - info["length"])
    rows = [{**info, "kind": "core", "method": "blurred", "status": "ok", **quality(reference, observed)}]
    panels = {"Input": observed}
    if reference is not None:
        panels = {"Reference": reference, **panels}
    wrap_output = None

    def record(method, psf_name, boundary, operation, *, k=None, iterations=None,
               kind="core", device="cpu", unavailable=None):
        nonlocal wrap_output
        row = {**info, "kind": kind, "method": method, "psf": psf_name, "boundary": boundary,
               "k": k, "iterations": iterations, "device": device}
        rows.append(row)
        if operation is None:
            row.update(status="skipped", detail=unavailable)
            if kind == "core":
                panels[f"{method} {psf_name}"] = None
            return
        start = perf_counter()
        try:
            result = operation()
            row.update(status="ok", elapsed_ms=(perf_counter() - start) * 1000, **quality(reference, result))
            if method == "own_wiener_wrap":
                wrap_output = result
            elif method == "skimage_wiener" and wrap_output is not None:
                row["wrap_wiener_max_pixel_delta"] = int(np.abs(result.astype(np.int16) - wrap_output).max())
            if kind == "core":
                panels[f"{method} {psf_name}"] = result
        except ERRORS as exc:
            row.update(status="failed", detail=str(exc))

    if true_psf is not None:
        record("own_wiener_wrap", "known", "wrap", lambda: wiener_restore(observed, true_psf, boundary="wrap"), k=.005)
        record("skimage_wiener", "known", "wrap", lambda: library_restore(observed, true_psf, k=.005, iterations=20), k=.005)
        record("skimage_rl", "known", "zero", lambda: library_restore(observed, true_psf, k=None, iterations=20), iterations=20)
        record("own_rl_wrap", "known", "wrap", lambda: richardson_lucy_restore(observed, true_psf, boundary="wrap"), iterations=20)
    for tag, psf in (("known", true_psf), ("estimated", estimated)):
        if tag == "known" and real:
            continue
        record("own_wiener", tag, "taper", None if psf is None else lambda: wiener_restore(observed, psf),
               k=.005, unavailable=estimate.reason)
        record("own_rl", tag, "taper", None if psf is None else lambda: richardson_lucy_restore(observed, psf),
               iterations=20, unavailable=estimate.reason)

    if sweep:
        for k in K_VALUES:
            record("own_wiener", "known", "taper", lambda: wiener_restore(observed, true_psf, k),
                   k=k, kind="k_sweep")
        for count in ITERATIONS:
            record("own_rl", "known", "taper", lambda: richardson_lucy_restore(observed, true_psf, count),
                   iterations=count, kind="rl_sweep")
    if nafnet is not None:
        record("nafnet", "", "", lambda: nafnet(observed), device=naf_device)
    elif naf_device:
        record("nafnet", "", "", None, device=None, unavailable=naf_device)
    return rows, panels


def main(argv: list[str] | None = None) -> int:
    """실험 조건을 고정하고 케이스별 완료 결과를 재사용한다."""
    from .quantitative_report import montage, save_reports

    parser = argparse.ArgumentParser(description="직접 구현·scikit-image 정량 복원 검증")
    parser.add_argument("input", type=Path, help="public, captured_sharp, captured_blurred 하위 폴더")
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-size", type=int, default=0, help="합성 전 긴 변 상한, 0은 원본")
    parser.add_argument("--resume", action="store_true", help="같은 설정으로 완료된 캐시 재사용")
    parser.add_argument("--nafnet", action="store_true", help="대표 조건과 실사에서만 비교")
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)
    if args.max_size < 0:
        parser.error("최대 크기는 0 이상이어야 합니다.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        inputs = find_inputs(args.input, False)
        if not inputs:
            raise ValueError("입력 이미지가 없습니다.")
        if args.output.exists() and not args.resume:
            raise FileExistsError(f"새 결과 폴더를 사용하거나 --resume을 지정하세요: {args.output}")
        versions = version_info(args.nafnet)
        weights = file_digest(args.weights) if args.nafnet and args.weights.is_file() else None
        sources = [(kind, path, file_digest(path)) for kind, path in inputs]
        config = {"versions": versions, "weights": weights, "max_size": args.max_size,
                  "quick": args.quick, "nafnet": args.nafnet, "device": args.device, "seed": SEED,
                  "inputs": [(kind, path.name, digest) for kind, path, digest in sources]}
        run_key = fingerprint(config)
        if args.output.exists():
            saved = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
            if saved["key"] != run_key:
                raise ValueError("입력, 설정 또는 코드가 변경되었습니다. 새 결과 폴더를 사용하세요.")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "run.json").write_text(json.dumps({"key": run_key, **config}, indent=2), encoding="utf-8")
        cache = args.output / "cases"
        cache.mkdir(exist_ok=True)
        all_rows = []
        comparisons = []
        reused = 0
        nafnet = None
        naf_device = ""
        initialized = False

        for kind, path, digest in sources:
            source = load_image(path)
            if args.max_size:
                source = limit_image_size(source, args.max_size)
            conditions = ((None, None, None),) if kind == "captured_blurred" else (
                ((15, 30, 0), REPRESENTATIVE) if args.quick else product(LENGTHS, ANGLES, NOISE)
            )
            for length, angle, sigma in conditions:
                representative = (length, angle, sigma) == REPRESENTATIVE or length is None
                sweep = length == 15 and angle == 30
                info = dict(image=path.name, data_type=kind, width=source.shape[1], height=source.shape[0],
                            length=length, angle=angle, sigma=sigma, seed=SEED if length is not None else None)
                key = fingerprint({"run": run_key, "image": digest, "case": info, "sweep": sweep})
                if representative:
                    comparisons.append((key, path.name))
                rows = read_case(cache, key)
                if rows is not None:
                    all_rows.extend(rows)
                    reused += 1
                    continue
                if (cache / key).exists():
                    raise ValueError(f"완료 캐시가 손상됐습니다. 보존 후 해당 폴더를 옮겨 재시도하세요: {cache / key}")
                LOGGER.info("%s L=%s angle=%s sigma=%s", path.name, length, angle, sigma)
                if representative and args.nafnet and not initialized:
                    initialized = True
                    try:
                        nafnet, naf_device = make_nafnet(args)
                    except ERRORS as exc:
                        naf_device = str(exc)
                        LOGGER.warning("NAFNet 생략 — %s", exc)
                use_ai = representative and args.nafnet
                rows, panels = evaluate(info=info, source=source, sweep=sweep,
                                        nafnet=nafnet if use_ai else None, naf_device=naf_device if use_ai else "")
                with TemporaryDirectory(prefix=".partial-", dir=cache) as tmp:
                    work = Path(tmp)
                    if representative:
                        montage(work / "comparison.png", info, panels)
                    files = {item.name: file_digest(item) for item in work.iterdir()}
                    (work / "result.json").write_text(json.dumps({"key": key, "files": files, "rows": rows},
                                                               allow_nan=False), encoding="utf-8")
                    work.rename(cache / key)
                all_rows.extend(rows)

        with NamedTemporaryFile(dir=args.output, newline="", encoding="utf-8-sig", mode="w", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        temporary.replace(args.output / "results.csv")
        comparison_dir = args.output / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        for index, (key, name) in enumerate(comparisons, 1):
            source = cache / key / "comparison.png"
            destination = comparison_dir / f"{index:02d}-{Path(name).stem}.png"
            if not destination.exists():
                with destination.open("xb") as handle:
                    with source.open("rb") as file:
                        shutil.copyfileobj(file, handle)
        save_reports(args.output, all_rows)
        LOGGER.info("완료: %d행, 재사용 %d조건", len(all_rows), reused)
        return int(any(row["status"] == "failed" for row in all_rows))
    except ERRORS as exc:
        LOGGER.error("실험 실패 — %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
