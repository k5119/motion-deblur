"""공개 및 촬영 이미지의 반복 복원 실험을 실행한다."""

import argparse
from collections.abc import Callable
import csv
from dataclasses import dataclass
import hashlib
from itertools import product
import logging
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import cv2
import numpy as np

from . import DEFAULT_WEIGHTS
from .analysis import analyze_image
from .classical import DEFAULT_WIENER_K, apply_motion_blur, compare_quality, linear_psf, wiener_restore
from .decision import ASSESSMENT_MAX_SIZE
from .experiment import estimated_wiener
from .image_io import Image, limit_image_size, load_image

LOGGER = logging.getLogger(__name__)
DATA_TYPES = ("public", "captured_sharp", "captured_blurred")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PUBLIC_SOURCE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"
PUBLIC_HASHES = {
    "camera.png": "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a",
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
}
FIELDS = (
    "image", "data_type", "original_width", "original_height", "processed_width", "processed_height",
    "analysis_width", "analysis_height", "laplacian_variance", "mean_brightness",
    "applied_angle_deg", "applied_length_px",
    "actual_angle_deg", "actual_length_px", "estimated_angle_deg", "estimated_length_px",
    "direction_score", "axial_error_deg", "method", "wiener_k", "psnr_db", "ssim", "elapsed_ms",
    "device", "status", "detail",
)
PROCESS_ERRORS = (OSError, ValueError, RuntimeError, ImportError, MemoryError, cv2.error)


@dataclass(frozen=True)
class BatchOptions:
    """CLI나 GUI에서 사용하는 배치 실행 설정."""

    input: Path
    output: Path
    quick: bool = False
    download_public: bool = False
    max_size: int = 0
    real_angle: float | None = None
    estimated_length: int | None = None
    real_length: int = 15
    k: float = DEFAULT_WIENER_K
    device: str = "auto"
    weights: Path = DEFAULT_WEIGHTS

    def __post_init__(self) -> None:
        if self.real_angle is not None and not np.isfinite(self.real_angle):
            raise ValueError("수동 각도는 유한한 값이어야 합니다.")
        if self.max_size < 0:
            raise ValueError("최대 크기는 0 이상이어야 합니다.")
        if self.real_length < 1 or (self.estimated_length is not None and self.estimated_length < 1):
            raise ValueError("길이는 양의 정수여야 합니다.")
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError("K는 유한한 양수여야 합니다.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("장치는 auto, cpu, cuda 중 하나여야 합니다.")


def download_public(directory: Path) -> None:
    """지정된 소형 PNG만 받아 SHA-256 검증 후 새 파일에 저장한다."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in PUBLIC_HASHES.items():
        path = directory / name
        already_present = path.exists()
        if already_present:
            LOGGER.info("기존 공개 이미지 사용 — %s", path)
            data = path.read_bytes()
        else:
            LOGGER.info("공개 이미지 다운로드 — %s", name)
            with urlopen(PUBLIC_SOURCE + name, timeout=20) as response:
                data = response.read(2_000_001)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"공개 이미지 SHA-256 불일치 — {path}")
        if not already_present:
            with path.open("xb") as file:
                file.write(data)


def find_inputs(root: Path, quick: bool) -> list[tuple[str, Path]]:
    """데이터 유형별 파일 목록을 정렬한다. 빠른 실행은 유형당 첫 파일을 쓴다."""
    inputs = []
    for kind in DATA_TYPES:
        directory = root / kind
        if directory.is_dir():
            paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            inputs.extend((kind, path) for path in (paths[:1] if quick else paths))
    return inputs


def evaluate_case(
    base: dict, source: Image, angle: int | None, length: int | None,
    args: BatchOptions, nafnet: Callable[[Image], Image] | None,
    device: str | None, model_error: str | None,
) -> tuple[list[dict], dict[str, Image]]:
    """같은 관측 영상에 방법들을 실행하고 실패는 해당 방법의 상태로 남긴다."""
    reference = source if length is not None else None
    psf = linear_psf(length, angle) if length is not None else None
    observed = apply_motion_blur(source, psf) if psf is not None else source
    candidate_length = args.estimated_length or length or args.real_length
    analysis_image = limit_image_size(observed, ASSESSMENT_MAX_SIZE)
    analysis = analyze_image(analysis_image)
    common = {
        **base, "actual_angle_deg": angle, "actual_length_px": length,
        "analysis_width": analysis_image.shape[1], "analysis_height": analysis_image.shape[0],
        "laplacian_variance": analysis.laplacian_variance, "mean_brightness": analysis.mean_brightness,
    }
    LOGGER.info("크기 — 원본 %s×%s / 분석 %d×%d / 복원 %d×%d",
                base.get("original_width", source.shape[1]), base.get("original_height", source.shape[0]),
                analysis_image.shape[1], analysis_image.shape[0], source.shape[1], source.shape[0])
    images = {"input": observed}
    rows = []
    baseline = {**common, "method": "input", "status": "ok"}
    if reference is not None:
        baseline["psnr_db"], baseline["ssim"] = compare_quality(reference, observed)
    rows.append(baseline)

    methods = ["wiener_known", "wiener_estimated", "nafnet"]
    if reference is None and args.real_angle is not None:
        methods.insert(2, "wiener_manual")
    for method in methods:
        row = {**common, "method": method, "device": device if method == "nafnet" else "cpu"}
        rows.append(row)
        if method != "nafnet":
            row["wiener_k"] = args.k
        if method == "wiener_known" and psf is None:
            row.update(status="skipped_no_ground_truth", detail="No ground-truth PSF for captured blur")
            continue
        if method == "nafnet" and nafnet is None:
            row.update(status="failed", detail=model_error)
            continue
        if method == "wiener_estimated":
            row["estimated_length_px"] = candidate_length

        started = perf_counter()
        try:
            if method == "wiener_known":
                image = wiener_restore(observed, psf, args.k)
            elif method == "wiener_estimated":
                image, info = estimated_wiener(observed, candidate_length, args.k, direction=analysis.motion_direction)
                estimated = info["angle_deg"]
                row.update(estimated_angle_deg=estimated, direction_score=info["direction_score"])
                if angle is not None and estimated is not None:
                    row["axial_error_deg"] = abs((estimated - angle + 90) % 180 - 90)
                if image is None:
                    row.update(status="skipped_no_direction", detail="No reliable FFT direction")
                    continue
            elif method == "wiener_manual":
                manual_angle = args.real_angle % 180
                row.update(applied_angle_deg=manual_angle, applied_length_px=args.real_length)
                image = wiener_restore(observed, linear_psf(args.real_length, manual_angle), args.k)
            else:
                image = nafnet(observed)

            row["elapsed_ms"] = (perf_counter() - started) * 1000
            if reference is not None:
                row["psnr_db"], row["ssim"] = compare_quality(reference, image)
            row["status"] = "ok"
            images[method] = image
        except PROCESS_ERRORS as exc:
            row.update(status="failed", detail=str(exc))
            LOGGER.error("%s %s — %s", base["image"], method, exc)

    return rows, images


def make_nafnet(args: BatchOptions) -> tuple[Callable[[Image], Image], str]:
    """모델은 한 번만 로드하며 GPU 실행은 완료 시점까지 동기화한다."""
    from .restoration import load_model, restore_image, select_device

    device = select_device(args.device)
    model = load_model(args.weights, device)
    if device.type == "cuda":
        from torch.cuda import synchronize

        synchronize(device)

        def infer(image: Image) -> Image:
            result = restore_image(image, model)
            synchronize(device)
            return result
    else:
        def infer(image: Image) -> Image:
            return restore_image(image, model)

    return infer, str(device)


def run_batch(args: BatchOptions) -> int:
    """CSV를 사례별로 기록하고 대표 입력별 몽타주와 요약 그래프를 저장한다."""
    from .batch_report import save_charts, save_direction_errors, save_montage

    if args.output.exists():
        raise FileExistsError(f"새 실험 폴더를 지정하세요. ({args.output})")
    if args.download_public:
        download_public(args.input / "public")
    inputs = find_inputs(args.input, args.quick)
    if not inputs:
        raise ValueError("public, captured_sharp, captured_blurred 폴더에 입력 이미지가 없습니다.")

    args.output.mkdir(parents=True, exist_ok=False)
    montage_dir = args.output / "comparisons"
    montage_dir.mkdir()
    nafnet = device = model_error = None
    try:
        nafnet, device = make_nafnet(args)
    except PROCESS_ERRORS as exc:
        model_error = str(exc)
        LOGGER.error("NAFNet 초기화 실패 — %s", exc)

    all_rows = []
    with (args.output / "results.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for index, (kind, path) in enumerate(inputs, 1):
            base = {"image": path.name, "data_type": kind}
            try:
                original = load_image(path)
                source = limit_image_size(original, args.max_size) if args.max_size else original
            except PROCESS_ERRORS as exc:
                row = {**base, "method": "input", "status": "failed", "detail": str(exc)}
                writer.writerow(row)
                all_rows.append(row)
                LOGGER.error("%s — %s", path, exc)
                continue
            base.update(
                original_width=original.shape[1], original_height=original.shape[0],
                processed_width=source.shape[1], processed_height=source.shape[0],
            )
            if kind == "captured_blurred":
                cases = [(None, None)]
            elif args.quick:
                cases = [(45, 15)]
            else:
                cases = product((0, 45, 90), (9, 15, 21))
            for case_index, (angle, length) in enumerate(cases):
                LOGGER.info("%s/%s — angle=%s length=%s", kind, path.name, angle, length)
                try:
                    rows, images = evaluate_case(base, source, angle, length, args, nafnet, device, model_error)
                except PROCESS_ERRORS as exc:
                    rows = [{
                        **base, "actual_angle_deg": angle, "actual_length_px": length,
                        "method": "input", "status": "failed", "detail": str(exc),
                    }]
                    images = {}
                    LOGGER.error("%s — %s", path, exc)
                writer.writerows(rows)
                file.flush()
                all_rows.extend(rows)
                if case_index == 0 and images:
                    save_montage(montage_dir / f"{index:03d}-{kind}.png", source if length is not None else None, images,
                                 {row["method"]: row["status"] for row in rows})

    save_charts(args.output, all_rows)
    save_direction_errors(args.output, all_rows)
    image_tables = args.output / "image_tables"
    image_tables.mkdir()
    for index, (kind, path) in enumerate(inputs, 1):
        with (image_tables / f"{index:03d}-{kind}.csv").open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in all_rows if (row["data_type"], row["image"]) == (kind, path.name))
    LOGGER.info("실험 결과 — %s (%d행)", args.output, len(all_rows))
    return int(any(row["status"] == "failed" for row in all_rows))


def main(argv: list[str] | None = None) -> int:
    """배치 실험 인자를 받고 실패 행이 있으면 종료 코드 1을 반환한다."""
    parser = argparse.ArgumentParser(description="모션 블러 복원의 반복 실험 및 PNG 보고서")
    parser.add_argument("input", type=Path, help="데이터 유형별 하위 폴더가 있는 입력 폴더")
    parser.add_argument("output", type=Path, help="새 실험 출력 폴더")
    parser.add_argument("--quick", action="store_true", help="유형당 첫 입력과 한 가지 합성 조건")
    parser.add_argument("--download-public", action="store_true", help="공개 샘플 PNG 3개만 받기")
    parser.add_argument("--max-size", type=int, default=0, help="실험 전 긴 변 상한 (기본 0, 원본 크기)")
    parser.add_argument("--estimated-length", type=int, help="추정 Wiener 길이 재지정(px)")
    parser.add_argument("--real-angle", type=float, help="실제 촬영 블러의 수동 Wiener 각도")
    parser.add_argument("--real-length", type=int, default=15, help="촬영 블러의 가정 길이(px)")
    parser.add_argument("--k", type=float, default=DEFAULT_WIENER_K)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    try:
        options = BatchOptions(**vars(parser.parse_args(argv)))
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return run_batch(options)
    except PROCESS_ERRORS as exc:
        LOGGER.error("실험 실패 — %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
