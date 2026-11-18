"""이미지 품질을 분석하고 선택적으로 NAFNet 복원 결과를 저장한다."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

from motion_blur import DEFAULT_WEIGHTS
from motion_blur.analysis import analyze_image
from motion_blur.decision import (
    ASSESSMENT_MAX_SIZE, HIGH_LAPLACIAN_VARIANCE, LOW_LAPLACIAN_VARIANCE, decide_correction,
)
from motion_blur.image_io import Image, limit_image_size, load_image, save_image
from motion_blur.motion_direction import MIN_CONFIDENCE

LOGGER = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    """CLI 문자열을 양의 정수로 변환하고 잘못된 값은 argparse에 알린다."""
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.") from exc
    if size < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return size


def restore_and_save(image: Image, output: Path, weights: Path, device_choice: str) -> None:
    """복원용 BGR 이미지와 설정을 받아 모델을 실행하고 PNG로 저장한다."""
    # 분석 전용 실행은 PyTorch와 복원 모듈을 불러올 필요가 없다.
    import torch

    from motion_blur.restoration import load_model, restore_image, select_device

    try:
        device = select_device(device_choice)
        LOGGER.info("실행 장치 — %s (요청 %s, PyTorch %s)", device, device_choice, torch.__version__)
        model = load_model(weights, device)
        LOGGER.info("모델 로드 완료 — NAFNetLocal GoPro width32")
        started = perf_counter()
        restored = restore_image(image, model)
        LOGGER.info("복원 완료 — %.2f초", perf_counter() - started)
        saved = save_image(output, restored)
        height, width = restored.shape[:2]
        LOGGER.info("저장 완료 — %s (%d × %d)", saved, width, height)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA 메모리가 부족합니다. 다른 GPU 작업을 닫거나 --device cpu로 재실행하세요. {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """명령줄 인자를 받아 성공 시 0, 처리 실패 시 1을 반환한다."""
    parser = argparse.ArgumentParser(description="이미지 품질을 분석하고 NAFNet으로 복원합니다.")
    parser.add_argument("input", type=Path, help="입력 이미지 경로")
    parser.add_argument("output", type=Path, nargs="?", help="출력 PNG 경로 (복원 시 필수)")
    parser.add_argument("--analyze-only", action="store_true", help="원본 이미지 품질만 분석")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="로컬 가중치 경로")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-size", type=positive_int, metavar="PX",
        help="복원 전 이미지의 긴 변을 최대 PX로 축소 (기본값은 원본 크기)",
    )
    parser.add_argument("--verbose", action="store_true", help="오류 발생 시 스택 추적도 출력")
    parser.add_argument("--fft-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--laplacian-low", type=float, default=LOW_LAPLACIAN_VARIANCE)
    parser.add_argument("--laplacian-high", type=float, default=HIGH_LAPLACIAN_VARIANCE)
    args = parser.parse_args(argv)
    if not (0 <= args.fft_confidence <= 1 and 0 <= args.laplacian_low < args.laplacian_high < float("inf")):
        parser.error("FFT 기준은 0~1, Laplacian 기준은 0 ≤ low < high여야 합니다.")
    if args.analyze_only and args.output is not None:
        parser.error("--analyze-only에는 출력 PNG 경로를 지정하지 않습니다.")
    if not args.analyze_only and args.output is None:
        parser.error("복원하려면 출력 PNG 경로를 지정하세요.")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.output is not None:
            if args.output.suffix.lower() != ".png":
                raise ValueError("출력 파일 확장자는 .png여야 합니다.")
            if args.output.exists():
                raise FileExistsError(f"출력 파일이 이미 있습니다. 새 경로를 지정하세요. ({args.output})")

        image = load_image(args.input)
        height, width = image.shape[:2]
        LOGGER.info("입력 — %s (%d × %d)", args.input, width, height)
        analysis = analyze_image(image, fft_confidence=args.fft_confidence)
        LOGGER.info(
            "원본 품질 분석 (%d × %d) — 평균 밝기 %.2f, 어두운 픽셀 %.2f%%, "
            "밝은 픽셀 %.2f%%, Laplacian 분산 %.4f",
            width, height, analysis.mean_brightness, 100 * analysis.dark_pixel_ratio,
            100 * analysis.bright_pixel_ratio, analysis.laplacian_variance,
        )
        direction = analysis.motion_direction
        angle = "추정 불가" if direction.angle_deg is None else f"{direction.angle_deg:.2f}°"
        LOGGER.info("FFT 모션 축 후보 — %s (방향성 점수 %.3f)", angle, direction.confidence)
        assessment_image = limit_image_size(image, ASSESSMENT_MAX_SIZE)
        assessment_height, assessment_width = assessment_image.shape[:2]
        assessment_analysis = analysis
        if assessment_image is not image:
            assessment_analysis = analyze_image(assessment_image, fft_confidence=args.fft_confidence)
        assessment = decide_correction(
            assessment_analysis, low_laplacian=args.laplacian_low, high_laplacian=args.laplacian_high,
        )
        LOGGER.info(
            "보정 판단 (%d × %d) — %s. %s",
            assessment_width, assessment_height, assessment.decision.label, assessment.reason,
        )
        if args.output is None:
            return 0

        if args.max_size is not None:
            image = limit_image_size(image, args.max_size)
            height, width = image.shape[:2]
        LOGGER.info("복원 입력 크기 — %d × %d", width, height)
        restore_and_save(image, args.output, args.weights, args.device)
    except (OSError, RuntimeError, ValueError, MemoryError, ImportError) as exc:
        LOGGER.error("처리 실패 — %s", exc or type(exc).__name__, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
