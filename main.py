"""이미지 한 장을 NAFNet으로 복원하고 PNG로 저장한다."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import torch

from motion_blur.image_io import limit_image_size, load_image, save_image
from motion_blur.restoration import load_model, restore_image, select_device

LOGGER = logging.getLogger(__name__)
DEFAULT_WEIGHTS = Path("models") / "NAFNet-GoPro-width32.pth"


def positive_int(value: str) -> int:
    """CLI 문자열을 양의 정수로 변환하고 잘못된 값은 argparse에 알린다."""
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.") from exc
    if size < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return size


def main(argv: list[str] | None = None) -> int:
    """명령줄 인자를 받아 성공 시 0, 복원 실패 시 1을 반환한다."""
    parser = argparse.ArgumentParser(description="NAFNet-GoPro-width32로 이미지를 복원합니다.")
    parser.add_argument("input", type=Path, help="입력 이미지 경로")
    parser.add_argument("output", type=Path, help="출력 PNG 경로")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="로컬 가중치 경로")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-size", type=positive_int, metavar="PX",
        help="복원 전 이미지의 긴 변을 최대 PX로 축소 (기본값은 원본 크기)",
    )
    parser.add_argument("--verbose", action="store_true", help="오류 발생 시 스택 추적도 출력")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.output.suffix.lower() != ".png":
            raise ValueError("출력 파일 확장자는 .png여야 합니다.")
        if args.output.exists():
            raise FileExistsError(f"출력 파일이 이미 있습니다. 새 경로를 지정하세요. ({args.output})")

        image = load_image(args.input)
        height, width = image.shape[:2]
        LOGGER.info("입력 — %s (%d × %d)", args.input, width, height)
        if args.max_size is not None:
            image = limit_image_size(image, args.max_size)
            height, width = image.shape[:2]
            LOGGER.info("복원 입력 크기 — %d × %d (긴 변 제한 %dpx)", width, height, args.max_size)
        device = select_device(args.device)
        LOGGER.info("실행 장치 — %s (요청 %s, PyTorch %s)", device, args.device, torch.__version__)

        model = load_model(args.weights, device)
        LOGGER.info("모델 로드 완료 — NAFNetLocal GoPro width32")
        started = perf_counter()
        restored = restore_image(image, model)
        LOGGER.info("복원 완료 — %.2f초", perf_counter() - started)
        output = save_image(args.output, restored)
        LOGGER.info("저장 완료 — %s (%d × %d)", output, width, height)
    except torch.cuda.OutOfMemoryError as exc:
        LOGGER.error(
            "CUDA 메모리가 부족합니다. 다른 GPU 작업을 닫거나 --device cpu로 재실행하세요. %s",
            exc,
            exc_info=args.verbose,
        )
        return 1
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        LOGGER.error("처리 실패 — %s", exc or type(exc).__name__, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
