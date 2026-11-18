"""공식 NAFNet-GoPro-width32 체크포인트를 사용하는 단일 이미지 추론."""

import logging
from pathlib import Path
from pickle import UnpicklingError
from typing import Literal

import numpy as np
import torch

from ._nafnet.network import NAFNetLocal
from .image_io import Image

DeviceChoice = Literal["auto", "cpu", "cuda"]
LOGGER = logging.getLogger(__name__)


def select_device(choice: DeviceChoice) -> torch.device:
    """장치 옵션을 실제 PyTorch 장치로 변환한다. auto는 CUDA를 우선한다."""
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cuda":
        raise RuntimeError(
            "CUDA를 사용할 수 없습니다. NVIDIA 드라이버와 CUDA용 PyTorch를 "
            "확인하거나 --device cpu로 실행하세요."
        )
    return torch.device("cpu")


def load_model(weights: Path, device: torch.device) -> NAFNetLocal:
    """로컬 .pth와 실행 장치를 받아 가중치를 로드한 평가 모드 모델을 반환한다."""
    LOGGER.info("가중치 로드 — %s", weights)
    if not weights.is_file():
        raise FileNotFoundError(
            f"가중치 파일이 없습니다. 공식 NAFNet-GoPro-width32.pth를 받아 "
            f"--weights로 지정하세요. ({weights})"
        )
    # 공식 체크포인트의 params 항목을 CPU에 먼저 읽어 GPU 저장 여부와 무관하게 로드한다.
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except (RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
        raise ValueError(f"체크포인트를 읽을 수 없습니다. 공식 파일을 확인하세요. ({weights})") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("params"), dict):
        raise ValueError("공식 체크포인트의 params 가중치 사전이 없습니다.")

    # 공식 GoPro width32 평가 설정. NAFNetLocal의 기본값은 256×256과 정확한 로컬 풀링이다.
    model = NAFNetLocal(
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    try:
        model.load_state_dict(checkpoint["params"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"가중치가 NAFNet-GoPro-width32 구조와 호환되지 않습니다. ({weights})") from exc
    return model.eval().to(device)


@torch.inference_mode()
def restore_image(image: Image, model: NAFNetLocal) -> Image:
    """(H, W, 3) uint8 BGR과 로드된 모델을 받아 같은 형식의 복원 배열을 반환한다."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("복원 입력은 비어 있지 않은 (H, W, 3) uint8 BGR 배열이어야 합니다.")

    # BGR HWC uint8 → RGB NCHW float32 [0, 1].
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    device = next(model.parameters()).device
    batch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    batch = batch.to(device=device, dtype=torch.float32).div_(255.0)

    # 공식 forward는 오른쪽과 아래를 16의 배수까지 패딩한 뒤 원래 크기로 자른다.
    restored = model(batch)[0]
    if restored.shape != batch.shape[1:]:
        raise RuntimeError("모델 출력 크기가 입력 이미지와 일치하지 않습니다.")
    if not torch.isfinite(restored).all().item():
        raise RuntimeError("모델 출력에 비유한 값이 포함되어 있습니다.")

    # RGB CHW float32 → BGR HWC uint8. 공식 저장 경로와 같이 반올림한다.
    rgb8 = restored.clamp(0, 1).mul(255).round().to(torch.uint8)
    rgb8 = rgb8.permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(rgb8[:, :, ::-1])
