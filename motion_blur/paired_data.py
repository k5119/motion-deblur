"""선택한 GoPro·RealBlur 쌍의 로컬 경로를 읽는다."""

import csv
from dataclasses import dataclass
from pathlib import Path

from .image_io import Image, load_image


@dataclass(frozen=True)
class PairedImage:
    """데이터셋 이름과 사용자 지정 쌍 ID 및 파일 경로."""

    dataset: str
    pair_id: str
    blur: Path
    sharp: Path

    def load(self) -> tuple[Image, Image]:
        """흐림·정답 영상을 읽고 동일한 해상도를 확인한다."""
        observed, reference = load_image(self.blur), load_image(self.sharp)
        if observed.shape != reference.shape:
            raise ValueError(f"영상 쌍의 해상도가 다릅니다: {self.dataset}/{self.pair_id}")
        return observed, reference


def load_manifest(path: Path) -> list[PairedImage]:
    """dataset,id,blur,sharp CSV를 읽고 상대 경로를 CSV 위치에서 해석한다."""
    pairs = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"dataset", "id", "blur", "sharp"} <= set(reader.fieldnames or []):
            raise ValueError("manifest에 dataset,id,blur,sharp 열이 필요합니다.")
        for index, row in enumerate(reader, 2):
            dataset, pair_id, blur, sharp = ((row[key] or "").strip() for key in ("dataset", "id", "blur", "sharp"))
            dataset = dataset.lower()
            key = (dataset, pair_id)
            if dataset not in {"gopro", "realblur", "realblur-j", "realblur-r"} or not all((pair_id, blur, sharp)):
                raise ValueError(f"지원 데이터셋과 필수 열을 확인하세요: {path}:{index}")
            if key in ids:
                raise ValueError(f"중복 쌍 ID: {dataset}/{pair_id}")
            ids.add(key)
            blurred_path, sharp_path = path.parent / blur, path.parent / sharp
            if blurred_path.resolve() == sharp_path.resolve():
                raise ValueError(f"서로 다른 흐림·정답 파일이 필요합니다: {dataset}/{pair_id}")
            for image_path in (blurred_path, sharp_path):
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
            pairs.append(PairedImage(dataset, pair_id, blurred_path, sharp_path))
    return pairs
