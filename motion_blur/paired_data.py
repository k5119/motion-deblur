"""선택한 GoPro·RealBlur 쌍의 로컬 경로를 읽는다."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from .image_io import Image, load_image

LOGGER = logging.getLogger(__name__)
GOPRO_CONDITIONS = ("blur", "blur_gamma")


@dataclass(frozen=True)
class PairedImage:
    """데이터셋 이름과 사용자 지정 쌍 ID 및 파일 경로."""

    dataset: str
    pair_id: str
    blur: Path
    sharp: Path
    condition: str | None = None

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
            condition = (row.get("condition") or "").strip() or None
            if dataset == "gopro":
                folder = Path(blur).parent.name
                condition = condition or (folder if folder in GOPRO_CONDITIONS else "blur")
                if condition not in GOPRO_CONDITIONS or (folder in GOPRO_CONDITIONS and folder != condition):
                    raise ValueError(f"GoPro 조건이 경로와 맞지 않습니다: {path}:{index}")
            key = (dataset, pair_id, condition)
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
            pairs.append(PairedImage(dataset, pair_id, blurred_path, sharp_path, condition))
    return pairs

def discover_gopro(
    root: Path, *, conditions: tuple[str, ...] = ("blur",), limit: int | None = None,
) -> list[PairedImage]:
    """GoPro test의 같은 시퀀스·같은 파일명만 매칭한다. limit는 조건당 개수다."""
    if not conditions or any(condition not in GOPRO_CONDITIONS for condition in conditions):
        raise ValueError("GoPro 조건은 blur 또는 blur_gamma여야 합니다.")
    if limit is not None and limit < 1:
        raise ValueError("GoPro 쌍 수는 1 이상이어야 합니다.")
    test = next((path for path in (root / "GOPRO_Large" / "test", root / "test", root)
                 if path.is_dir() and path.name == "test"), None)
    if test is None:
        raise FileNotFoundError(f"GoPro test 디렉터리를 찾을 수 없습니다: {root}")

    sequences = sorted(path for path in test.iterdir() if path.is_dir())
    pairs = []
    for condition in dict.fromkeys(conditions):
        matches = 0
        skipped = 0
        for sequence in sequences:
            if limit is not None and matches >= limit:
                break
            blur = sequence / condition
            sharp = sequence / "sharp"
            if not blur.is_dir():
                continue
            references = {path.name: path for path in sharp.iterdir() if path.is_file()} if sharp.is_dir() else {}
            for path in sorted(blur.iterdir()):
                if not path.is_file() or path.suffix.lower() != ".png":
                    continue
                reference = references.get(path.name)
                if reference is None:
                    skipped += 1
                    continue
                pairs.append(PairedImage("gopro", f"{sequence.name}/{path.name}", path, reference, condition))
                matches += 1
                if limit is not None and matches >= limit:
                    break
        if skipped:
            LOGGER.warning("GoPro %s — 검사한 파일 중 같은 이름의 sharp가 없는 %d개 생략", condition, skipped)
        if not matches:
            LOGGER.warning("GoPro %s — 일치하는 쌍이 없습니다: %s", condition, test)
    return pairs
