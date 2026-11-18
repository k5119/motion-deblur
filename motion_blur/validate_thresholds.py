"""라벨이 있는 영상의 검증/테스트 분할로 분석 임계값을 평가한다."""

import argparse
import csv
import hashlib
from pathlib import Path

from matplotlib.figure import Figure

from .analysis import analyze_image
from .batch_report import save_figure
from .decision import ASSESSMENT_MAX_SIZE
from .image_io import limit_image_size, load_image


def measure_manifest(manifest: Path) -> dict[str, list[tuple]]:
    """path,label,split CSV의 서로 다른 파일 내용으로 두 분할을 구성한다."""
    groups = {"validation": [], "test": []}
    fingerprints = {}
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError("manifest에 path,label,split 열이 필요합니다.")
        for row in reader:
            path = manifest.parent / row["path"]
            label, split = row["label"], row["split"]
            if label not in {"sharp", "blurred"} or split not in groups:
                raise ValueError(f"잘못된 label 또는 split — {path}")
            digest = hashlib.sha256(path.read_bytes()).digest()
            if digest in fingerprints:
                raise ValueError(f"동일 이미지 내용이 중복되었습니다 — {path}, {fingerprints[digest]}")
            fingerprints[digest] = path
            analysis = analyze_image(limit_image_size(load_image(path), ASSESSMENT_MAX_SIZE), fft_confidence=0)
            groups[split].append((label == "blurred", analysis.laplacian_variance, analysis.motion_direction))
    if any({case[0] for case in rows} != {False, True} for rows in groups.values()):
        raise ValueError("각 분할에 sharp와 blurred를 모두 지정하세요.")
    return groups


def laplacian_scores(samples: list[tuple], threshold: float) -> dict:
    """blurred를 양성으로 둔 혼동 행렬과 분류 지표를 반환한다."""
    tp = fp = fn = tn = 0
    for blurred, variance, _ in samples:
        predicted = variance <= threshold
        tp += blurred and predicted
        fp += not blurred and predicted
        fn += blurred and not predicted
        tn += not blurred and not predicted
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(samples),
                precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn),
                f1=2 * tp / (2 * tp + fp + fn))


def run_validation(manifest: Path, output: Path, laplacian: list[float], fft: list[float]) -> None:
    """검증 후보를 비교하고 선택된 Laplacian 기준을 독립 테스트에 적용한다."""
    groups = measure_manifest(manifest)
    ranked = [(t, laplacian_scores(groups["validation"], t)) for t in sorted(set(laplacian))]
    selected = max(ranked, key=lambda item: (item[1]["f1"], item[1]["accuracy"]))[0]
    rows = [dict(split="validation", threshold=t, selected=t == selected, **scores) for t, scores in ranked]
    rows.append(dict(split="test", threshold=selected, selected=True, **laplacian_scores(groups["test"], selected)))

    directions = []
    for threshold in sorted(set(fft)):
        for split, samples in groups.items():
            for blurred in (False, True):
                axes = [axis for label, _, axis in samples if label == blurred]
                accepted = sum(axis.angle_deg is not None and axis.confidence >= threshold for axis in axes)
                directions.append(dict(split=split, label="blurred" if blurred else "sharp",
                                       threshold=threshold, n=len(axes), accepted=accepted, rate=accepted / len(axes)))

    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("laplacian", rows), ("fft_coverage", directions)):
        with (output / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    fig = Figure(figsize=(10, 4), layout="constrained")
    left, right = fig.subplots(1, 2)
    for key in ("accuracy", "f1"):
        left.plot([t for t, _ in ranked], [scores[key] for _, scores in ranked], "o-", label=key)
    left.set(xlabel="Laplacian threshold", title=f"Validation (n={len(groups['validation'])})")
    left.legend()
    for split in groups:
        for label in ("sharp", "blurred"):
            points = [r for r in directions if r["split"] == split and r["label"] == label]
            right.plot([r["threshold"] for r in points], [r["rate"] for r in points], "o-",
                       label=f"{split} {label} (n={points[0]['n']})")
    right.set(xlabel="FFT score threshold", title="Direction coverage")
    right.legend()
    save_figure(fig, output / "thresholds.png", dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨이 있는 분할로 분석 기준 검증")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--laplacian", type=float, nargs="+", default=[30, 100, 300, 500, 1000])
    parser.add_argument("--fft", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    args = parser.parse_args()
    if any(not 0 <= t < float("inf") for t in args.laplacian) or any(not 0 <= t <= 1 for t in args.fft):
        parser.error("유효한 임계값을 지정하세요.")
    run_validation(args.manifest, args.output, args.laplacian, args.fft)


if __name__ == "__main__":
    main()
