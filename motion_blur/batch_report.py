"""배치 실험의 정적 PNG 보고서."""

from collections import defaultdict
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .image_io import Image

METHOD_LABELS = {
    "input": "Input",
    "wiener_known": "Wiener (known)",
    "wiener_estimated": "Wiener (FFT)",
    "nafnet": "NAFNet",
}


def paired_quality(rows: list[dict], metric: str) -> dict[tuple[int, int], list[dict]]:
    """동일 입력·해상도·PSF에서 네 방법의 유효 지표가 모두 있는 사례만 묶는다."""
    cases = defaultdict(dict)
    for row in rows:
        angle, length = row.get("actual_angle_deg"), row.get("actual_length_px")
        value = row.get(metric)
        if (row["data_type"] not in {"public", "captured_sharp"} or angle is None
                or length is None or row["status"] != "ok" or value is None
                or not (np.isfinite(value) or (metric == "psnr_db" and value == np.inf))):
            continue
        case = (
            angle, length, row["data_type"], row["image"],
            row["processed_width"], row["processed_height"],
        )
        cases[case][row["method"]] = value

    conditions = defaultdict(list)
    for case, methods in cases.items():
        if METHOD_LABELS.keys() <= methods.keys():
            conditions[case[:2]].append(methods)
    return dict(conditions)


def plot_quality(ax, groups: list[tuple[str, list[dict]]], unit: str) -> None:
    """비교 조건별 공통 표본 수와 네 방법의 평균을 표시한다."""
    if not any(samples for _, samples in groups):
        ax.text(0.5, 0.5, "No complete cases", transform=ax.transAxes, ha="center")
        ax.set_ylabel(unit)
        return

    positions = np.arange(len(groups))
    width = 0.19
    for index, (method, label) in enumerate(METHOD_LABELS.items()):
        means = [float(np.mean([case[method] for case in samples])) if samples else np.nan
                 for _, samples in groups]
        bars = ax.bar(positions + (index - 1.5) * width,
                      [value if np.isfinite(value) else np.nan for value in means], width, label=label)
        for bar, mean in zip(bars, means):
            if mean == np.inf:
                ax.annotate("∞", (bar.get_x() + width / 2, 0.96),
                            xycoords=("data", "axes fraction"), ha="center")
    ax.set_xticks(positions, [f"{label}\nn={len(samples)}" for label, samples in groups])
    ax.set_ylabel(unit if unit == "SSIM" else "PSNR (dB)")
    ax.margins(y=0.15)
    ax.legend()


def save_quality_charts(output: Path, rows: list[dict], metric: str, unit: str) -> None:
    """전체 평균과 각도별 패널에서 길이별 평균을 PNG로 저장한다."""
    conditions = paired_quality(rows, metric)
    figure = Figure(figsize=(8, 4), layout="constrained")
    axes = figure.subplots()
    axes.set_title(f"Mean {unit} by method")
    plot_quality(axes, [("All paired cases", [case for samples in conditions.values() for case in samples])],
                 "dB" if metric == "psnr_db" else "SSIM")
    save_figure(figure, output / f"{metric}.png", dpi=140)

    angles = sorted({row["actual_angle_deg"] for row in rows if row.get("actual_angle_deg") is not None})
    lengths = sorted({row["actual_length_px"] for row in rows if row.get("actual_length_px") is not None})
    figure = Figure(figsize=(6 * max(1, len(angles)), 4), layout="constrained")
    axes = figure.subplots(1, max(1, len(angles)), squeeze=False, sharey=True)[0]
    figure.suptitle(f"Mean {unit} by actual angle and blur length")
    for ax, angle in zip(axes, angles or [None]):
        ax.set_title(f"{angle}°" if angle is not None else "Synthetic inputs")
        plot_quality(ax, [(f"{length} px", conditions.get((angle, length), [])) for length in lengths],
                     "dB" if metric == "psnr_db" else "SSIM")
    save_figure(figure, output / f"{metric}_by_condition.png", dpi=140)


def save_figure(figure: Figure, path: Path, dpi: int) -> None:
    """GUI의 Matplotlib 백엔드를 바꾸지 않고 오프스크린 PNG를 생성한다."""
    canvas = FigureCanvasAgg(figure)
    with path.open("xb") as file:
        canvas.print_figure(file, format="png", dpi=dpi)


def save_charts(output: Path, rows: list[dict]) -> None:
    """공통 품질 표본과 성공한 복원 시간의 그래프를 저장한다."""
    save_quality_charts(output, rows, "psnr_db", "PSNR")
    save_quality_charts(output, rows, "ssim", "SSIM")

    timings = defaultdict(list)
    for row in rows:
        elapsed = row.get("elapsed_ms")
        if row["status"] == "ok" and elapsed is not None and np.isfinite(elapsed):
            timings[row["method"], row["device"]].append(elapsed)

    labels = {**METHOD_LABELS, "wiener_manual": "Wiener (manual)"}
    figure = Figure(figsize=(8, 4), layout="constrained")
    ax = figure.subplots()
    if timings:
        groups = sorted(timings)
        bars = ax.bar(
            [f"{labels[method]} ({device})\nn={len(timings[method, device])}" for method, device in groups],
            [np.mean(timings[group]) for group in groups],
        )
        ax.bar_label(bars, fmt="%.3g", padding=3)
        ax.margins(y=0.18)
    else:
        ax.text(0.5, 0.5, "No eligible samples", transform=ax.transAxes, ha="center")
    ax.set_title("Mean processing time by method")
    ax.set_ylabel("ms")
    save_figure(figure, output / "elapsed_ms.png", dpi=140)


def save_direction_errors(output: Path, rows: list[dict]) -> None:
    """합성 조건별 FFT 축 오차와 후보 반환 건수를 표시한다."""
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == "wiener_estimated" and row["data_type"] in {"public", "captured_sharp"}:
            groups[row["actual_angle_deg"], row["actual_length_px"]].append(row.get("axial_error_deg"))
    figure = Figure(figsize=(10, 4), layout="constrained")
    ax = figure.subplots()
    labels, means = [], []
    for (angle, length), errors in sorted(groups.items()):
        valid = [error for error in errors if error is not None]
        labels.append(f"{angle}° / {length}px\n{len(valid)}/{len(errors)}")
        means.append(float(np.mean(valid)) if valid else np.nan)
    if groups:
        bars = ax.bar(labels, means)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.margins(y=0.18)
    else:
        ax.text(0.5, 0.5, "No synthetic conditions", transform=ax.transAxes, ha="center")
    ax.set(title="Mean FFT axial error (estimates / cases)", ylabel="Degrees")
    save_figure(figure, output / "fft_axial_error.png", dpi=140)


def save_montage(path: Path, reference: Image | None, images: dict[str, Image], statuses: dict[str, str] | None = None) -> None:
    """참조와 방법별 결과를 동일한 패널 크기의 PNG로 저장한다."""
    statuses = statuses or {}
    labels = dict(METHOD_LABELS)
    if "wiener_manual" in statuses:
        labels["wiener_manual"] = "Wiener (manual)"
    panels = [("Reference", reference, "No reference")]
    panels.extend((label, images.get(method), statuses.get(method, "Not run")) for method, label in labels.items())
    figure = Figure(figsize=(13, 3.2), layout="constrained")
    axes = figure.subplots(1, len(panels))
    for ax, (label, image, status) in zip(axes, panels):
        ax.set_title(label)
        ax.axis("off")
        if image is None:
            ax.text(0.5, 0.5, status.replace("_", " "), transform=ax.transAxes, ha="center")
        else:
            ax.imshow(image[:, :, ::-1])
    save_figure(figure, path, dpi=120)
