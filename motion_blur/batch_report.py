"""배치 실험의 정적 PNG 보고서."""

from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from .image_io import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt

METHOD_LABELS = {
    "input": "Input",
    "wiener_known": "Wiener (known)",
    "wiener_estimated": "Wiener (FFT)",
    "nafnet": "NAFNet",
}


def save_charts(output: Path, rows: list[dict]) -> None:
    """성공한 결과의 평균과 표본 수를 PSNR·SSIM·시간 PNG에 표시한다."""
    for key, title, unit in (
        ("psnr_db", "Mean PSNR", "dB"),
        ("ssim", "Mean SSIM", "SSIM"),
        ("elapsed_ms", "Mean processing time", "ms"),
    ):
        values = defaultdict(list)
        infinite = defaultdict(int)
        for row in rows:
            value = row.get(key)
            if row["status"] == "ok" and value is not None:
                group = row["method"], row.get("device")
                if np.isfinite(value):
                    values[group].append(value)
                else:
                    infinite[group] += 1

        groups = sorted(set(values) | set(infinite))
        labels = []
        means = []
        for method, device in groups:
            group = method, device
            samples = values[group]
            label = METHOD_LABELS[method]
            if key == "elapsed_ms" and device:
                label += f" ({device})"
            label += f"\nn={len(samples)}"
            if infinite[group]:
                label += f", +inf={infinite[group]}"
            labels.append(label)
            means.append(float(np.mean(samples)) if samples else float("nan"))

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        try:
            if groups:
                bars = ax.bar(labels, means)
                ax.bar_label(bars, labels=[
                    f"{mean:.3g}" if values[group] else ""
                    for group, mean in zip(groups, means)
                ], padding=3)
                ax.margins(y=0.18)
            else:
                ax.text(0.5, 0.5, "No eligible samples", transform=ax.transAxes, ha="center")
            ax.set_title(title + " by method")
            ax.set_ylabel(unit)
            with (output / f"{key}.png").open("xb") as file:
                fig.savefig(file, format="png", dpi=140)
        finally:
            plt.close(fig)


def save_montage(path: Path, reference: Image | None, images: dict[str, Image]) -> None:
    """참조와 방법별 결과를 동일한 패널 크기의 PNG로 저장한다."""
    panels = [("Reference", reference)]
    panels.extend((label, images.get(method)) for method, label in METHOD_LABELS.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.2), layout="constrained")
    try:
        for ax, (label, image) in zip(axes, panels):
            ax.set_title(label)
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
            else:
                ax.imshow(image[:, :, ::-1])
        with path.open("xb") as file:
            fig.savefig(file, format="png", dpi=120)
    finally:
        plt.close(fig)
