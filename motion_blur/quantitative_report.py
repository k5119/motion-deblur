"""정량 실험의 표·그래프·정성 비교 자료를 생성한다."""

from collections import defaultdict
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def save(figure: Figure, path: Path) -> None:
    """완성된 PNG만 교체해 재개 도중 기존 그림을 보호한다."""
    temporary = path.with_suffix(".tmp")
    FigureCanvasAgg(figure).print_figure(temporary, format="png", dpi=150)
    temporary.replace(path)


def montage(path: Path, info: dict, panels: dict) -> None:
    """동일 입력의 정성 비교와 PSF·잡음 설정을 읽기 쉬운 PNG로 저장한다."""
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    fig = Figure(figsize=(12, 3.3 * rows), layout="constrained")
    axes = fig.subplots(rows, columns, squeeze=False).flat
    for ax in axes:
        ax.axis("off")
    for ax, (name, image) in zip(fig.axes, panels.items()):
        ax.set_title(name.replace("_", " "))
        if image is None:
            ax.text(0.5, 0.5, "Unavailable / skipped", ha="center")
        else:
            ax.imshow(image[:, :, ::-1])
    suffix = ("no reference" if info["length"] is None else
              f'L={info["length"]} px, angle={info["angle"]}°, sigma={info["sigma"]}')
    ax_title = info["image"].encode("ascii", errors="replace").decode()
    fig.suptitle(f'{ax_title} | {info["width"]}×{info["height"]} | {suffix}')
    save(fig, path)


def condition(row: dict) -> tuple:
    """서로 다른 입력 유형·이미지·PSF·잡음 조건을 구분한다."""
    return tuple(row.get(k) for k in ("data_type", "image", "width", "height", "length", "angle", "sigma", "seed"))


def paired_core(rows: list[dict], methods: tuple, metric: str) -> list[dict]:
    """모든 비교 방법의 지표가 유한한 공통 조건만 반환한다."""
    cases = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        tag = row["method"], row.get("psf") or ""
        if row["kind"] == "core" and row["status"] == "ok" and tag in methods and isinstance(value, (float, int)):
            cases[condition(row)][tag] = row
    return [case for case in cases.values() if set(methods) <= case.keys()]


def save_reports(output: Path, rows: list[dict]) -> None:
    """짝지은 품질 비교, 파라미터 민감도, 추정 오차를 저장한다."""
    core = (("blurred", ""), ("own_wiener_wrap", "known"), ("skimage_wiener", "known"),
            ("own_rl_wrap", "known"), ("skimage_rl", "known"), ("own_wiener", "known"),
            ("own_rl", "known"), ("own_wiener", "estimated"), ("own_rl", "estimated"))
    with_ai = (("blurred", ""), ("own_wiener", "known"), ("own_rl", "known"), ("nafnet", ""))
    for metric in ("psnr_db", "ssim"):
        fig = Figure(figsize=(11, 7), layout="constrained")
        for ax, (title, methods) in zip(fig.subplots(2, 1), (("Classical families", core), ("NAFNet representative cases", with_ai))):
            paired = paired_core(rows, methods, metric)
            if paired:
                labels = [" ".join(tag).replace("_", " ").strip() for tag in methods]
                bars = ax.bar(labels, [np.mean([case[tag][metric] for case in paired]) for tag in methods])
                ax.bar_label(bars, fmt="%.3g")
                ax.tick_params(axis="x", labelrotation=18)
                ax.margins(y=.16)
            else:
                ax.text(.5, .5, "No complete reference cases", transform=ax.transAxes, ha="center")
            ax.set(title=f"{title} (paired n={len(paired)})", ylabel=metric)
        save(fig, output / f"{metric}.png")

    for kind, param in (("k_sweep", "k"), ("rl_sweep", "iterations")):
        selected = [r for r in rows if r["kind"] == kind and r["status"] == "ok"]
        fig = Figure(figsize=(12, 4), layout="constrained")
        for ax, metric in zip(fig.subplots(1, 3), ("psnr_db", "ssim", "elapsed_ms")):
            for sigma in sorted({r["sigma"] for r in selected}):
                group = defaultdict(list)
                for row in selected:
                    value = row.get(metric)
                    if row["sigma"] == sigma and isinstance(value, (int, float)):
                        group[row[param]].append(value)
                keys = sorted(group)
                if keys:
                    counts = "/".join(str(len(group[key])) for key in keys)
                    ax.plot(keys, [np.mean(group[key]) for key in keys], "o-", label=f"sigma={sigma} (n={counts})")
            ax.set(xlabel=param, title=metric)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.suptitle("Known PSF, taper boundary | 15px, 30°")
        save(fig, output / f"{kind}.png")

    baseline = [r for r in rows if r["method"] == "blurred" and r["length"] is not None]
    fig = Figure(figsize=(10, 4), layout="constrained")
    for ax, metric in zip(fig.subplots(1, 2), ("angle_error", "length_error")):
        names, values = [], []
        for sigma in sorted({r["sigma"] for r in baseline}):
            group = [r for r in baseline if r["sigma"] == sigma]
            present = [r[metric] for r in group if r.get(metric) is not None]
            names.append(f"{sigma}\n{len(present)}/{len(group)}")
            values.append(np.mean(present) if present else np.nan)
        ax.bar(names, values)
        ax.set(xlabel="Noise sigma (returned / total)", title=metric, ylabel="Degrees" if metric == "angle_error" else "Pixels")
    save(fig, output / "estimation_error.png")
