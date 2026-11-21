"""후보 검색과 별도 정답 평가의 결과를 저장하는 비교 명령."""

import argparse
import csv
from dataclasses import asdict
import logging
from pathlib import Path

import cv2
from matplotlib.figure import Figure
import numpy as np

from . import DEFAULT_WEIGHTS
from .batch import make_nafnet
from .classical import compare_quality, linear_psf, synthesize_motion_blur
from .experiment import save_json
from .image_io import Image, limit_image_size, load_image
from .paired_data import GOPRO_CONDITIONS, discover_gopro, load_manifest
from .psf_search import Candidate, Method, restore_candidate, search_psf
from .quantitative_report import montage, save

LOGGER = logging.getLogger(__name__)
METHODS = ("wiener", "rl")
SYNTHETIC = ((0, 9, 0.0), (30, 15, 0.01), (135, 35, 0.01))
SEED = 702
FIELDS = (
    "dataset", "id", "condition", "width", "height", "method", "role", "status", "angle", "length", "origin",
    "true_angle", "true_length", "sigma", "angle_error", "length_error", "selected", "score", "reblur_mae",
    "flat_noise", "overshoot", "psnr_db", "ssim", "worse_than_input", "worse_than_single",
    "other_candidate_better", "detail",
)


def evaluate_case(
    observed: Image, reference: Image | None, summary: dict, *, method: Method, k: float,
    iterations: int, nafnet=None, nafnet_error: str | None = None,
) -> tuple[list[dict], dict[str, Image | None]]:
    """선택을 먼저 확정한 후 참조 영상과 합성 정답을 사용해 평가한다."""
    search = search_psf(observed, method=method, k=k, iterations=iterations)
    rows = []
    panels = {"Input": observed} if reference is None else {"Reference": reference, "Input": observed}
    base_quality: float | None = None
    singles = {}
    true_angle, true_length = summary.get("true_angle"), summary.get("true_length")

    def record(label, role, image, candidate=None, **extra):
        row = {**summary, "method": label, "role": role,
               "status": "ok" if image is not None else "skipped", **extra}
        if candidate is not None:
            row.update(asdict(candidate))
            if true_angle is not None and true_length is not None:
                row["angle_error"] = abs((candidate.angle - true_angle + 90) % 180 - 90)
                row["length_error"] = abs(candidate.length - true_length)
        if reference is not None and image is not None:
            psnr, ssim = compare_quality(reference, image)
            row.update(psnr_db=psnr, ssim=ssim)
            if base_quality is not None:
                row["worse_than_input"] = psnr < base_quality
            if label in singles:
                row["worse_than_single"] = psnr < singles[label]
        rows.append(row)
        if role != "candidate":
            panels[f"{role} {label}".strip()] = image
        return row

    input_row = record("blurred", "input", observed)
    base_quality = input_row.get("psnr_db")
    panels.pop("input blurred")
    for label in METHODS:
        output = (restore_candidate(observed, search.baseline, label, k=k, iterations=iterations)
                  if search.baseline is not None else None)
        row = record(label, "single_fft", output, search.baseline,
                     detail="FFT direction or length unavailable" if output is None else None)
        if "psnr_db" in row:
            singles[label] = row["psnr_db"]

    selected_psnr = None
    candidates_start = len(rows)
    for scored in search.scores:
        is_selected = scored is search.selected
        image = (search.restored if is_selected else
                 restore_candidate(observed, scored.candidate, method, k=k, iterations=iterations)
                 if reference is not None else None)
        row = record(method, "candidate", image, scored.candidate, selected=is_selected,
                     score=scored.score, reblur_mae=scored.reblur_mae,
                     flat_noise=scored.flat_noise, overshoot=scored.overshoot)
        # 비참조 입력도 각 후보는 검색 단계에서 실제로 복원 및 채점했다.
        row["status"] = "ok"
        if is_selected:
            selected_psnr = row.get("psnr_db")
    candidates = rows[candidates_start:]
    for label in METHODS:
        candidate = search.selected.candidate if search.selected else None
        if candidate is None:
            output = None
        elif label == method:
            output = search.restored
        else:
            output = restore_candidate(observed, candidate, label, k=k, iterations=iterations)
        final = record(label, "selected", output, candidate, detail="No candidates" if output is None else None)
        if label == method and selected_psnr is not None:
            final["other_candidate_better"] = any(r["psnr_db"] > selected_psnr for r in candidates)
    if nafnet is not None:
        try:
            record("nafnet", "comparison", nafnet(observed))
        except (RuntimeError, OSError, ValueError, MemoryError) as exc:
            record("nafnet", "comparison", None, status="failed", detail=str(exc))
    elif nafnet_error:
        record("nafnet", "comparison", None, detail=nafnet_error)
    return rows, panels


def plot(output: Path, rows: list[dict]) -> None:
    """후보 점수와 참조 품질 관계 및 동일 사례 복원 개선량을 표시한다."""
    fig = Figure(figsize=(11, 7), layout="constrained")
    singles = {(r["dataset"], r["id"], r.get("condition"), r["method"]): r for r in rows if r["role"] == "single_fft"}
    for (scatter, deltas), (key, label) in zip(fig.subplots(2, 2), (("psnr_db", "PSNR (dB)"), ("ssim", "SSIM"))):
        candidates = [r for r in rows if r["role"] == "candidate" and np.isfinite(r.get(key, np.nan))]
        for selected in (False, True):
            group = [r for r in candidates if r["selected"] == selected]
            scatter.scatter([r["score"] for r in group], [r[key] for r in group],
                            label=f"{'Selected' if selected else 'Other'} (n={len(group)})", alpha=.7)
        scatter.set(xlabel="Blind score (lower preferred)", ylabel=label, title="Score vs. reference quality")
        scatter.legend()
        for index, method in enumerate(METHODS):
            group = []
            for row in rows:
                if row["role"] != "selected" or row["method"] != method:
                    continue
                baseline = singles[(row["dataset"], row["id"], row.get("condition"), method)]
                first, second = baseline.get(key, np.nan), row.get(key, np.nan)
                if np.isfinite(first) and np.isfinite(second):
                    group.append(second - first)
            deltas.scatter([index] * len(group), group, label=f"{method} (n={len(group)})")
        deltas.set(xticks=[0, 1], xticklabels=METHODS, ylabel=f"{label} difference",
                   title="Selected PSF minus single FFT")
        deltas.axhline(0, color="grey", linewidth=1)
        deltas.legend()
    save(fig, output / "candidate_quality.png")


def main(argv: list[str] | None = None) -> int:
    """로컬 쌍 또는 고정 합성 조건을 선택하여 CSV·PNG를 생성한다."""
    parser = argparse.ArgumentParser(description="블라인드 PSF 후보 복원 실험")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gopro-root", type=Path, help="GOPRO_Large 또는 이를 담은 폴더")
    parser.add_argument("--gopro-condition", choices=(*GOPRO_CONDITIONS, "both"), default="blur")
    parser.add_argument("--gopro-limit", type=int, help="조건별 최대 GoPro 쌍 수")
    parser.add_argument("--synthetic", type=Path, action="append", default=[])
    parser.add_argument("--observed", type=Path, action="append", default=[])
    parser.add_argument("--method", choices=METHODS, default="wiener")
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--k", type=float, default=.005)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--nafnet", action="store_true")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not (args.manifest or args.gopro_root or args.synthetic or args.observed):
        parser.error("최소 하나의 입력 옵션이 필요합니다.")
    if args.max_size < 0 or not np.isfinite(args.k) or args.k <= 0 or args.iterations < 1:
        parser.error("크기, K, 반복 횟수를 확인하세요.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def scaled(image):
        return limit_image_size(image, args.max_size) if args.max_size else image

    def cases(pairs):
        for pair in pairs:
            observed, reference = pair.load()
            yield scaled(observed), scaled(reference), dict(dataset=pair.dataset, id=pair.pair_id, condition=pair.condition)
        for index, path in enumerate(args.synthetic, 1):
            reference = scaled(load_image(path))
            for angle, length, sigma in SYNTHETIC:
                observed = synthesize_motion_blur(reference, linear_psf(length, angle), noise_sigma=sigma, seed=SEED)
                yield observed, reference, dict(dataset="synthetic", id=f"{index}-{path.stem}-{angle}-{length}",
                                                true_angle=angle, true_length=length, sigma=sigma)
        for index, path in enumerate(args.observed, 1):
            yield scaled(load_image(path)), None, dict(dataset="unpaired", id=f"{index}-{path.stem}")

    try:
        pairs = load_manifest(args.manifest) if args.manifest else []
        if args.gopro_root:
            conditions = GOPRO_CONDITIONS if args.gopro_condition == "both" else (args.gopro_condition,)
            pairs.extend(discover_gopro(args.gopro_root, conditions=conditions, limit=args.gopro_limit))
        keys = {(pair.dataset, pair.pair_id, pair.condition) for pair in pairs}
        if len(keys) != len(pairs):
            raise ValueError("manifest와 GoPro 디렉터리에서 같은 쌍 ID가 중복됩니다.")
        if not (pairs or args.synthetic or args.observed):
            raise ValueError("입력 쌍이 없습니다.")
        args.output.mkdir(parents=True, exist_ok=False)
        model = error = None
        if args.nafnet:
            try:
                model, _ = make_nafnet(args)
            except (ImportError, OSError, RuntimeError, ValueError, MemoryError) as exc:
                error = str(exc)
                LOGGER.warning("NAFNet 생략 — %s", exc)
        save_json(args.output / "run.json", {"method": args.method, "k": args.k, "iterations": args.iterations,
                                             "max_size": args.max_size, "synthetic_seed": SEED,
                                             "nafnet_requested": args.nafnet})
        all_rows = []
        with (args.output / "results.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for index, (observed, reference, info) in enumerate(cases(pairs), 1):
                LOGGER.info("%s/%s (조건 %s)", info["dataset"], info["id"], info.get("condition"))
                height, width = observed.shape[:2]
                summary = {**info, "width": width, "height": height}
                rows, panels = evaluate_case(observed, reference, summary, method=args.method, k=args.k,
                                              iterations=args.iterations, nafnet=model, nafnet_error=error)
                writer.writerows(rows)
                handle.flush()
                all_rows.extend(rows)
                montage(args.output / f"comparison-{index:03d}.png",
                        {"image": f'{info["id"]} [{info["condition"]}]' if info.get("condition") else info["id"],
                         "width": width, "height": height,
                         "length": info.get("true_length"), "angle": info.get("true_angle"),
                         "sigma": info.get("sigma"), "has_reference": reference is not None}, panels)
        plot(args.output, all_rows)
        return int(any(r["status"] == "failed" for r in all_rows))
    except (OSError, ValueError, RuntimeError, MemoryError, cv2.error) as exc:
        LOGGER.error("실험 실패 — %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
