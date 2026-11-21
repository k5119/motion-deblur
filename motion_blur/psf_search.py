"""참조 영상 없이 PSF 후보를 생성하고 재블러 및 안정성으로 순위를 매긴다."""

from dataclasses import dataclass
from itertools import islice
from typing import Literal

import cv2
import numpy as np

from .classical import apply_motion_blur, linear_psf, richardson_lucy_restore, wiener_restore
from .image_io import Image, limit_image_size
from .psf_estimator import ANALYSIS_MAX_SIZE, cepstral_length, estimate_psf, to_original_axis

Method = Literal["wiener", "rl"]


@dataclass(frozen=True)
class Candidate:
    """입력 크기 기준 PSF와 후보 출처."""

    angle: float
    length: int
    origin: str


@dataclass(frozen=True)
class CandidateScore:
    """낮을수록 바람직하다고 가정한 무참조 점수와 구성 요소."""

    candidate: Candidate
    score: float
    reblur_mae: float
    flat_noise: float
    overshoot: float


@dataclass(frozen=True)
class SearchResult:
    """단일 FFT 기준 후보, 평가 순서의 후보 지표 및 블라인드 선택 영상."""

    baseline: Candidate | None
    selected: CandidateScore | None
    scores: tuple[CandidateScore, ...]
    restored: Image | None


def generate_candidates(observed: Image) -> tuple[Candidate | None, tuple[Candidate, ...]]:
    """FFT·그래디언트 방향과 주변 각도에서 켑스트럼 국소 후보를 만든다."""
    fft = estimate_psf(observed)
    baseline = (Candidate(fft.angle_deg, round(fft.length_original_px), "fft")
                if fft.angle_deg is not None and fft.length_original_px is not None else None)
    gray = cv2.cvtColor(limit_image_size(observed, ANALYSIS_MAX_SIZE), cv2.COLOR_BGR2GRAY)
    candidates: dict[tuple[float, int], Candidate] = {}
    if baseline is not None:
        candidates[(round(baseline.angle, 2) % 180, baseline.length)] = baseline

    for source, direction in (("fft", fft.fft), ("gradient", fft.gradient)):
        if direction.angle_deg is None:
            continue
        for offset in (0, -8, 8):
            angle, _ = to_original_axis(direction.angle_deg + offset, 1, fft.analysis_size, fft.original_size)
            scores = fft.length_scores if source == "fft" and offset == 0 else cepstral_length(gray, direction.angle_deg + offset)[2]
            peaks = sorted((
                (length, score) for index, (length, score) in enumerate(scores)
                if score >= 0 and all(score >= neighbor[1] for neighbor in
                                      scores[max(0, index - 1):index + 2])
            ), key=lambda pair: pair[1], reverse=True)
            for length, _ in islice(peaks, 3):
                _, original_length = to_original_axis(direction.angle_deg + offset, length,
                                                      fft.analysis_size, fft.original_size)
                key = (round(angle, 2) % 180, round(original_length))
                candidates.setdefault(key, Candidate(*key, source))
    return baseline, tuple(candidates.values())


def restore_candidate(observed: Image, candidate: Candidate, method: Method, *, k: float, iterations: int) -> Image:
    """하나의 후보를 기존 Wiener 또는 RL 복원 함수에 전달한다."""
    psf = linear_psf(candidate.length, candidate.angle)
    if method == "wiener":
        return wiener_restore(observed, psf, k)
    if method == "rl":
        return richardson_lucy_restore(observed, psf, iterations)
    raise ValueError("복원 방식은 wiener 또는 rl이어야 합니다.")


def search_psf(observed: Image, *, method: Method = "wiener", k: float = .005, iterations: int = 20) -> SearchResult:
    """오직 흐린 입력으로 모든 후보를 복원·평가하고 최소 점수를 고른다."""
    baseline, candidates = generate_candidates(observed)
    original = observed.astype(np.float32) / 255
    low = cv2.erode(original, np.ones((7, 7), np.uint8))
    high = cv2.dilate(original, np.ones((7, 7), np.uint8))
    flat = np.max(high - low, axis=-1) < .06
    scores = []
    best = None
    restored = None
    for candidate in candidates:
        result = restore_candidate(observed, candidate, method, k=k, iterations=iterations)
        restored_float = result.astype(np.float32) / 255
        reblurred = apply_motion_blur(result, linear_psf(candidate.length, candidate.angle))
        reblur = float(np.mean(np.abs(reblurred.astype(np.float32) / 255 - original)))
        residual = restored_float - cv2.GaussianBlur(restored_float, (0, 0), 1.0)
        noise = float(np.mean(np.abs(residual[flat]))) if np.any(flat) else 0.0
        ringing = float(np.mean(np.maximum(restored_float - high, 0) + np.maximum(low - restored_float, 0)))
        row = CandidateScore(candidate, reblur + 2 * noise + 2 * ringing, reblur, noise, ringing)
        scores.append(row)
        if best is None or row.score < best.score:
            best, restored = row, result
    return SearchResult(baseline, best, tuple(scores), restored)
