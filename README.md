# NAFNet 모션 블러 복원

사진 한 장의 품질을 분석하고 복원 여부를 평가하는 Python CLI입니다. 복원 명령은 분석 지표와 판단 근거를 출력한 뒤 공식 NAFNet-GoPro-width32로 복원 결과를 PNG에 저장합니다.

## 개발 환경

- Windows 11, Python 3.11.9
- PyTorch 2.6 이상, NumPy, OpenCV (headless)
- CPU 또는 CUDA 지원 NVIDIA GPU

## 설치

프로젝트 루트에서 PowerShell을 열고 가상 환경과 의존성을 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU 전용 PC에서는 아래 명령으로 PyTorch를 설치합니다. NVIDIA GPU를 사용하려면 공식 설치 선택기에서 드라이버와 호환되는 Windows CUDA 빌드를 골라 같은 가상 환경에 설치합니다. [PyTorch 설치 선택기](https://pytorch.org/get-started/locally/)

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
```

공식 가중치를 한 번 다운로드해 프로젝트의 `models/NAFNet-GoPro-width32.pth`에 둡니다. 다른 위치를 쓰려면 `--weights`로 경로를 지정합니다. [공식 NAFNet-GoPro-width32 가중치](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view)

## 실행

```powershell
.\.venv\Scripts\python.exe .\main.py ".\흔들린 사진.jpg" ".\results\복원.png"
```

기본 설정은 CUDA를 사용할 수 있으면 CUDA를 선택하고, 그 외에는 CPU를 사용합니다. 장치를 지정하려면 `--device cpu` 또는 `--device cuda`를 추가합니다. 출력 폴더는 자동으로 만들며 기존 파일을 보호하므로 재실행 시 새 출력 파일명을 지정합니다.

기본 실행은 원본 해상도를 사용합니다. `--max-size`에 양의 정수를 지정하면
가로세로 비율을 유지하며 긴 변을 해당 픽셀 수로 제한합니다. 제한 이하의 입력은
그대로 처리하고, 결과 PNG는 실제 복원 해상도로 저장합니다.

```powershell
python main.py "test_images\test.jpg" "results\restored_768.png" --max-size 768
python main.py "test_images\test.jpg" "results\restored_512.png" --max-size 512
python main.py "test_images\test.jpg" "results\restored_cpu.png" --max-size 512 --device cpu
```

## 품질 분석만 실행

```powershell
python main.py "test_images\test.jpg" --analyze-only
```

분석 전용 실행에는 NumPy와 OpenCV가 필요합니다. 결과는 터미널에 출력됩니다.
복원 명령에서도 같은 분석을 먼저 실행합니다. 두 경우 모두 원본 해상도를
분석하며, `--max-size`는 복원에만 적용됩니다.

| 지표 | 계산 방법 |
| --- | --- |
| 평균 밝기 | OpenCV로 변환한 8비트 그레이스케일의 산술평균 (0~255) |
| 어두운·밝은 픽셀 비율 | 각각 0~50, 205~255 구간의 픽셀 수 ÷ 전체 픽셀 수. 터미널은 백분율로 표시 |
| 그레이스케일 히스토그램 | 0부터 255까지 각 명암값의 픽셀 수 |
| 선명도 | 그레이스케일에 적용한 64비트 Laplacian 응답의 모집단 분산 (`ksize=1`) |
| FFT 모션 축 후보 | 로그 크기 스펙트럼의 방향성으로 추정한 축과 0~1 점수 |

밝기 구간은 픽셀 분포를 요약하는 기준입니다. Laplacian 분산은 경계
강도에 반응하며 이미지의 질감, 노이즈, 해상도에 따라서도 달라집니다.
히스토그램 256개 값은 분석 결과 객체에 저장됩니다.

FFT 분석은 평균을 제거한 그레이스케일에 Hann 윈도우를 적용하고, 2차원 FFT의
영주파수를 중심으로 이동한 뒤 로그 크기를 계산합니다. 0.06~0.32 cycles/pixel을
세 대역으로 나누고 극좌표 표본의 이중각 성분을 평균해 강한 주파수 축을 찾습니다.
선형 이동은 이동 방향의 주파수 성분을 약화시키므로, 강한 축에 90°를 더해
공간 영역의 이동 축 후보를 얻습니다. 오른쪽 수평이 0°이고 시계 방향으로
`[0°, 180°)`입니다. 대역 간 일관성과 방향성의 크기를 반영한 점수가
0.20 미만이면 추정 불가로 표시합니다.

글자, 직선, 반복 무늬도 방향성 점수를 높일 수 있습니다.
[합성 PSF 테스트 및 결과표](docs/fft-direction-validation.md)

## 복원 필요성 판단

분석 전용 명령과 복원 명령 모두 `복원 권장`, `복원 불필요`, `판단 보류` 중
하나와 그 근거를 출력합니다. 판단에는 긴 변을 최대 512px로 줄인 별도 영상의
지표를 사용하며, 작은 입력은 원래 크기로 판단합니다. 원본 분석 로그와
복원용 `--max-size`는 이 판단 해상도와 독립적입니다.

잠정 기준은 `motion_blur/decision.py`의 상수로 설정합니다. 판단은 사용자에게
제공하는 안내이며, 복원 명령은 세 판단 결과에서 모두 NAFNet을 실행합니다.
[판단 규칙과 검증 결과](docs/restoration-assessment.md)

`motion_blur/_nafnet/`에는 공식 평가 설정에 맞춘 추론 코드와 라이선스가 포함되어 있습니다. [NAFNet 공식 평가 설정](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/options/test/GoPro/NAFNet-width32.yml)

## 고전 복원 실험

NumPy와 OpenCV만으로 합성 블러와 Wiener 복원을 실행할 수 있습니다.
아래 명령은 PowerShell에서 프로젝트 루트를 기준으로 실행합니다.
합성과 비교에는 새 폴더를, 단독 복원에는 새 PNG 경로를 지정합니다.

```powershell
python -m motion_blur.experiment synthesize "test_images\sharp.jpg" "results\synthetic" --angle 45 --length 15
python -m motion_blur.experiment wiener "results\synthetic\blurred.png" "results\wiener-known.png" --psf "results\synthetic\psf.json" --k 0.005
python -m motion_blur.experiment wiener "test_images\blurry.jpg" "results\wiener-estimated.png" --estimate --length 15 --k 0.005
python -m motion_blur.experiment compare "test_images\sharp.jpg" "results\comparison" --angle 45 --length 15 --estimated-length 15 --max-size 384 --device cpu
```

각도는 오른쪽 수평 0°에서 시계 방향으로 측정하며, 길이는 처리 해상도의
픽셀 단위입니다. 추정 모드의 길이는 사용자가 지정합니다. FFT 방향을
추정할 수 없으면 해당 Wiener 복원을 생략하고 로그에 표시합니다.

`compare`는 선명한 원본에서 블러를 합성하고, 동일 크기의 원본·블러·정답
PSF Wiener·추정 PSF Wiener·NAFNet PNG와 `psf.json`, `metrics.json`을
저장합니다. `--max-size`는 합성 **전에** 원본에 적용됩니다. 단독
`wiener` 명령은 참조 영상이 없는 사진에도 사용할 수 있습니다.
PSNR·SSIM은 선명한 참조 영상으로 합성한 비교 실험에서만 계산합니다.
[수학적 원리와 실험 결과](docs/classical-restoration.md)

## 공개 이미지 반복 실험

기존 `test_images`의 파일 경로는 기존 CLI에서 그대로 사용할 수 있습니다.
배치 CLI는 입력 폴더 아래의 다음 하위 폴더에서 이미지 유형을 읽습니다.

| 하위 폴더 | 입력과 평가 방식 |
| --- | --- |
| `public` | 공개 선명 이미지. 축소 후 블러를 합성하고 처리 원본으로 PSNR·SSIM 계산 |
| `captured_sharp` | 직접 촬영한 선명 이미지. 공개 이미지와 동일한 합성·평가 절차 |
| `captured_blurred` | 직접 촬영한 흔들린 이미지. 추정 Wiener와 NAFNet 실행. 참조 화질 지표 및 실제 PSF 항목은 빈칸 |

## 공개 샘플 출처

세 이미지는 scikit-image v0.25.2의 샘플입니다.
`--download-public`를 명시한 실행에서만 아래 PNG 세 개를 받으며, SHA-256으로
검증합니다. 이미 존재하는 파일은 읽고 확인합니다. 네트워크 없이 실행하려면
`public`에 준비한 선명 이미지를 두고 다운로드 옵션을 생략합니다.
[scikit-image 출처·라이선스 설명](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py)

| 파일 | 원 출처 | 라이선스 |
| --- | --- | --- |
| [camera.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/camera.png) | Lav Varshney 촬영 | CC0 |
| [chelsea.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/chelsea.png) | Stefan van der Walt 촬영 | CC0 |
| [astronaut.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png) | NASA Great Images, Eileen Collins | 퍼블릭 도메인 (scikit-image 표기) |

### 빠른 실행과 전체 실행

배치 전용 그래프 의존성을 설치하고 기존 PyTorch 환경과 NAFNet 가중치를
사용합니다. 실행마다 새 출력 폴더를 지정합니다.

```powershell
python -m pip install -r requirements-batch.txt
python -m motion_blur.batch test_images results\batch-quick --quick --download-public --max-size 192 --device cpu
python -m motion_blur.batch test_images results\batch-full --device auto
```

- 빠른 실행은 유형마다 파일명 순서의 첫 이미지를 선택하고, 선명 입력에는 45°·15px 한 조건을 적용합니다.
- 전체 실행은 모든 입력을 처리하고, 선명 이미지마다 0°·45°·90°와 9·15·21px의 아홉 조건을 비교합니다. 기본값과 `--max-size 0`은 원본 해상도입니다. 양의 값은 블러 합성 전에 적용합니다. 배치 분석은 관측 영상의 긴 변을 최대 512px로 제한하며 PSF 길이는 복원 해상도의 픽셀 단위입니다.
- FFT는 방향을 추정합니다. 비교에서 추정 Wiener에 주는 길이는 기본적으로 합성의 실제 길이이며, 촬영 블러의 기본 가정 길이는 15px입니다. `--estimated-length`로 공통 지정하거나 `--real-length`로 촬영 블러의 기본값을 바꿀 수 있습니다.
- 각 실험 폴더에 `results.csv`, PSNR·SSIM의 전체 평균 및 각도×길이별 그래프와 처리 시간·FFT 축 오차 그래프, `image_tables/`의 이미지별 표, `comparisons/`의 입력별 첫 조건 비교 이미지가 저장됩니다. 입력 기준 행도 CSV에 기록합니다. 누락되거나 계산 불가능한 값은 빈칸으로 기록합니다. 동일 영상의 PSNR은 `inf`입니다.
- 시간은 모델 로드, 합성, 지표 계산, 저장을 제외합니다. FFT는 별도의 분석 단계에서 계산합니다. 복원 처리 시간에는 FFT 방향 추정이 포함되지 않습니다. 시간 그래프는 성공한 복원 표본의 평균과 표본 수를 표시합니다. 품질 그래프는 참조 원본이 있는 합성 사례 중 입력과 세 복원 방법 모두에서 해당 지표를 얻은 공통 사례만 집계합니다. 각도 패널의 길이별 막대는 별도 조건이며, 전체 평균은 같은 공통 사례를 고르게 한 건씩 반영합니다. 무한대 PSNR 평균은 ∞로 표시합니다.
- 실패한 방법은 CSV에 원인을 기록합니다. 방향 추정 불가 또는 정답 PSF 부재로 생략한 경우는 `skipped_…`, 실행 실패는 `failed`로 구분합니다. 실패 행이 있으면 명령은 종료 코드 1을 반환합니다.

GUI에서는 CLI 인자 해석을 거치지 않고
`motion_blur.batch.run_batch(BatchOptions(input=Path(...), output=Path(...)))`를
호출할 수 있습니다. 이 함수는 동기 실행이므로 GUI에서는 작업 스레드에서
호출합니다. 실패 행이 있으면 1을 반환하고, 실행 자체가 불가능하면 예외를
전달합니다.

## 임계값 및 복원 품질 검증

라벨 CSV는 `path,label,split` 열을 사용합니다. `label`은 `sharp` 또는
`blurred`, `split`은 `validation` 또는 `test`입니다. 각 분할에 두 라벨을
포함하고, 같은 원본에서 만든 파생 영상은 모두 같은 분할에 넣습니다.
Laplacian 후보는 검증 분할에서 선택하고 테스트 분할에서 선택값만 평가합니다.
FFT 그래프의 후보 반환률은 각도 정확도나 모션 블러 확률을 뜻하지 않습니다.

```powershell
python -m motion_blur.validate_thresholds labels.csv results/thresholds
python -m motion_blur.wiener_sweep test_images/public/camera.png results/wiener-camera --angle 45 --length 15
python -m motion_blur.check_nafnet test_images/public/camera.png results/nafnet-camera --device cpu --max-size 384
python -m motion_blur.check_nafnet test_images/captured_blurred/blur.jpg results/nafnet-real --real --device cuda
python -m motion_blur.experiment wiener test_images/captured_blurred/blur.jpg results/manual.png --angle 30 --length 15
python -m motion_blur.batch test_images results/batch-manual --quick --real-angle 30 --real-length 15
python main.py test_images/public/camera.png --analyze-only --laplacian-low 300 --laplacian-high 1000 --fft-confidence 0.2
```

배치의 수동 각도는 촬영 블러 유형에만 적용하며 자동 추정 결과와 별도입니다.
Wiener K 실험은 정답 PSF, 추정 방향 PSF, 방향·길이 오차 PSF를 구분하고
복원 PNG, 원본과의 절대 차 PNG, CSV와 그래프를 저장합니다. NAFNet 점검은
같은 관측 영상을 원본 크기와 명시한 축소 크기로 각각 실행합니다. 축소
출력은 지표 비교에 한해 원본 크기로 보간합니다. CUDA에서는 최고 allocated/
reserved 바이트를 기록하며, 메모리가 부족해도 해상도를 자동으로 낮추지
않습니다. `--real`은 참조 지표를 비워 둡니다.
[실험 수치 및 남은 검증](docs/followup-validation.md)

## 직접 구현 코어 검증

합성 시 `--noise-sigma`는 정규화된 픽셀 범위의 가우시안 표준편차이며,
기본값 0은 기존 무잡음 합성입니다. `--seed`는 `psf.json`에 기록됩니다.
Wiener와 Richardson–Lucy(RL)는 CPU에서 실행되며 경계 처리를 선택할 수 있습니다.

```powershell
python -m motion_blur.experiment synthesize test_images/public/camera.png results/core-synth --angle 45 --length 15 --noise-sigma 0.01 --seed 702
python -m motion_blur.experiment rl results/core-synth/blurred.png results/rl.png --psf results/core-synth/psf.json --iterations 20 --boundary taper
python -m motion_blur.experiment wiener results/core-synth/blurred.png results/wiener.png --angle 45 --length 15 --boundary reflect
python -m motion_blur.experiment core-compare test_images/public/camera.png results/core-study --angle 45 --length 15 --max-size 256 --noise-sigma 0.01 --seed 702
python -m pip install pytest
python -m pytest -q
```

`core-compare`는 같은 블러 입력으로 `wrap`, `reflect`, `taper`를 모두
실행합니다. 기본 RL 반복 횟수는 10·20·40으로 비교하며, 새 결과 디렉터리에
PNG와 `metrics.json`을 저장합니다.
[수학적 원리와 경계 실험 결과](docs/classical-core-validation.md)

## 자동 PSF 후보로 Wiener·RL 복원

자동 추정은 긴 변 최대 512px에서 FFT 또는 그래디언트 방향과
켑스트럼 길이 후보를 구합니다. 원본 픽셀 단위로 변환한 PSF 하나를
두 CPU 복원에 사용합니다. 자동 후보가 부족하면 결과 JSON에 이유를
기록하고, 수동 각도나 길이를 지정해 다시 실행할 수 있습니다.

```powershell
python -m motion_blur.auto_restore test_images/captured_blurred/blur.jpg results/auto-fft
python -m motion_blur.auto_restore test_images/captured_blurred/blur.jpg results/auto-gradient --direction gradient
python -m motion_blur.auto_restore test_images/captured_blurred/blur.jpg results/manual-psf --angle 31 --length 15 --k 0.015 --iterations 8
python -m tests.validate_psf_estimator test_images/public/camera.png test_images/public/chelsea.png test_images/public/astronaut.png | Set-Content -Encoding utf8 results/psf-validation.csv
```

새 출력 디렉터리에 `psf_estimate.json`과, 복원에 성공한 경우
`wiener.png`, `rl.png`를 저장합니다. 참조 없는 사진에서는 PSNR과 SSIM을
계산하지 않습니다. [추정 방법과 검증 결과](docs/psf-estimation-validation.md)

## 4단계 정량 복원 비교

직접 구현과 scikit-image를 같은 입력·PSF로 비교하는 별도 명령입니다.
기존 실험 명령과 복원 기본값은 그대로 사용할 수 있습니다.

```powershell
python -m pip install -r requirements-quantitative.txt
python -c "from pathlib import Path; from motion_blur.batch import download_public; download_public(Path('test_images/public'))"
python -m motion_blur.quantitative test_images results/quant-quick --quick --max-size 128
python -m motion_blur.quantitative test_images results/quant-full --nafnet --device cpu
python -m motion_blur.quantitative test_images results/quant-full --nafnet --device cpu --resume
```

이미 공개 샘플이 있으면 공개 이미지 준비 명령을 생략합니다. 입력 하위 폴더는
`public`, `captured_sharp`, `captured_blurred`입니다. Full은 선명 이미지마다
4 길이 × 4 방향 × 4 잡음, 총 64조건을 원본 크기에서 실행합니다.
`--nafnet`은 대표 조건과 참조 없는 사진에서만 선택적으로 AI를 비교합니다.
AI 의존성이나 가중치가 없으면 해당 행만 생략 사유를 기록합니다.

결과 폴더에는 전체 `results.csv`, 품질·K·반복 횟수·추정 오차 PNG,
`comparisons/`의 정성 비교, `cases/`의 완료 캐시가 저장됩니다.
중단 뒤에는 같은 설정과 실행 환경에서 `--resume`을 사용합니다. 원본,
해상도, 옵션, 코드, 라이브러리 버전 또는 모델 가중치가 달라지면 새 결과
폴더를 사용합니다. 하나의 결과 폴더는 한 프로세스에서 실행합니다.
[비교 조건과 검증 결과](docs/quantitative-validation.md)

## GoPro·RealBlur 소규모 쌍과 블라인드 PSF 후보 비교

GoPro 또는 RealBlur 전체 다운로드는 필요하지 않습니다. 사용할 원본 쌍만
선택하고 다음 예처럼 CSV에 지정합니다. 경로는 CSV 위치를 기준으로
해석합니다. 이미 데이터가 다른 폴더에 있으면 CSV 경로만 조정합니다.

```text
test_images/paired/
├── pairs.csv
├── gopro/scene01/blur/000001.png
├── gopro/scene01/sharp/000001.png
├── realblur-j/scene02/blur/001.png
└── realblur-j/scene02/sharp/001.png
```

```csv
dataset,id,blur,sharp
gopro,gopro-01,gopro/scene01/blur/000001.png,gopro/scene01/sharp/000001.png
realblur-j,real-j-01,realblur-j/scene02/blur/001.png,realblur-j/scene02/sharp/001.png
```

`realblur-r`과 `realblur`도 지원합니다. 영상 쌍은 원래 데이터셋의
동일 샘플과 정렬된 해상도를 사용해야 합니다. 커맨드는 로컬 파일만 읽습니다.
[GoPro 데이터셋](https://seungjunnah.github.io/Datasets/gopro.html) · [RealBlur 데이터셋](https://github.com/rimchang/RealBlur)

```powershell
python -m pip install -r requirements-batch.txt
python -m motion_blur.candidate_experiment results/candidates-paired --manifest test_images/paired/pairs.csv --nafnet --device cpu
python -m motion_blur.candidate_experiment results/candidates-synthetic --synthetic test_images/public/camera.png --max-size 256
python -m motion_blur.candidate_experiment results/candidates-rl --manifest test_images/paired/pairs.csv --method rl --iterations 8
```

기본은 입력 해상도이며 `--max-size`는 축소 검증용입니다. `--method`는
후보를 채점할 때 사용할 복원이며, 최종 후보로 Wiener와 RL을 모두
비교합니다. 별도 `--observed` 입력에는 참조 지표를 쓰지 않습니다.
`--nafnet`은 기존 모델과 비교가 가능한 경우에만 실행합니다.

새 출력 폴더에는 후보별 점수·PSNR·SSIM과 실패 플래그를 담은
`results.csv`, 비교 PNG, 점수와 실제 품질의 관계를 그린
`candidate_quality.png`가 저장됩니다. GoPro·RealBlur에서는 합성 PSF
오차 열을 비워 둡니다. [선택 경계, 점수와 실패 결과](docs/blind-psf-candidates.md)
