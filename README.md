# NAFNet 모션 블러 복원

사진 한 장의 밝기, 노출 분포, 선명도를 분석하고 공식 NAFNet-GoPro-width32 모델로 복원해 PNG로 저장하는 Python CLI입니다. 보정 여부 판단은 별도 인터페이스이며, 복원 명령은 분석 지표를 출력한 뒤 복원을 진행합니다.

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

글자, 직선, 반복 무늬도 방향성 점수를 높일 수 있습니다. FFT 결과는 분석
로그에 사용되며 보정 판단이나 NAFNet 추론에 전달하지 않습니다.
[합성 PSF 테스트 및 결과표](docs/fft-direction-validation.md)

`motion_blur/_nafnet/`에는 공식 평가 설정에 맞춘 추론 코드와 라이선스가 포함되어 있습니다. [NAFNet 공식 평가 설정](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/options/test/GoPro/NAFNet-width32.yml)
