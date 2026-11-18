# NAFNet 모션 블러 복원

흔들린 사진 한 장을 공식 NAFNet-GoPro-width32 모델로 복원해 PNG로 저장하는 Python CLI입니다. 자동 블러 분석과 보정 여부 판단은 인터페이스 단계이며, CLI는 지정한 사진에 복원을 적용합니다.

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

`motion_blur/_nafnet/`에는 공식 평가 설정에 맞춘 추론 코드와 라이선스가 포함되어 있습니다. [NAFNet 공식 평가 설정](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/options/test/GoPro/NAFNet-width32.yml)
