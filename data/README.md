# 로컬 입력 데이터 배치

이 폴더에는 경로 안내와 예제 CSV만 둡니다. 원본, 압축 데이터셋,
동영상, 모델 가중치는 로컬에 보관합니다. `pairs.example.csv`를 복사해
실제 사용할 소수의 쌍 경로를 지정하세요. CSV 경로는 그 파일의 위치
기준이며, 기존 4열 CSV도 사용할 수 있습니다.
[GoPro 데이터셋](https://seungjunnah.github.io/Datasets/gopro.html) · [RealBlur 데이터셋](https://github.com/rimchang/RealBlur)

```text
data/
├── gopro/GOPRO_Large/test/<sequence>/{blur,blur_gamma,sharp}/
├── realblur/<variant>/<scene>/
├── captured/  (사용자 원본을 둘 때 생성)
└── web_blur/  (참조 없는 공개 입력을 둘 때 생성)
```

GoPro 자동 탐색은 시퀀스 안에서 동일한 파일명의 PNG를 매칭합니다.
같은 이름의 sharp가 없는 입력은 로그와 함께 생략합니다. `blur`가
기본이며, `blur_gamma` 및 `both`를 선택할 수 있습니다. 조건별 한도는
이름순으로 매칭된 쌍에 적용됩니다. RealBlur는 사용자 manifest로
정확히 매칭하고, captured·web_blur는 기존 `--observed`에 파일을 전달합니다.

## 다운로드 없는 검증

아래 스크립트는 작은 정수 시드 합성 영상과 manifest를 만듭니다.
자동 탐색 및 manifest의 데이터 경로를 시험할 수 있습니다. 이 출력의
GoPro·RealBlur 표기는 폴더 형식 검증용이며 실제 벤치마크 자료가 아닙니다.

```powershell
python -m tests.make_paired_fixture results/pair-fixture
python -m motion_blur.candidate_experiment results/pair-native-check --gopro-root results/pair-fixture/gopro --gopro-condition both --gopro-limit 1
python -m motion_blur.candidate_experiment results/pair-manifest-check --manifest results/pair-fixture/pairs.csv
```

다음 구조 정리에서는 촬영·웹 입력의 메타데이터(촬영 조건, 출처, 라이선스)를
정하고, 재현 명령이나 실험 설정을 관리할 `experiments/`의 역할을
확정합니다. 기존 실행 모듈은 `motion_blur/`, 테스트·검증용 생성기는
`tests/`, 문서는 `docs/`, 출력은 Git에서 제외된 `results/`에 있습니다.
