# CSV를 이용한 MUSK 형태학 분류 학습

## Delta에서는 Slurm으로 제출

로그인 노드(dt-login01 등)에서 아래의 학습 Python 명령을 직접 실행하지 마세요.
로그인 노드는 편집·설치·작업 제출에 사용합니다.
GPU 계산은 `train_delta.sbatch`로 제출하거나 이미 할당받은 계산 노드에서만 실행합니다.

```bash
# 로그인 노드, 저장소의 musk 폴더에서
git pull origin main
accounts
sinfo -s
mkdir -p logs
sbatch train_delta.sbatch
squeue -u "$USER"
```

제출 파일은 공식 hackathon submit.sbatch를 참고해 bfqi-delta-gpu 계정,
gpuA100x4 파티션, GPU 1개, CPU 4개, RAM 32GB, 1시간으로 구성했습니다.
별도 reservation 옵션은 없습니다. 환경과 계정 접근은 서버에서 확인해야 합니다.
Slurm이 로그 파일을 먼저 열기 때문에 `mkdir -p logs`는 제출 전에 실행하세요.
완료 여부는 `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed`로 확인합니다.
취소는 `scancel JOBID`, 로그 확인은 `tail -f logs/musk-JOBID.out`입니다.

기본 환경은 공식 예제의 `module load pytorch-conda/2.8`입니다.
이 모듈에 MUSK 추가 의존성이 모두 있다는 보장은 없습니다.
공식 모델 requirements의 오래된 torch 버전을 이 공유 모듈에 덮어 설치하지 마세요.
기존에 MUSK 설치를 완료한 전용 환경이 있으면 그 Python의 절대 경로를 지정합니다.

```bash
export MUSK_PYTHON=/absolute/path/to/musk-env/bin/python
sbatch --export=ALL train_delta.sbatch
```

학습 전 작업 내부에서 CUDA와 MUSK import 검사를 실행합니다.
실패하면 logs/musk-JOBID.err의 오류를 확인하고 환경을 먼저 수정하세요.
Hugging Face 접근 승인과 로그인도 미리 완료해야 합니다.
원격 작업 제출 및 환경 동작 검증은 아직 수행하지 않았습니다.

발표자료의 권장 경로는 encoder를 고정해 임베딩을 한 번 추출한 뒤 CPU 작업에서
작은 분류층을 학습하는 것입니다. 현재 스크립트는 encoder를 고정하지만 임베딩을
epoch마다 다시 계산하므로 GPU 작업으로 제출합니다. 임베딩 캐싱은 아직 구현하지 않았습니다.
팀당 GPU 한 개를 사용하고 작업 시간을 짧게 유지하세요.

참고: https://github.com/ncsa/ashby-hackathon-2026/blob/main/submit.sbatch

## 데이터와 한계

입력 CSV: `data/tum_1000_histology_classification.csv` (이 패키지에 포함).
이미지 기본 경로: `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM`.
파일 경로는 이미지 폴더 + CSV의 `filename`으로 연결합니다.
`tum_index`, `global_dataset_index`는 모델 입력이나 정답에 사용하지 않습니다.

확정한 라벨 의미: `no=-1`, `unknown=0`, `yes=1`.
unknown은 미라벨링이 아니라 판단 불가라는 실제 정답입니다.
내부 CrossEntropy 인덱스는 각각 0, 1, 2이며 추론 시 1을 빼서 복원합니다.
5개 항목마다 독립적인 3분류 loss를 구해 평균합니다.

| 항목 | no | unknown | yes |
|---|---:|---:|---:|
| abnormal_gland_arch | 1 | 6 | 993 |
| nuclear_atypia | 0 | 6 | 994 |
| abnormal_nuclear_org | 0 | 6 | 994 |
| stromal_invasion | 320 | 6 | 674 |
| intraluminal_necrosis | 734 | 6 | 260 |

앞의 두 핵 관련 항목에는 음성 예제가 없고, gland architecture도 1개뿐입니다.
가중치 조정이나 반복 추출로 없는 음성 정보를 만들 수는 없습니다.
이 CSV로 학습 실행은 가능하지만 5항목 모두의 3분류 성능을 검증할 수는 없습니다.
병리 전문가가 확인한 음성 및 판단 불가 예제를 추가해야 합니다.
라벨이 AI로 생성됐다면 전문가 검토 및 독립 평가 라벨도 준비하세요.

## 설치

`SERVER_GUIDE.md`의 Python 3.10 환경 설치와 Hugging Face 로그인을 먼저 진행합니다.
아래 명령은 모두 프로젝트 루트에서 실행합니다.
공식 구현의 입력 해상도와 정규화를 사용합니다(384×384, mean/std=0.5).
단일 스케일 CLS 1024차원 특징을 사용하며 multi-scale은 꺼서 출력층 차원을 고정했습니다.

## 데이터 확인 (PyTorch 없이도 실행 가능)

```bash
python train_morphology.py \
  --csv data/tum_1000_histology_classification.csv \
  --check-data
```

전체 파일 존재 여부, 중복 파일명, 열과 라벨을 검사하고 분포를 출력합니다.
이미지 디코딩 문제는 학습 로더에서 오류로 보고됩니다.

## 1단계: encoder 고정, 새 출력층만 학습

원본 CSV에는 환자/슬라이드 ID가 없습니다.
아래는 동작 확인용 이미지 단위 80/20 분할입니다. 환자나 슬라이드가 겹칠 수 있으므로
논문용 독립 검증 성능으로 해석하지 마세요. 별도 테스트 세트도 아닙니다.

```bash
python train_morphology.py \
  --csv data/tum_1000_histology_classification.csv \
  --allow-patch-split \
  --output runs/head_pilot \
  --epochs 20 --batch-size 4 --workers 4 \
  --head-lr 0.001
```

클래스 불균형 보정 비교 실험은 `--class-weighting sqrt-inverse`를 추가하고
별도 출력 폴더를 지정하세요. 기본은 가중치 없는 cross entropy입니다.

환자 또는 슬라이드 매핑을 확보하면 CSV에 `patient_id` 또는 `slide_id` 열을 추가하고
`--allow-patch-split` 대신 `--group-column patient_id`를 사용하세요.
그룹별 무작위 분할이며 클래스별 비율을 보장하지 않습니다.
이미지별 고유 인덱스를 환자/슬라이드 ID 대신 사용하면 안 됩니다.

## 2단계: encoder도 함께 미세조정

아래는 같은 CSV·seed·분할 설정으로 1단계의 가중치에서 이어 학습합니다.
optimizer는 새로 시작하며 중단 지점 복원 기능은 아닙니다.
이 데이터 크기와 라벨 분포에서는 먼저 출력층 학습 결과를 확인하세요.

```bash
python train_morphology.py \
  --csv data/tum_1000_histology_classification.csv \
  --allow-patch-split \
  --finetune-encoder --init runs/head_pilot/best.pt \
  --output runs/encoder_pilot \
  --epochs 5 --batch-size 1 --workers 4 \
  --head-lr 0.0001 --encoder-lr 0.000001
```

GPU 메모리 부족 시 batch-size 1을 사용하세요. 그래도 부족하면 encoder 고정으로
돌아가거나 더 큰 GPU가 필요합니다. 메모리 사용량과 위 설정은 서버에서 검증하지 않았습니다.
CPU도 실행 경로는 있지만 학습 속도는 매우 느릴 수 있습니다.
위 epoch/lr은 시작 실험값이며 최적값이라는 근거는 없습니다.

이미 받은 공식 가중치가 있으면 `--pretrained /path/to/model.safetensors`를 추가합니다.
추가하지 않으면 공식 HF 저장소에서 접근 권한을 이용해 로드합니다.

## 결과

각 출력 폴더에 다음을 저장합니다.

- `best.pt`: 가장 낮은 unweighted validation loss의 전체 모델 가중치와 라벨 정의.
- `train.csv`, `val.csv`: 실제 분할 목록.
- `config.json`: 설정, 라벨 매핑, 각 분할의 클래스 수.
- `metrics.jsonl`: epoch별 loss, 항목별 accuracy, 3클래스 macro-F1, confusion matrix.

macro-F1은 no/unknown/yes 세 클래스를 고정하고 분모가 0인 클래스는 F1=0으로 둡니다.
클래스가 없는 평가셋의 점수는 반드시 support와 함께 해석하세요.
현재 다수 클래스만 예측해도 앞의 항목 accuracy가 높게 나올 수 있습니다.

추론 시:

```python
import torch
from morphology_head import load_pretrained

device = "cuda"
model = load_pretrained().to(device)
checkpoint = torch.load("runs/head_pilot/best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()
# images: 학습과 동일한 정규화, [B, 3, 384, 384]
# result = model.predict(images.to(device))
# print(result["labels"].cpu().tolist())
```

`best.pt`는 SERVER_GUIDE의 head-only 예시와 달리 전체 모델 checkpoint입니다.
학습 스크립트로 저장한 파일은 위 방식으로 불러오세요.

## 검증 범위

로컬에서는 CSV 전체 라벨 통계, 파싱·분할·오류 처리, 코드 문법을 확인했습니다.
서버 이미지, GPU, 승인된 MUSK 가중치가 없어 실제 학습은 실행하지 않았습니다.

공식 MUSK: https://github.com/lilab-stanford/MUSK
