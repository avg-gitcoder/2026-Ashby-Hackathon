# 서버 실행 안내

공식 MUSK 전체 소스와 `morphology_head.py`를 포함합니다.
가중치, 학습 데이터, 학습된 형태학 출력층은 포함하지 않습니다.
원본 프로젝트: https://github.com/lilab-stanford/MUSK
원본 모델의 이용 조건은 README.md 및 공식 모델 페이지를 확인하세요.

## 1. Linux 서버 설치

```bash
git clone https://github.com/avg-gitcoder/2026-Ashby-Hackathon.git
cd 2026-Ashby-Hackathon/musk
conda create -n musk python=3.10 -y
conda activate musk
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

서버의 NVIDIA 드라이버와 PyTorch/CUDA 호환성을 확인하세요.
원본 requirements.txt를 그대로 보존했으며 서버에서 설치 검증하지 않았습니다.

## 2. 가중치 접근 승인 및 로그인

https://huggingface.co/xiangjx/musk 에서 접근 승인을 받은 계정을 사용하세요.
토큰은 채팅이나 소스 파일에 적지 말고 서버의 로그인 프롬프트에 입력하세요.

```bash
python -c "from huggingface_hub import login; login()"
```

`load_pretrained()` 첫 실행 시 공식 가중치를 다운로드합니다.
원본 로더는 기본적으로 `~/.cache/model.safetensors`를 사용합니다.
해당 경로에 다른 모델 파일이 있으면 전용 캐시 경로를 지정하도록 원본 로더 호출을 조정하세요.

## 3. 출력층 사용 및 학습

프로젝트 루트에서 아래 Python 예시를 실행하세요.
입력 images는 공식 README의 전처리를 거친 `[B, 3, 384, 384]` 텐서입니다.

```python
import torch
from morphology_head import FEATURES, load_pretrained, morphology_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_pretrained(freeze_encoder=True).to(device)
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

# 실제 데이터 로더의 images, labels로 학습합니다.
# images: [B, 3, 384, 384], labels: [B, 5], 값은 -1/0/1
# model.train()
# optimizer.zero_grad()
# loss = morphology_loss(model(images.to(device)), labels.to(device))
# loss.backward()
# optimizer.step()

# 학습 완료 후 저장:
# torch.save(model.head.state_dict(), "morphology_head.pt")

# 추론 시 학습된 출력층 로드:
# model.head.load_state_dict(torch.load("morphology_head.pt", map_location=device))
# model.eval()
# result = model.predict(images.to(device))
# print(result["labels"].cpu().tolist())
```

출력층은 처음에 무작위 초기화됩니다. 학습 전 출력은 의미 있는 예측이 아닙니다.
학습 데이터 준비와 전체 학습 루프는 별도로 구성해야 합니다.

## 출력 규칙

항목 순서:
1. abnormal_gland_arch
2. nuclear_atypia
3. abnormal_nuclear_org
4. stromal_invasion
5. intraluminal_necrosis

`forward`: `[B, 5, 3]` logits.
`predict`: 각 항목의 세 클래스에 softmax를 적용하고 최댓값을 선택합니다.
`labels`: `[B, 5]`, -1/0/1 중 하나.
`probabilities`: `[B, 5, 3]`, 클래스 순서 -1, 0, 1.
`confidence`: `[B, 5]`, 선택한 클래스의 확률이며 보정된 신뢰도를 보장하지 않습니다.
동률이면 클래스 순서상 먼저 나오는 값을 선택합니다.
0도 학습하는 클래스이며 누락 라벨이나 확률 임계값을 뜻하지 않습니다.

## 검증 상태

추가 코드의 Python 문법 검사만 완료했습니다.
현 PC에 PyTorch가 없어 모델 실행 및 학습 검증은 수행하지 않았습니다.
현 PC에서 가중치 접근 시 401 오류가 발생했습니다.
