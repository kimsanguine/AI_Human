# Daily AI Paper Recommendations

> **Date:** 2026-10-10
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Neural Network Fundamentals and Training

---

## Paper 1 (Classic): Mixed Precision Training
- **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu
- **Year:** 2017 (ICLR 2018)
- **arXiv:** https://arxiv.org/abs/1710.03740
- **PDF:** [./mixed-precision-training-micikevicius-2017.pdf](./mixed-precision-training-micikevicius-2017.pdf)
- **Citation Count:** ~2,500+

### 요약
신경망의 가중치·활성값·그래디언트를 16비트 부동소수점(FP16)으로 저장하고 연산하면서도, 정확도 손실이나 하이퍼파라미터 변경 없이 학습할 수 있는 방법론을 제시한 논문입니다. 메모리 사용량을 약 절반으로 줄이고, Tensor Core를 갖춘 GPU에서 연산 속도를 크게 높였습니다. 오늘날 거의 모든 대규모 모델 학습(FP16/BF16 AMP)의 출발점입니다.

### 핵심 기여
- **FP32 마스터 가중치(master weights):** 업데이트는 FP32 사본에 누적해, 작은 그래디언트 업데이트가 FP16에서 0으로 사라지는 문제를 방지
- **Loss Scaling:** 손실 값을 일정 배수로 키워 역전파시 작은 그래디언트가 FP16 표현 범위 아래로 언더플로되지 않도록 함 (이후 동적 loss scaling으로 발전)
- **FP16 곱셈 + FP32 누산(accumulation):** 내적·리덕션 연산은 FP32로 누적해 수치 오차를 억제
- CNN, RNN, GAN, 음성인식, 기계번역 등 다양한 과제에서 FP32와 동등한 정확도를 실험적으로 입증

### 이 논문이 중요한 이유
"모델을 더 크게, 더 빠르게 학습시키는 것"은 결국 메모리와 연산 효율의 문제입니다. PyTorch의 `torch.cuda.amp`, BF16 학습, 최근의 FP8 학습(DeepSeek-V3 등)까지 모두 이 논문의 세 가지 원칙(마스터 가중치, 스케일링, 고정밀 누산)을 계승합니다. 학습 비용과 GPU 사용 효율을 이해하려는 AI 엔지니어라면 반드시 알아야 할 기본기입니다.

### 사전 지식
- IEEE 754 부동소수점 표현(지수부·가수부, FP32/FP16/BF16 차이)
- 역전파와 SGD 기반 가중치 업데이트
- GPU 메모리 구성(가중치, 활성값, 옵티마이저 상태)에 대한 기본 이해

### 관련 논문
- [Training Deep Neural Networks with 8-bit Floating Point Numbers (Wang et al., 2018)](https://arxiv.org/abs/1812.08011)
- [A Study of BFLOAT16 for Deep Learning Training (Kalamkar et al., 2019)](https://arxiv.org/abs/1905.12322)
- [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054)

### 실무 적용
- PyTorch `autocast` + `GradScaler`, Hugging Face `Trainer(fp16=True / bf16=True)` 설정 한 줄로 학습 속도 1.5~3배, 메모리 절감 효과를 얻을 수 있습니다.
- 학습 중 loss가 NaN/Inf로 튀는 문제를 디버깅할 때, loss scale·오버플로 원인을 이해하는 데 직접적으로 쓰입니다.
- 파인튜닝 비용 산정(GPU 몇 장, 몇 시간)이나 FP8 학습 도입 여부 같은 인프라 의사결정의 근거가 됩니다.

---

## Paper 2 (Classic): Gaussian Error Linear Units (GELUs)
- **Authors:** Dan Hendrycks, Kevin Gimpel
- **Year:** 2016
- **arXiv:** https://arxiv.org/abs/1606.08415
- **PDF:** [./gelu-hendrycks-2016.pdf](./gelu-hendrycks-2016.pdf)
- **Citation Count:** ~5,000+

### 요약
입력의 부호로 값을 "자르는" ReLU와 달리, 입력값 x에 그 값이 표준정규분포에서 차지하는 누적확률 Φ(x)를 곱해 부드럽게 가중하는 활성화 함수 GELU(x·Φ(x))를 제안했습니다. 이는 뉴런 입력을 확률적으로 그대로 통과시키거나 0으로 만드는 확률적 정규화기(dropout과 유사)의 기댓값으로 해석됩니다. 비전·NLP·음성 과제 전반에서 ReLU, ELU보다 좋은 성능을 보였습니다.

### 핵심 기여
- 활성화 함수와 확률적 정규화(dropout, zoneout)를 하나의 관점으로 연결한 이론적 동기 제시
- 미분 가능하고 비단조(non-monotonic)인 부드러운 곡선으로, 음수 영역에서도 작은 그래디언트를 유지
- 계산 효율을 위한 tanh/sigmoid 근사식 제공 (`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`)
- MNIST, CIFAR, TIMIT, Twitter POS 등 다양한 과제에서 ReLU/ELU 대비 일관된 성능 향상

### 이 논문이 중요한 이유
GELU는 BERT, GPT-2/3, ViT 등 Transformer 계열의 기본 활성화 함수가 되었고, 현재 LLM 표준인 SwiGLU/GeGLU의 직접적인 뿌리이기도 합니다. "왜 Transformer FFN은 ReLU 대신 GELU를 쓰나?"라는 질문에 답할 수 있어야 모델 아키텍처 선택과 커스터마이징을 제대로 할 수 있습니다.

### 사전 지식
- ReLU, ELU, sigmoid 등 기본 활성화 함수와 그 한계(Dying ReLU 등)
- 정규분포의 누적분포함수(CDF) 개념
- Dropout의 동작 원리

### 관련 논문
- [Rectified Linear Units Improve Restricted Boltzmann Machines / ReLU (Nair & Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- [Fast and Accurate Deep Network Learning by ELUs (Clevert et al., 2015)](https://arxiv.org/abs/1511.07289)
- [Searching for Activation Functions / Swish (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941)
- [GLU Variants Improve Transformer / SwiGLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

### 실무 적용
- Hugging Face 모델 config의 `hidden_act: "gelu"` / `"gelu_new"`(tanh 근사) 차이를 이해하면, 모델 변환·ONNX 내보내기·추론 엔진 포팅 시 수치 불일치 문제를 빠르게 해결할 수 있습니다.
- 소형 커스텀 모델(분류기, 임베딩 헤드)을 설계할 때 ReLU 대비 GELU/SwiGLU 선택의 근거로 활용됩니다.
- 양자화·커널 퓨전 시 근사식 선택이 정확도·속도 트레이드오프에 영향을 줍니다.

---

## Paper 3 (Recent): nGPT: Normalized Transformer with Representation Learning on the Hypersphere
- **Authors:** Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg (NVIDIA)
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2410.01131
- **PDF:** [./ngpt-normalized-transformer-loshchilov-2024.pdf](./ngpt-normalized-transformer-loshchilov-2024.pdf)
- **Citation Count:** ~100+

### 요약
임베딩, 어텐션·MLP 가중치 행렬, 히든 스테이트 등 Transformer의 모든 벡터를 단위 길이로 정규화해 초구(hypersphere) 위에서 학습하도록 재설계한 아키텍처입니다. 각 레이어의 어텐션·MLP 블록은 토큰 표현을 출력 방향으로 이동시키는 "변위"를 더하며, 그 보폭은 학습 가능한 "eigen learning rate"로 조절됩니다. 저자들은 시퀀스 길이에 따라 동일 정확도 도달에 필요한 학습 스텝을 4~20배 줄였다고 보고합니다.

### 핵심 기여
- 모든 표현을 초구 위에 두어, 행렬곱이 코사인 유사도로 해석되는 단순하고 일관된 기하학적 구조 제시
- Transformer를 "레이어당 두 단계씩 진행하는 초구 위의 다단계 옵티마이저"로 재해석
- LayerNorm/RMSNorm, weight decay, warmup 의존도를 줄이는 정규화 중심 설계
- 1B급 모델에서 동일 정확도 도달 학습 스텝 4~20배 단축 결과 보고

### 이 논문이 중요한 이유
오늘의 두 클래식(혼합 정밀도, 활성화 함수)이 "학습을 안정적이고 효율적으로 만드는 기본 부품"이라면, nGPT는 정규화·학습률·가중치 감쇠 같은 부품들을 하나의 기하학적 원리로 통합하려는 최신 시도입니다. AdamW의 저자인 Loshchilov가 참여해 옵티마이저 관점과 아키텍처 관점을 연결했다는 점에서, "학습 효율은 옵티마이저만의 문제가 아니다"라는 관점을 길러줍니다. 다만 대규모(수십B 이상) 재현과 산업 표준 채택은 아직 진행 중인 연구 단계라는 점을 감안하고 읽어야 합니다.

### 사전 지식
- Transformer 블록 구조(어텐션, FFN, residual, Pre-LN)
- LayerNorm/RMSNorm의 역할
- 코사인 유사도, 벡터 정규화, 학습률·weight decay의 의미

### 관련 논문
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
- [Transformers without Normalization (Zhu et al., 2025)](https://arxiv.org/abs/2503.10622)

### 실무 적용
- 소형 도메인 LM을 직접 사전학습할 때, 학습 스텝(=GPU 비용) 절감 후보 아키텍처로 실험해 볼 수 있습니다 (NVIDIA 공개 구현 참고).
- 임베딩이 초구 위에 있으므로 코사인 유사도 기반 검색·RAG 임베딩과 개념적으로 잘 맞습니다.
- "학습 불안정 → 정규화 위치/방식 변경"이라는 디버깅 사고 틀을 넓혀 줍니다.

---

## 추천 읽기 순서
1. **GELU (2016)** — 짧고 수식이 단순해 워밍업으로 적합. 활성화 함수와 정규화의 연결 고리를 먼저 잡습니다.
2. **Mixed Precision Training (2017)** — 학습을 "수치 표현과 하드웨어" 관점에서 보는 시야를 엽니다.
3. **nGPT (2024)** — 앞의 두 관점(표현의 기하학, 학습의 수치 안정성)을 바탕으로 최신 아키텍처 재설계를 읽습니다.

## 핵심 테이크어웨이
- **Q. 신경망 학습 효율은 무엇으로 결정되는가?** → 옵티마이저뿐 아니라 수치 정밀도(FP16/BF16/FP8), 활성화 함수, 정규화 구조가 함께 결정합니다.
- **Q. 정밀도를 낮추면 왜 학습이 깨지고, 어떻게 막는가?** → 작은 그래디언트의 언더플로와 업데이트 소실 때문이며, 마스터 가중치·loss scaling·고정밀 누산으로 해결합니다.
- **Q. 왜 Transformer는 GELU 계열을 쓰는가?** → 부드럽고 확률적 정규화와 연결된 비선형성이 깊은 모델의 최적화에 유리하기 때문이며, 이는 SwiGLU로 이어집니다.
- **Q. 다음 흐름은?** → nGPT처럼 "정규화를 아키텍처의 기하학으로 내재화"해 하이퍼파라미터 의존을 줄이는 방향이 탐구되고 있습니다.

## 다음 토픽과의 연결
다음 주제는 **CNN Architectures and Computer Vision**입니다. 오늘 다룬 활성화 함수와 학습 안정화 기법은 ResNet·ConvNeXt·ViT 같은 비전 아키텍처가 깊어질 수 있었던 기반입니다. 특히 혼합 정밀도 학습은 대규모 비전 모델 학습의 필수 요소이고, GELU는 ViT와 ConvNeXt의 기본 활성화 함수로 그대로 이어집니다.
