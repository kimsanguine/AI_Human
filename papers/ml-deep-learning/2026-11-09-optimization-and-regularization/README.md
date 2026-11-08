# Daily AI Paper Recommendations

> **Date:** 2026-11-09
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
- **Authors:** Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1706.02677
- **PDF:** [./large-minibatch-sgd-goyal-2017.pdf](./large-minibatch-sgd-goyal-2017.pdf)
- **Citation Count:** ~4,000+

### 요약
배치 크기를 256에서 8,192로 32배 키워도 정확도 손실 없이 ResNet-50을 학습할 수 있음을 보인 논문입니다. 핵심은 "배치를 k배 키우면 학습률도 k배 키운다"는 **선형 스케일링 규칙**과, 초기 몇 에폭 동안 학습률을 서서히 올리는 **점진적 워밍업(gradual warmup)** 입니다. 그 결과 256개 GPU에서 ImageNet 학습을 29시간 → 1시간으로 줄였습니다.

### 핵심 기여
- 선형 학습률 스케일링 규칙(Linear Scaling Rule)의 경험적·이론적 근거 제시
- 학습 초기 불안정을 해결하는 gradual warmup 전략 제안 — 오늘날 거의 모든 LLM 학습 스케줄의 표준
- 분산 학습 구현 시 흔한 함정(weight decay 스케일링, momentum 보정, BN 통계 계산 단위 등)을 체계적으로 정리

### 이 논문이 중요한 이유
"왜 LLM 학습 스케줄은 항상 warmup으로 시작할까?"라는 질문의 출발점이 이 논문입니다. 배치 크기와 학습률의 관계를 이해하지 못하면 GPU를 늘렸을 때 성능이 오히려 떨어지는 현상을 디버깅할 수 없습니다. 분산 학습을 다루는 AI 엔지니어의 기본 상식입니다.

### 사전 지식
- SGD + Momentum의 업데이트 식
- 학습률 스케줄(step decay, cosine)
- Batch Normalization과 데이터 병렬(Data Parallel) 학습의 기본 개념

### 관련 논문
- [Large Batch Training of Convolutional Networks / LARS (You et al., 2017)](https://arxiv.org/abs/1708.03888)
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (Keskar et al., 2016)](https://arxiv.org/abs/1609.04836)
- [An Empirical Model of Large-Batch Training (McCandlish et al., 2018)](https://arxiv.org/abs/1812.06162)

### 실무 적용
- GPU 수를 늘려 글로벌 배치를 키울 때 학습률을 비례 조정하고 warmup 구간을 함께 늘리는 것이 기본 레시피
- HuggingFace `Trainer`의 `warmup_ratio`, `warmup_steps` 설정이 바로 이 아이디어
- 파인튜닝 시 loss가 초반에 튀거나 NaN이 나면 warmup 부족을 먼저 의심

---

## Paper 2 (Classic): Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (LAMB)
- **Authors:** Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh
- **Year:** 2019 (ICLR 2020)
- **arXiv:** https://arxiv.org/abs/1904.00962
- **PDF:** [./lamb-large-batch-you-2019.pdf](./lamb-large-batch-you-2019.pdf)
- **Citation Count:** ~1,500+

### 요약
Paper 1의 선형 스케일링은 CNN에선 잘 통했지만 BERT 같은 Transformer에선 깨졌습니다. 이 논문은 Adam에 **레이어별 신뢰 비율(layer-wise trust ratio)** 을 곱해 각 레이어의 업데이트 크기를 가중치 크기에 맞춰 정규화하는 **LAMB** 옵티마이저를 제안합니다. 배치 32K로 BERT를 3일 → 76분 만에 학습하면서도 성능 손실이 없었습니다.

### 핵심 기여
- LARS(SGD 기반 레이어별 적응)를 Adam 계열로 확장한 LAMB 제안
- 레이어별 적응 학습률의 비볼록(non-convex) 수렴성 분석
- 하이퍼파라미터 튜닝 거의 없이 대규모 배치 BERT 사전학습 성공 — "Transformer 대배치 학습" 가능성을 입증

### 이 논문이 중요한 이유
"Adam이면 다 되는 것 아닌가?"라는 가정을 깨는 논문입니다. 레이어마다 그래디언트와 가중치의 스케일이 크게 다르다는 관찰은 이후 Muon, Shampoo, Adam-mini 같은 **구조 인식(structure-aware) 옵티마이저** 계보의 출발점이 됩니다.

### 사전 지식
- Adam/AdamW 업데이트 식 (1차·2차 모멘트, bias correction)
- Paper 1의 선형 스케일링 규칙과 warmup
- BERT 사전학습 구조(MLM)

### 관련 논문
- [Large Batch Training of Convolutional Networks / LARS (You et al., 2017)](https://arxiv.org/abs/1708.03888)
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

### 실무 적용
- NVIDIA Megatron/Apex, DeepSpeed 등에 `FusedLAMB`로 구현되어 대규모 사전학습에서 사용
- 배치를 크게 키워야 하는 TPU Pod·멀티노드 환경에서 AdamW 대안으로 검토
- 레이어별 업데이트 비율(‖Δw‖/‖w‖)을 모니터링 지표로 쓰는 습관 — 학습 불안정 진단에 유용

---

## Paper 3 (Recent): Adam-mini: Use Fewer Learning Rates To Gain More
- **Authors:** Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, Ruoyu Sun
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2406.16793
- **PDF:** [./adam-mini-zhang-2024.pdf](./adam-mini-zhang-2024.pdf)
- **Citation Count:** ~150+

### 요약
Adam은 파라미터마다 개별 학습률(2차 모멘트 v)을 저장하느라 모델 크기의 2배 메모리를 추가로 씁니다. Adam-mini는 Transformer의 **Hessian이 블록 대각(block-diagonal) 구조**라는 관찰에 기반해, 파라미터 블록(예: attention head, 뉴런 단위)마다 v의 평균 하나만 학습률로 사용합니다. 그 결과 v의 99.9% 이상을 제거해 옵티마이저 메모리를 약 50% 줄이면서도 AdamW와 동등 이상의 성능을 냈습니다.

### 핵심 기여
- Hessian 블록 구조에 근거한 파라미터 분할 원칙 제시
- 125M~7B(버전에 따라 13B) LLM의 사전학습·SFT·RLHF 전반에서 AdamW와 동등 이상 성능
- Llama2-7B 사전학습 시 처리량 49.6% 향상 (GPU 간 통신·메모리 절감 효과)

### 이 논문이 중요한 이유
Adam 원저자(Kingma)가 공저자로 참여해 "Adam의 per-parameter 학습률은 대부분 중복"이라는 결론을 내렸다는 점이 상징적입니다. Paper 2의 "레이어별 적응"을 더 세밀한 "블록별 적응"으로 발전시킨 연장선이며, 한정된 GPU로 LLM을 학습·파인튜닝해야 하는 팀에게 직접적인 비용 절감 수단입니다.

### 사전 지식
- Adam의 2차 모멘트(v)의 역할과 메모리 비용
- Hessian, 조건수(condition number)의 직관적 의미
- Transformer 파라미터 구성(Q/K/V, MLP, embedding)

### 관련 논문
- [Symbolic Discovery of Optimization Algorithms / Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (Shazeer & Stern, 2018)](https://arxiv.org/abs/1804.04235)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507)
- [SOAP: Improving and Stabilizing Shampoo using Adam (Vyas et al., 2024)](https://arxiv.org/abs/2409.11321)

### 실무 적용
- 공식 구현(github.com/zyushun/Adam-mini)을 `torch.optim.AdamW` 대신 drop-in으로 교체 가능
- 동일 GPU에서 더 큰 배치 또는 더 큰 모델 학습 → 실험 회전율 향상
- 옵티마이저 상태가 체크포인트 크기의 큰 비중을 차지하므로 저장·전송 비용도 함께 절감

---

## 추천 읽기 순서
1. **Goyal et al. (2017)** — "배치를 키우면 무엇이 깨지는가"라는 문제 정의와 warmup의 기원 파악
2. **You et al. (2019, LAMB)** — 그 해법이 Transformer에선 왜 부족했고, 레이어별 적응이 어떻게 해결했는지 이해
3. **Zhang et al. (2024, Adam-mini)** — 적응의 단위를 "파라미터 → 레이어 → 블록"으로 재설계하며 메모리까지 줄이는 최신 흐름 확인

## 핵심 테이크어웨이
- **배치 크기·학습률·warmup은 한 세트다.** 하나를 바꾸면 나머지도 함께 조정해야 한다.
- **"적응의 단위(granularity)"가 옵티마이저 설계의 핵심 질문이다.** 전역(SGD) → 파라미터별(Adam) → 레이어별(LARS/LAMB) → 블록별(Adam-mini)로 진화해 왔다.
- **최적화는 곧 비용 문제다.** 학습 시간(Paper 1, 2)과 메모리(Paper 3)를 줄이는 것이 실무에서 옵티마이저 선택의 실질적 기준이다.

## 다음 토픽과의 연결
다음 주제는 **Transfer Learning and Foundation Models**입니다. 오늘 다룬 대배치 학습 기법(warmup, LAMB)과 메모리 효율 옵티마이저(Adam-mini)는 파운데이션 모델을 "만들고(사전학습)" "적응시키는(파인튜닝)" 단계의 인프라 기반입니다. 사전학습된 모델이 어떻게 다운스트림 태스크로 전이되는지 볼 때, 오늘 배운 학습률 스케줄과 옵티마이저 선택이 왜 파인튜닝 품질을 좌우하는지 연결해 보세요.
