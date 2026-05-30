# Daily AI Paper Recommendations

> **Date:** 2026-05-31
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): SGDR: Stochastic Gradient Descent with Warm Restarts
- **Authors:** Ilya Loshchilov, Frank Hutter
- **Year:** 2016 (ICLR 2017)
- **arXiv:** https://arxiv.org/abs/1608.03983
- **PDF:** [./sgdr-warm-restarts-loshchilov-2016.pdf](./sgdr-warm-restarts-loshchilov-2016.pdf)
- **Citation Count:** approx. 9,000+

### 요약
학습률(learning rate)을 단조롭게 감소시키는 대신, 주기적으로 학습률을 높은 값으로 "재시작(warm restart)"하고 코사인 함수 형태로 다시 줄여나가는 스케줄을 제안한다. 이 단순한 변경만으로 학습이 손실 표면의 더 나은 영역으로 빠져나갈 수 있게 되어, 같은 학습 시간 안에서 더 빠르고 더 좋은 일반화 성능을 얻는다. CIFAR-10/100에서 당시 SOTA를 갱신했다.

### 핵심 기여
- 코사인 어닐링(cosine annealing) 기반의 주기적 학습률 재시작 스케줄(SGDR)을 제안
- 재시작 주기를 점점 늘려가는(T_mult) 방식으로 anytime 성능(언제 멈추든 좋은 성능)을 확보
- 각 재시작 직전의 모델들을 모아 사실상 "스냅샷 앙상블"을 무료로 얻을 수 있음을 보임

### 이 논문이 중요한 이유
오늘날 거의 모든 딥러닝 학습 파이프라인에서 표준으로 쓰이는 "cosine learning rate schedule"의 출발점이다. LLM, 비전 모델, 디퓨전 모델 사전학습까지 대규모 학습의 기본 스케줄로 자리잡았기 때문에, AI 엔지니어가 하이퍼파라미터를 다룰 때 반드시 이해해야 하는 개념이다.

### 사전 지식
- SGD와 모멘텀, 학습률(learning rate)이 학습에 미치는 영향
- 손실 표면(loss landscape)과 지역 최소값(local minima) 개념
- 기본적인 step decay, exponential decay 등 기존 학습률 스케줄

### 관련 논문
- [Cyclical Learning Rates for Training Neural Networks (Smith, 2015)](https://arxiv.org/abs/1506.01186)
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)

### 실무 적용
PyTorch의 `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`로 바로 사용 가능하다. Llama, GPT 계열 LLM 사전학습의 표준 스케줄(warmup + cosine decay)이 바로 이 논문 계열이며, 짧은 파인튜닝부터 수개월짜리 사전학습까지 폭넓게 적용된다.

---

## Paper 2 (Classic): Sharpness-Aware Minimization for Efficiently Improving Generalization (SAM)
- **Authors:** Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- **Year:** 2020 (ICLR 2021)
- **arXiv:** https://arxiv.org/abs/2010.01412
- **PDF:** [./sharpness-aware-minimization-foret-2020.pdf](./sharpness-aware-minimization-foret-2020.pdf)
- **Citation Count:** approx. 3,000+

### 요약
단순히 손실값이 낮은 지점이 아니라, 주변 이웃(neighborhood) 전체에서 손실이 균일하게 낮은 "평평한(flat)" 최소값을 찾도록 학습을 유도한다. 이를 위해 현재 파라미터 주변에서 손실을 최대로 만드는 방향으로 한 걸음 이동한 뒤 그 지점의 기울기로 업데이트하는 min-max 최적화를 수행한다. 평평한 최소값이 더 좋은 일반화로 이어진다는 직관을 실제 알고리즘으로 구현했다.

### 핵심 기여
- 손실의 "날카로움(sharpness)"을 명시적으로 최소화하는 최적화 목표(SAM)를 정식화
- 1차 근사를 통해 추가 1회의 forward-backward만으로 효율적으로 구현
- CIFAR, ImageNet, 파인튜닝 등 다양한 벤치마크에서 일반화 성능을 끌어올리고, 레이블 노이즈에 대한 강건성도 향상

### 이 논문이 중요한 이유
"왜 어떤 모델은 학습 손실이 같은데도 더 잘 일반화되는가"라는 근본 질문에 대한 손실 기하학(loss geometry) 관점의 실용적 답을 제시한다. 이후 ASAM, GSAM 등 수많은 후속 연구를 낳았고, 일반화를 정규화(regularization) 관점에서 이해하려는 AI 엔지니어에게 핵심 레퍼런스다.

### 사전 지식
- 일반화(generalization)와 과적합(overfitting)의 개념
- 손실 표면의 곡률(curvature), Hessian, 평평한/날카로운 최소값 직관
- 기울기 기반 최적화와 1차 테일러 근사

### 관련 논문
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (Keskar et al., 2016)](https://arxiv.org/abs/1609.04836)
- [ASAM: Adaptive Sharpness-Aware Minimization (Kwon et al., 2021)](https://arxiv.org/abs/2102.11600)

### 실무 적용
비전 모델 파인튜닝, 데이터가 적거나 레이블 노이즈가 있는 상황에서 일반화 성능을 끌어올리는 정규화 기법으로 활용된다. 다만 매 스텝 2배의 연산이 필요하므로, 효율형 변종(efficient SAM, LookSAM 등)과 함께 비용-효과를 따져 적용한다.

---

## Paper 3 (Recent): ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate
- **Authors:** Shohei Taniguchi, Keno Harada, Gouki Minegishi, Yuta Oshima, Seong Cheol Jeong, Go Nagahara, Tomoshi Iiyama, Masahiro Suzuki, Yusuke Iwasawa, Yutaka Matsuo
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2411.02853
- **PDF:** [./adopt-optimizer-taniguchi-2024.pdf](./adopt-optimizer-taniguchi-2024.pdf)
- **Citation Count:** approx. 60+ (빠르게 증가 중)

### 요약
Adam이 이론적으로 β₂ 값을 문제마다 잘 골라야만 수렴이 보장된다는 오랜 약점을, 두 가지 간단한 수정으로 해결한다. (1) 2차 모멘트 추정에서 현재 기울기를 제외하고, (2) 모멘텀 업데이트와 2차 모멘트 정규화의 순서를 바꾼다. 이 변경만으로 β₂ 값과 무관하게 O(1/√T)의 최적 수렴률을 보장하며, 기존 AMSGrad처럼 "기울기 노이즈가 유한하게 제한된다"는 비현실적 가정도 필요 없다.

### 핵심 기여
- 어떤 β₂ 선택에서도 최적 수렴률을 보장하는 새로운 옵티마이저 ADOPT를 제안
- Adam의 비수렴 원인을 이론적으로 규명하고, 최소한의 수정으로 해결
- 이미지 분류, 생성 모델, NLP, 강화학습 등 폭넓은 과제에서 Adam 및 그 변종 대비 우수한 성능 입증

### 이 논문이 중요한 이유
Adam/AdamW는 사실상 모든 딥러닝 학습의 기본 옵티마이저인데, 그 이론적 수렴 보장과 하이퍼파라미터 민감성 문제를 정면으로 다룬다. 옵티마이저를 "블랙박스"로 쓰지 않고 왜 작동/실패하는지 이해하려는 엔지니어에게, 고전(Adam)과 최신 연구를 잇는 다리 역할을 한다.

### 사전 지식
- Adam의 1차/2차 모멘트 추정과 β₁, β₂ 하이퍼파라미터의 역할
- 확률적 최적화의 수렴률(convergence rate) 개념과 O(1/√T) 표기
- AMSGrad 등 Adam 수렴성 개선 시도의 배경

### 관련 논문
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
- [On the Convergence of Adam and Beyond / AMSGrad (Reddi et al., 2018)](https://arxiv.org/abs/1904.09237)

### 실무 적용
공식 PyTorch 구현이 공개되어 있어 Adam/AdamW를 거의 그대로 대체해 시도해볼 수 있다. β₂ 튜닝에 민감하거나 학습이 불안정한 LLM·생성 모델 학습에서, 하이퍼파라미터 탐색 비용을 줄이는 대안 옵티마이저로 검토할 가치가 있다.

---

## 추천 읽기 순서
1. **SGDR (Paper 1)** — 학습률 스케줄이라는 가장 직관적이고 즉시 실무 적용 가능한 주제로 시작한다. 오늘날 cosine schedule의 뿌리를 이해한다.
2. **ADOPT (Paper 3)** — 옵티마이저 자체(Adam의 수렴성)로 한 단계 들어가, 학습률 스케줄과 옵티마이저가 어떻게 맞물리는지 본다.
3. **SAM (Paper 2)** — 최적화에서 일반화/정규화로 관점을 확장한다. "어디로 수렴하느냐"가 왜 중요한지 손실 기하학으로 마무리한다.

## 핵심 테이크어웨이
- **학습률 스케줄은 공짜 성능이다:** SGDR의 cosine warm restart는 추가 연산 없이 더 빠른 수렴과 더 나은 일반화를 준다. 대규모 학습의 사실상 표준.
- **수렴하는 지점의 "모양"이 일반화를 좌우한다:** SAM은 평평한 최소값을 명시적으로 찾아 일반화를 개선한다. 최적화와 정규화는 분리된 문제가 아니다.
- **기본 옵티마이저도 여전히 개선 여지가 있다:** ADOPT는 Adam의 이론적 약점을 최소 수정으로 해결하며, "당연하게 쓰던 도구"를 다시 검증하는 연구 태도의 중요성을 보여준다.
- **세 논문의 공통 질문:** 같은 모델·데이터라도 *어떻게* 최적화하느냐(스케줄·옵티마이저·정규화)에 따라 최종 성능이 크게 달라진다.

## 다음 토픽과의 연결
다음 토픽인 **Transfer Learning and Foundation Models**에서는 잘 최적화된 모델을 *어떻게 재사용·전이*하는지를 다룬다. 오늘 배운 안정적 최적화·일반화 기법은 대규모 사전학습 모델을 만드는 전제 조건이며, 좋은 일반화 성능(SAM)과 안정적 수렴(ADOPT·SGDR)이 곧 강력한 파운데이션 모델의 기반이 된다.
