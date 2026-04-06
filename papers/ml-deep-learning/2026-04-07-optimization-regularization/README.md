# Daily AI Paper Recommendations

> **Date:** 2026-04-07
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): Adam: A Method for Stochastic Optimization

- **Authors:** Diederik P. Kingma, Jimmy Ba
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1412.6980
- **PDF:** [./adam-kingma-ba-2014.pdf](./adam-kingma-ba-2014.pdf)
- **Citation Count:** ~100,000+

### 요약
Adam은 1차 그래디언트 정보만을 사용하는 확률적 최적화 알고리즘으로, 각 파라미터마다 적응적인 학습률(adaptive learning rate)을 계산한다. 그래디언트의 1차 모멘트(평균)와 2차 모멘트(분산)를 추적하여 편향을 보정(bias correction)하는 방식으로 안정적인 학습을 가능하게 한다. 메모리 효율적이고 하이퍼파라미터에 강건하여 딥러닝의 사실상 표준(de facto standard) 옵티마이저로 자리잡았다.

### 핵심 기여
- **적응적 학습률**: 각 파라미터에 대해 독립적으로 학습률을 조정하여 희소한(sparse) 그래디언트 환경에서도 효과적
- **편향 보정(Bias Correction)**: 모멘트 추정의 초기 편향을 수학적으로 보정하여 학습 초기 단계 안정화
- **낮은 메모리 요구사항**: SGD + Momentum 대비 추가 메모리 비용이 매우 작음 (O(n) 파라미터 수에 비례)

### 이 논문이 중요한 이유
Adam은 현재 거의 모든 딥러닝 프레임워크(PyTorch, TensorFlow, JAX)에 기본으로 내장되어 있으며, GPT, BERT, ResNet 등 현대 AI의 핵심 모델들이 Adam 또는 그 변형으로 학습되었다. 옵티마이저를 이해하지 않고는 모델 학습 문제를 디버깅할 수 없기 때문에 AI 엔지니어에게 필독 논문이다.

### 사전 지식
- 확률적 경사 하강법(SGD)과 미니배치 학습의 개념
- 그래디언트(gradient)와 역전파(backpropagation) 기초
- 지수 이동 평균(Exponential Moving Average, EMA) 개념
- 볼록 최적화(convex optimization) 기초 이론

### 관련 논문
- [Adaptive Subgradient Methods for Online Learning / AdaGrad (Duchi et al., 2011)](https://jmlr.org/papers/v12/duchi11a.html)
- [RMSProp (Tieleman & Hinton, 2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond / AMSGrad (Reddi et al., 2018)](https://arxiv.org/abs/1904.09237)

### 실무 적용
AI 서비스 개발 시 모델 학습 파이프라인의 기본 설정으로 Adam을 사용한다. LLM 파인튜닝(fine-tuning)에서는 AdamW(Adam + Weight Decay)가 표준이며, 학습률 스케줄러(warmup + cosine decay)와 함께 사용하면 성능이 극대화된다. 학습이 수렴하지 않거나 loss가 발산할 때 학습률(lr), beta1, beta2, epsilon 하이퍼파라미터를 조정하는 것이 일반적인 디버깅 방법이다.

---

## Paper 2 (Classic): Decoupled Weight Decay Regularization (AdamW)

- **Authors:** Ilya Loshchilov, Frank Hutter
- **Year:** 2017 (ICLR 2019 게재)
- **arXiv:** https://arxiv.org/abs/1711.05101
- **PDF:** [./adamw-loshchilov-hutter-2017.pdf](./adamw-loshchilov-hutter-2017.pdf)
- **Citation Count:** ~30,000+

### 요약
이 논문은 L2 정규화(L2 regularization)와 weight decay가 Adam 옵티마이저에서 동일하지 않다는 핵심 문제를 발견한다. 기존 Adam에 L2 정규화를 추가하면 적응적 학습률과 상호작용하여 weight decay의 효과가 파라미터마다 달라지는 버그가 있었다. 이를 해결하기 위해 weight decay를 그래디언트 업데이트와 분리(decouple)한 AdamW 알고리즘을 제안한다.

### 핵심 기여
- **L2 정규화 ≠ Weight Decay**: 적응적 옵티마이저에서 두 방법이 동등하지 않음을 수학적으로 증명
- **Decoupled Weight Decay**: weight decay를 그래디언트와 분리하여 적용함으로써 더 일관된 정규화 효과 달성
- **SGDW 및 AdamW 제안**: 분리된 weight decay를 SGD와 Adam에 각각 적용한 새로운 알고리즘 제시 및 실험적 우수성 입증

### 이 논문이 중요한 이유
GPT-2, GPT-3, BERT, LLaMA 등 현대의 거의 모든 대형 언어 모델(LLM)이 AdamW로 학습된다. 단순히 Adam의 버그를 고친 것처럼 보이지만, 실제로 정규화 효과가 크게 개선되어 과적합(overfitting) 없이 모델 일반화 성능을 높이는 핵심 기법이 되었다.

### 사전 지식
- Adam 옵티마이저 (arXiv:1412.6980) 이해 필수
- L2 정규화와 weight decay의 개념 및 차이
- 과적합(overfitting)과 정규화(regularization) 이론
- 학습률 스케줄링(Warmup, Cosine Annealing) 기초

### 관련 논문
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
- [Three Mechanisms of Weight Decay Regularization (Zhang et al., 2018)](https://arxiv.org/abs/1810.12281)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Fixing Weight Decay Regularization in Adam (Loshchilov & Hutter, 2018)](https://arxiv.org/abs/1711.05101)

### 실무 적용
LLM 사전학습(pre-training) 및 파인튜닝(fine-tuning) 프로젝트에서 AdamW는 기본 옵티마이저로 채택된다. Hugging Face Transformers 라이브러리에서도 `AdamW`가 기본 설정이며, weight_decay=0.01~0.1 범위에서 실험하는 것이 일반적이다. AI Dubbing/Avatar 모델 같은 생성 모델에서도 AdamW + cosine LR schedule 조합이 표준적으로 사용된다.

---

## Paper 3 (Recent): The Road Less Scheduled

- **Authors:** Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2405.15682
- **PDF:** [./schedule-free-defazio-2024.pdf](./schedule-free-defazio-2024.pdf)
- **Citation Count:** ~500+ (빠르게 증가 중)

### 요약
Schedule-Free 최적화는 학습률 스케줄(warmup, decay 등)을 완전히 제거하면서도 기존 스케줄 방식 대비 동등하거나 우수한 성능을 보이는 새로운 최적화 프레임워크다. 내부적으로 이터레이트 평균화(iterate averaging)와 모멘텀을 결합하는 방식으로 스케줄링의 효과를 내재화한다. 이 방법으로 개발된 Schedule-Free AdamW는 MLCommons 2024 AlgoPerf 알고리즘 효율성 챌린지의 Self-Tuning 트랙에서 우승했다.

### 핵심 기여
- **스케줄 제거**: 학습 종료 시점을 사전에 알 필요 없이 최적화 가능 — 연속 학습(continual learning) 시나리오에 이상적
- **통합 이론**: 스케줄링과 이터레이트 평균화를 하나의 프레임워크로 통합하는 새로운 이론 제시
- **추가 하이퍼파라미터 없음**: 기존 Adam 대비 추가 설정 없이 사용 가능하여 실용성 극대화

### 이 논문이 중요한 이유
현업에서 LLM 학습 시 학습률 스케줄 설계(총 학습 스텝 결정, warmup 비율 등)는 매우 번거롭고 실수가 잦은 작업이다. Schedule-Free 방식은 이 문제를 근본적으로 해결하여 지속적 학습(continual training)이나 온라인 학습 환경에서 특히 강력하다. 2024년 기준 가장 실용적인 새 옵티마이저 연구로 평가받는다.

### 사전 지식
- Adam 및 AdamW 옵티마이저 이해 필수
- 이터레이트 평균화(Polyak-Ruppert averaging) 개념
- 학습률 스케줄링 기법 (warmup, cosine decay, linear decay)
- 온라인 학습 및 비볼록 최적화(non-convex optimization) 기초

### 관련 논문
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [Sophia: A Scalable Stochastic Second-order Optimizer (Liu et al., 2023)](https://arxiv.org/abs/2305.14342)
- [Symbolic Discovery of Optimization Algorithms / Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675)
- [schedule_free GitHub Repository (Meta AI)](https://github.com/facebookresearch/schedule_free)

### 실무 적용
Agentic AI 제품처럼 모델을 지속적으로 업데이트하거나 스트리밍 방식으로 학습하는 환경에서 Schedule-Free AdamW는 큰 장점을 갖는다. 특히 총 학습 스텝을 미리 정할 수 없는 온라인 학습이나 점진적 파인튜닝 시나리오에서 스케줄 설계 없이 바로 적용할 수 있어 엔지니어링 비용을 크게 절감한다. PyTorch 기반 프로젝트에서 `pip install schedulefree`로 즉시 사용 가능하다.

---

## 추천 읽기 순서

1. **Adam (Paper 1)** → 적응적 학습률의 핵심 원리와 수식을 이해한다
2. **AdamW (Paper 2)** → Adam의 weight decay 문제를 파악하고, 현재 LLM 표준 옵티마이저를 이해한다
3. **Schedule-Free (Paper 3)** → 스케줄링의 번거로움을 해결한 최신 접근법을 학습한다

---

## 핵심 테이크어웨이

- **Adam은 딥러닝 최적화의 표준**이지만, L2 정규화 방식에 근본적인 결함이 있었다
- **AdamW는 그 결함을 수정**한 버전으로, 모든 현대 LLM 학습의 기본이다
- **Schedule-Free 최적화는 패러다임 전환**을 시도한다 — 학습률 스케줄 없이도 최고 성능 달성이 가능함을 보였다
- 실무에서는 **AdamW + cosine decay + warmup**이 여전히 가장 안전한 조합이나, 지속 학습 시나리오에서는 Schedule-Free AdamW가 강력한 대안이다

---

## 다음 토픽과의 연결

다음 Day 6 토픽인 **Transfer Learning and Foundation Models**에서는 오늘 학습한 AdamW와 학습률 스케줄링이 어떻게 대규모 사전학습(pre-training)과 파인튜닝(fine-tuning)에 적용되는지를 다룬다. 특히 ImageNet 사전학습 모델을 다운스트림 태스크에 전이할 때 학습률을 얼마나 작게 설정해야 하는지(learning rate warmup, layer-wise learning rate decay)와 같은 실무 기법들이 오늘 내용과 직결된다.
