# Daily AI Paper Recommendations

> **Date:** 2026-09-16
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
- **Authors:** John Duchi, Elad Hazan, Yoram Singer
- **Year:** 2011
- **arXiv:** https://jmlr.org/papers/volume12/duchi11a/duchi11a.html (JMLR 12:2121-2159)
- **PDF:** [./adagrad-duchi-2011.pdf](./adagrad-duchi-2011.pdf)
- **Citation Count:** ~20,000+

### 요약
파라미터마다 서로 다른 학습률을 자동으로 부여하는 AdaGrad를 제안한 논문이다. 각 파라미터의 과거 그래디언트 제곱합을 누적해 그 제곱근으로 학습률을 나누어, 자주 업데이트되는 파라미터는 작게, 드물게 등장하는 파라미터는 크게 움직이도록 만든다. 온라인 볼록 최적화 관점에서 regret 상한을 수학적으로 증명해, "적응적 학습률"이 휴리스틱이 아니라 이론적 근거를 가진 기법임을 보였다.

### 핵심 기여
- 파라미터별(per-parameter) 적응적 학습률이라는 개념을 최초로 정립하고 이론적으로 정당화
- 그래디언트 제곱합 누적이라는 단순한 통계량만으로 2차 정보(곡률)를 근사하는 실용적 방법 제시
- 희소(sparse) 피처 환경에서 기존 SGD 대비 압도적으로 빠른 수렴을 증명 및 실험으로 입증
- RMSProp → Adadelta → Adam → AdamW로 이어지는 현대 옵티마이저 계보의 출발점

### 이 논문이 중요한 이유
오늘날 거의 모든 딥러닝 학습은 Adam 계열 옵티마이저로 돌아가는데, Adam의 분모에 들어가는 `sqrt(v_t)` 항의 아이디어가 바로 이 논문에서 나왔다. AdaGrad를 이해하지 못하면 Adam의 하이퍼파라미터(beta2, eps)가 실제로 무엇을 조절하는지 감으로만 만지게 된다. 동시에 AdaGrad는 "누적합이 단조 증가해서 학습률이 0으로 수렴해 버린다"는 명확한 한계를 가지는데, 이 실패 지점이 곧 이후 모든 옵티마이저의 설계 동기가 되었다. 즉, 성공보다 실패를 통해 더 많이 배우게 되는 필독 논문이다.

### 사전 지식
- SGD와 미니배치 학습의 기본 동작
- 볼록 최적화와 regret(후회) 개념, 그래디언트/서브그래디언트의 정의
- 희소 벡터 표현(bag-of-words, one-hot)과 왜 희소 피처에서 학습이 어려운지
- 기본 선형대수: 대각 행렬 근사, L2 노름

### 관련 논문
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
- [ADADELTA: An Adaptive Learning Rate Method (Zeiler, 2012)](https://arxiv.org/abs/1212.5701)
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)

### 실무 적용
추천 시스템, 광고 CTR 예측, 검색 랭킹처럼 피처가 극도로 희소한 도메인에서는 지금도 AdaGrad/FTRL 계열이 현역이다. 임베딩 테이블 학습(예: 수백만 개의 아이템 ID 임베딩)에서는 대부분의 행이 배치마다 거의 등장하지 않기 때문에, 파라미터별 적응 학습률이 없으면 롱테일 아이템의 임베딩이 사실상 학습되지 않는다. PyTorch의 `torch.optim.Adagrad`와 `sparse=True` 임베딩 조합이 그 전형적인 구현이다. 또한 LLM 파인튜닝에서 loss가 정체될 때 "혹시 실효 학습률이 0으로 죽어가는 건 아닌가"를 의심하는 진단적 사고 역시 이 논문에서 출발한다.

---

## Paper 2 (Classic): Understanding Deep Learning Requires Rethinking Generalization
- **Authors:** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
- **Year:** 2016 (ICLR 2017 Best Paper)
- **arXiv:** https://arxiv.org/abs/1611.03530
- **PDF:** [./rethinking-generalization-zhang-2016.pdf](./rethinking-generalization-zhang-2016.pdf)
- **Citation Count:** ~10,000+

### 요약
"딥러닝은 왜 일반화가 잘 되는가"라는 질문에 대해, 기존의 정규화 이론으로는 설명이 불가능하다는 것을 충격적인 실험으로 보인 논문이다. 저자들은 CIFAR-10과 ImageNet의 레이블을 완전히 무작위로 섞어도 대형 신경망이 학습 데이터를 100% 암기(fit)할 수 있음을 보였다. 이는 모델의 표현력이 데이터셋 크기를 이미 초과했다는 뜻이며, VC 차원·Rademacher 복잡도 같은 전통적 복잡도 지표가 실제 일반화 성능을 전혀 예측하지 못함을 의미한다.

### 핵심 기여
- 랜덤 레이블 실험(randomization test)이라는 단순하고 강력한 진단 프로토콜을 제안
- 명시적 정규화(weight decay, dropout, data augmentation)가 일반화의 필요조건도 충분조건도 아님을 실증
- 깊이 2짜리 얕은 네트워크조차 파라미터 수가 n+d 정도면 임의의 n개 샘플을 표현할 수 있음을 구성적으로 증명
- SGD 자체가 암묵적 정규화(implicit regularization)로 작동한다는 관점을 제시 — 선형 모델에서 SGD가 최소 노름 해로 수렴함을 보임

### 이 논문이 중요한 이유
AI 엔지니어가 현장에서 가장 자주 마주치는 질문은 "왜 학습은 잘 되는데 실제 서비스에서는 성능이 떨어지는가"이다. 이 논문은 그 질문에 대한 답이 파라미터 수나 정규화 항 세팅에 있지 않다는 것을 명확히 한다. 오버피팅을 dropout 비율 몇 % 조정으로 해결하려는 습관적 접근을 근본적으로 의심하게 만들고, 대신 데이터 품질·레이블 노이즈·최적화 경로(optimization trajectory)를 보게 만든다. 특히 LLM 시대에 "모델이 벤치마크를 암기한 것인가, 일반화한 것인가"를 판별하는 논쟁의 이론적 뿌리가 바로 이 논문이다.

### 사전 지식
- 편향-분산 트레이드오프와 오버피팅/언더피팅의 고전적 정의
- VC 차원, Rademacher 복잡도 등 통계적 학습이론의 일반화 상한 개념
- weight decay(L2), dropout, data augmentation이 각각 무엇을 규제하는지
- SGD의 수렴 특성과 손실 지형(loss landscape)의 기본 직관

### 관련 논문
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (Keskar et al., 2016)](https://arxiv.org/abs/1609.04836)
- [Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2020)](https://arxiv.org/abs/2010.01412)
- [Deep Double Descent: Where Bigger Models and More Data Hurt (Nakkiran et al., 2019)](https://arxiv.org/abs/1912.02292)

### 실무 적용
이 논문의 랜덤 레이블 실험은 그대로 데이터셋 품질 검증 도구가 된다. 학습 데이터의 레이블을 셔플했는데도 train loss가 잘 떨어진다면, 그 모델은 지금 데이터의 "구조"가 아니라 "샘플"을 외우고 있을 가능성이 크다. 실무에서는 (1) 학습셋/평가셋 중복(data leakage) 탐지, (2) 레이블 노이즈가 심한 크라우드소싱 데이터의 신뢰도 측정, (3) LLM 파인튜닝 시 소량 데이터에서 암기가 시작되는 시점 판별에 활용한다. 또한 "정규화를 더 넣자"보다 "데이터를 더 정제하자"는 의사결정의 근거로 자주 인용된다.

---

## Paper 3 (Recent): Muon is Scalable for LLM Training
- **Authors:** Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, et al. (Moonshot AI)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.16982
- **PDF:** [./muon-scalable-llm-training-liu-2025.pdf](./muon-scalable-llm-training-liu-2025.pdf)
- **Citation Count:** ~300+ (2025년 발표 옵티마이저 논문 중 최상위권, 빠르게 증가 중)

### 요약
소규모 모델에서만 검증되었던 행렬 직교화(matrix orthogonalization) 기반 옵티마이저 Muon을 대규모 LLM 학습까지 확장한 기술 보고서다. 저자들은 스케일업에 필요한 두 가지 핵심 요소로 (1) weight decay 추가, (2) 파라미터별 업데이트 스케일(RMS) 조정을 지목했고, 이를 통해 하이퍼파라미터 재튜닝 없이도 Muon이 바로 동작하게 만들었다. 스케일링 법칙 실험 결과 compute-optimal 조건에서 AdamW 대비 약 2배의 연산 효율을 달성했으며, 이를 실증하기 위해 5.7T 토큰으로 학습한 3B/16B MoE 모델 Moonlight를 공개했다.

### 핵심 기여
- Muon의 대규모 학습 실패 원인을 진단하고, weight decay와 업데이트 RMS 정합이라는 최소 수정으로 해결
- AdamW 대비 약 2배 연산 효율(동일 성능 도달에 절반의 FLOPs)을 스케일링 법칙 실험으로 입증
- 메모리 최적·통신 효율적인 분산 Muon(ZeRO-1 스타일) 구현을 오픈소스로 공개
- Moonlight 3B/16B MoE 모델과 중간 체크포인트를 공개해 재현 가능한 대규모 검증 제공

### 이 논문이 중요한 이유
2014년 Adam 이후 10여 년간 "대규모 학습의 기본값은 AdamW"라는 전제가 거의 흔들리지 않았는데, 이 논문은 그 전제를 실제 프로덕션 규모에서 처음으로 설득력 있게 반박했다. 옵티마이저가 2배 효율적이라는 것은 동일 예산으로 2배 큰 모델을 학습하거나 학습 비용을 절반으로 줄일 수 있다는 뜻이며, 이는 GPU 예산이 곧 제품 경쟁력인 조직에서 가장 레버리지가 큰 의사결정 지점이다. 또한 "그래디언트를 스칼라로 스케일링"하던 사고에서 "가중치 행렬 구조 자체를 고려한 업데이트"로 관점을 옮긴다는 점에서, Adam 이후 옵티마이저 연구의 방향 전환을 상징한다.

### 사전 지식
- Adam/AdamW의 동작 원리와 decoupled weight decay의 의미
- 행렬의 직교화, SVD, Newton-Schulz 반복법의 기본 개념
- 스케일링 법칙(Kaplan/Chinchilla)과 compute-optimal 학습의 정의
- 분산 학습 기초: ZeRO, 데이터/텐서 병렬, 옵티마이저 상태 샤딩
- MoE(Mixture-of-Experts) 아키텍처의 기본 구조

### 관련 논문
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [Symbolic Discovery of Optimization Algorithms / Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675)
- [SOAP: Improving and Stabilizing Shampoo using Adam (Vyas et al., 2024)](https://arxiv.org/abs/2409.11321)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Kimi k1.5 / Moonshot AI 기술 보고서 (2025)](https://arxiv.org/abs/2501.12599)

### 실무 적용
사전학습(pre-training)을 직접 수행하는 조직이라면 옵티마이저 교체만으로 GPU 비용을 절감할 수 있는지가 곧바로 A/B 실험 대상이 된다. 공개된 분산 Muon 구현은 기존 ZeRO-1 파이프라인에 비교적 적은 변경으로 붙일 수 있도록 설계되어 있어, 소규모(1B 이하) 파일럿에서 손실 곡선과 step time을 함께 측정한 뒤 확대하는 방식이 현실적이다. 다만 파인튜닝·LoRA처럼 스텝 수가 적은 워크로드에서는 이득이 작을 수 있으므로, "우리 워크로드가 사전학습형인가 적응형인가"를 먼저 구분하는 것이 판단 기준이다. 또한 Moonlight 체크포인트는 3B/16B MoE 구조를 참고 설계로 삼으려는 팀에게 유용한 레퍼런스다.

---

## 추천 읽기 순서

1. **AdaGrad (2011)** — 먼저 읽는다. "왜 파라미터마다 학습률이 달라야 하는가"라는 문제의식과 그 최초의 답을 잡아두면, 이후 모든 옵티마이저를 변주로 읽을 수 있다. 3~4장의 regret 증명은 처음에는 결론만 확인하고 넘어가도 좋다.
2. **Muon (2025)** — 두 번째로 읽는다. AdaGrad의 "스칼라 적응"과 대비해 "행렬 구조 기반 업데이트"가 무엇이 다른지 비교하며 읽으면 14년간의 진화가 한 줄로 이어진다.
3. **Rethinking Generalization (2016)** — 마지막에 읽는다. 앞의 두 논문이 "어떻게 빠르게 학습시킬 것인가"를 다뤘다면, 이 논문은 "그렇게 학습된 모델이 왜 일반화되는지 우리는 사실 모른다"고 되묻는다. 순서를 뒤집어 읽으면 허무해지지만, 이 순서로 읽으면 다음 질문이 열린다.

## 핵심 테이크어웨이

- **최적화와 정규화는 분리된 두 축이 아니다.** SGD/AdamW/Muon이 선택하는 수렴 경로 자체가 암묵적 정규화로 작동한다. 옵티마이저 교체는 속도 문제가 아니라 일반화 특성의 변경이다.
- **적응적 학습률의 계보는 하나의 문제의식에서 나왔다.** 그래디언트 스케일이 파라미터마다 다르다는 사실. AdaGrad는 누적합으로, Adam은 지수이동평균으로, Muon은 행렬 직교화로 이 문제를 풀었다.
- **명시적 정규화는 생각보다 힘이 약하다.** dropout과 weight decay를 모두 꺼도 모델은 여전히 어느 정도 일반화되고, 모두 켜도 랜덤 레이블은 여전히 암기된다. 정규화 하이퍼파라미터 튜닝의 기대효과를 과대평가하지 말 것.
- **옵티마이저 개선은 가장 레버리지가 큰 인프라 투자 중 하나다.** 2배 연산 효율은 아키텍처 개선이나 데이터 증강으로 쉽게 얻을 수 없는 크기의 이득이다.
- **실패 사례가 설계 동기를 만든다.** AdaGrad의 학습률 소멸 → RMSProp/Adam, 대형 배치의 일반화 갭 → LARS/SAM, Muon의 스케일업 실패 → weight decay + RMS 정합. 논문을 읽을 때 "무엇이 안 되었는가"를 먼저 찾을 것.

## 다음 토픽과의 연결

다음 토픽인 **Transfer Learning and Foundation Models**는 오늘 다룬 두 축이 그대로 확장된다. 첫째, 파운데이션 모델의 사전학습은 곧 수천 GPU-일 규모의 최적화 문제이므로 Muon 같은 옵티마이저 효율이 모델 크기의 상한을 결정한다. 둘째, 전이학습의 본질은 "사전학습된 표현이 새로운 도메인으로 일반화되는가"이며, 이는 오늘 세 번째 논문이 던진 일반화 질문의 도메인 확장판이다. 특히 파인튜닝에서 catastrophic forgetting을 어떻게 막을 것인가는 정규화 문제로 환원되며, LoRA 같은 파라미터 효율적 기법도 결국 "업데이트를 어떤 부분공간으로 제한할 것인가"라는 정규화 설계로 읽을 수 있다.
