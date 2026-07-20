# Daily AI Paper Recommendations

> **Date:** 2026-07-20
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Classical ML Algorithms and Foundations

---

## Paper 1 (Classic): Greedy Function Approximation: A Gradient Boosting Machine
- **Authors:** Jerome H. Friedman
- **Year:** 2001
- **Journal:** The Annals of Statistics, 29(5), 1189-1232 (DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451))
- **PDF:** [./greedy-function-approximation-friedman-2001.pdf](./greedy-function-approximation-friedman-2001.pdf)
- **Citation Count:** 30,000+ (approximate)

### 요약
함수 추정 문제를 "함수 공간에서의 수치 최적화"로 재해석한 논문이다. 단계적(stagewise) 가법 모형의 확장이 곧 최급강하법(steepest descent)과 동치임을 보이고, 임의의 미분 가능한 손실 함수에 대해 동작하는 일반적 gradient boosting 프레임워크를 제시한다. 여기에 회귀 트리를 base learner로 결합한 GBM(Gradient Boosting Machine)을 정의하고, shrinkage와 subsampling 같은 실무 필수 기법까지 함께 정립했다.

### 핵심 기여
- Boosting을 "손실 함수의 gradient를 잔차로 삼아 순차적으로 base learner를 적합하는 함수 공간 경사하강법"으로 일반화
- 회귀·분류·robust 손실(LAD, Huber, deviance) 전반에 적용 가능한 통합 알고리즘 유도
- Shrinkage(learning rate), 트리 깊이를 통한 상호작용 차수 제어, relative influence 기반 변수 중요도, partial dependence plot 제안

### 이 논문이 중요한 이유
오늘날 XGBoost, LightGBM, CatBoost가 정형 데이터에서 여전히 최강자인데, 이 세 라이브러리는 모두 이 논문의 직계 후손이다. GBM의 하이퍼파라미터(learning rate, n_estimators, max_depth, subsample)가 왜 그렇게 설계되었는지를 원리 수준에서 이해해야 튜닝이 감이 아니라 논리가 된다. 또한 partial dependence plot은 오늘날 모델 해석(XAI)의 출발점으로, 프로덕트에서 "왜 이 예측이 나왔는가"를 설명해야 할 때 반드시 필요한 도구다.

### 사전 지식
- 경사하강법과 손실 함수의 기본 개념
- 의사결정 트리(CART)의 분할 기준과 구조
- 편향-분산 트레이드오프, 앙상블(bagging vs boosting)의 차이

### 관련 논문
- [Bagging Predictors (Breiman, 1996)](https://doi.org/10.1007/BF00058655)
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., 2017)](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

### 실무 적용
이탈 예측, LTV 예측, 리드 스코어링, 광고 CTR 예측처럼 정형 피처가 중심인 그로스 문제에서 GBM 계열은 여전히 1순위 베이스라인이다. Learning rate를 낮추고 트리 수를 늘리는 전형적 튜닝 전략, 그리고 relative influence로 "어떤 행동 지표가 리텐션을 설명하는가"를 뽑아 제품 가설로 전환하는 워크플로가 모두 이 논문에서 나왔다.

---

## Paper 2 (Classic): Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?
- **Authors:** Manuel Fernández-Delgado, Eva Cernadas, Senén Barro, Dinani Amorim
- **Year:** 2014
- **Journal:** Journal of Machine Learning Research, 15, 3133-3181 ([JMLR link](https://jmlr.org/papers/v15/delgado14a.html))
- **PDF:** [./do-we-need-hundreds-of-classifiers-fernandez-delgado-2014.pdf](./do-we-need-hundreds-of-classifiers-fernandez-delgado-2014.pdf)
- **Citation Count:** 3,500+ (approximate)

### 요약
179개의 분류 알고리즘(17개 계열, R·Weka·Matlab·C 구현 포함)을 UCI 121개 데이터셋에 전부 돌려 비교한 대규모 실증 연구다. 결론은 명확했다: Random Forest 계열이 평균적으로 가장 강했고, 그다음이 Gaussian 커널 SVM이었다. 알고리즘 선택 논쟁을 "취향"이 아니라 "벤치마크 증거"의 영역으로 옮긴 논문이다.

### 핵심 기여
- 알고리즘 비교의 표준적인 대규모 벤치마크 프로토콜 제시(동일 데이터셋, 동일 튜닝 예산, 순위 기반 집계)
- Random Forest > SVM(RBF) > Boosting > Neural Net 순의 실증적 강도 순위를 정형 데이터에서 확립
- "새 알고리즘이 좋다"는 주장에는 다수 데이터셋에 걸친 검증이 필요하다는 규범을 정착

### 이 논문이 중요한 이유
AI 엔지니어가 흔히 저지르는 실수는 최신 모델부터 시작하는 것이다. 이 논문은 강한 베이스라인이 얼마나 이기기 어려운지를 숫자로 보여준다. 또한 "우리 데이터에서 하나 돌려보고 좋았다"는 식의 결론이 왜 위험한지, 데이터셋 수와 순위 통계가 왜 필요한지를 알려준다. 이는 모델 선택뿐 아니라 A/B 테스트 해석 습관에도 그대로 이어지는 사고방식이다.

### 사전 지식
- 교차검증과 하이퍼파라미터 튜닝 프로토콜
- Random Forest, SVM, kNN 등 주요 분류기의 기본 동작
- 다중 데이터셋 비교 시 쓰이는 평균 순위·Friedman 검정 개념

### 관련 논문
- [Random Forests (Breiman, 2001)](https://doi.org/10.1023/A:1010933404324)
- [Statistical Comparisons of Classifiers over Multiple Data Sets (Demšar, 2006)](https://jmlr.org/papers/v7/demsar06a.html)
- [Why do tree-based models still outperform deep learning on typical tabular data? (Grinsztajn et al., 2022)](https://arxiv.org/abs/2207.08815)

### 실무 적용
새 ML 기능을 스펙할 때 "우선 Random Forest / GBM 베이스라인을 세우고, 그보다 유의미하게 나은 경우에만 복잡한 모델로 간다"는 의사결정 규칙의 근거가 된다. 모델 실험 리포트를 만들 때 단일 지표가 아니라 여러 데이터 슬라이스에 걸친 순위를 함께 제시하는 관행도 여기서 출발한다.

---

## Paper 3 (Recent): Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data
- **Authors:** David Holzmüller, Léo Grinsztajn, Ingo Steinwart
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** [https://arxiv.org/abs/2407.04491](https://arxiv.org/abs/2407.04491)
- **PDF:** [./realmlp-better-by-default-holzmuller-2024.pdf](./realmlp-better-by-default-holzmuller-2024.pdf)
- **Citation Count:** 100+ (approximate, 2026년 기준 증가 중)

### 요약
정형 데이터에서 GBDT가 신경망을 계속 이겨온 이유 중 상당 부분이 "튜닝 예산의 비대칭"이라는 문제의식에서 출발한다. 저자들은 개선된 MLP인 RealMLP와, GBDT·RealMLP 양쪽에 대한 메타 튜닝된 강력한 기본값(default)을 제안한다. 118개 데이터셋으로 구성된 meta-train 벤치마크에서 기본값을 학습한 뒤, 겹치지 않는 90개 데이터셋 meta-test 벤치마크에서 하이퍼파라미터 최적화 버전과 비교했다.

### 핵심 기여
- RealMLP: 정규화·초기화·학습률 스케줄·수치 임베딩 등을 개선해 정형 데이터에 특화한 MLP 설계
- 메타 학습된 기본 하이퍼파라미터 세트 — 튜닝 없이도 HPO에 근접한 성능을 내는 "좋은 출발점" 제공
- RealMLP + GBDT 앙상블이 튜닝된 단일 모델 대비 시간 대비 성능(time-accuracy tradeoff)에서 우수함을 실증

### 이 논문이 중요한 이유
"정형 데이터엔 무조건 GBDT"라는 통념을 조건부로 바꿔놓는다. 핵심 통찰은 모델 아키텍처보다 **기본값의 품질과 튜닝 예산**이 비교 결과를 좌우한다는 점이다. AI 엔지니어에게 이는 벤치마크를 읽는 법 자체를 바꾸는 교훈이고, 제한된 컴퓨트·시간 안에서 어떤 모델을 고를지 결정하는 실질적 기준을 준다.

### 사전 지식
- MLP의 정규화(BatchNorm/LayerNorm), 학습률 스케줄, 임베딩 기법
- GBDT의 주요 하이퍼파라미터와 HPO(random search, Optuna 등)
- 메타 학습(meta-learning)과 meta-train/meta-test 분리 개념

### 관련 논문
- [Why do tree-based models still outperform deep learning on typical tabular data? (Grinsztajn et al., 2022)](https://arxiv.org/abs/2207.08815)
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second (Hollmann et al., 2022)](https://arxiv.org/abs/2207.01848)
- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (Gorishniy et al., 2024)](https://arxiv.org/abs/2410.24210)

### 실무 적용
컴퓨트 예산이 빠듯한 팀에서 "튜닝 없이 좋은 기본값"은 곧 배포 속도다. RealMLP의 사전 튜닝 기본값을 첫 실험에 쓰면 HPO 파이프라인을 세우기 전에 성능 상한을 빠르게 가늠할 수 있고, GBDT와의 앙상블은 추가 튜닝 없이 안정적 이득을 준다. 사내 ML 플랫폼의 AutoML 기본 설정을 정의할 때 참고할 만한 레퍼런스다.

---

## 추천 읽기 순서
1. **Friedman (2001)** — 먼저 gradient boosting의 원리를 이해한다. 오늘날 정형 데이터 모델링의 뿌리이며, 나머지 두 논문을 읽는 언어를 제공한다. (수식이 부담되면 1~4장과 partial dependence 부분 위주로)
2. **Fernández-Delgado et al. (2014)** — 그 원리가 실제 세계 데이터에서 어떻게 검증되는지, 그리고 알고리즘 비교를 어떻게 해야 하는지를 배운다.
3. **Holzmüller et al. (2024)** — 2024년 시점에서 그 결론이 어떻게 갱신되었는지, 그리고 "공정한 비교"의 조건이 무엇인지 확인한다.

## 핵심 테이크어웨이
- Boosting은 마법이 아니라 **함수 공간에서의 경사하강법**이다. 이 관점을 가지면 하이퍼파라미터 튜닝이 원리적으로 설명된다.
- 강한 베이스라인(RF, GBDT)은 놀랄 만큼 이기기 어렵다. 새 모델의 우위는 **다수 데이터셋 + 동일 튜닝 예산**에서 검증되어야 한다.
- 2024년 이후, 신경망도 좋은 기본값을 갖추면 정형 데이터에서 경쟁력이 있다. 승부처는 아키텍처가 아니라 **기본값·튜닝 예산·앙상블**로 이동했다.
- 세 논문을 관통하는 메타 교훈: **평가 프로토콜이 결론을 만든다.** 제품 실험에서도 동일하다.

## 다음 토픽과의 연결
다음 토픽은 **신경망 기초와 학습 기법(Neural Network Fundamentals and Training)** 이다. 오늘 RealMLP에서 확인했듯 신경망의 성능은 아키텍처보다 정규화·초기화·학습률 스케줄 같은 "학습 레시피"에 크게 좌우된다. 다음 편에서 다룰 Batch Normalization과 Dropout은 바로 그 레시피의 핵심 부품으로, 오늘 배운 "기본값이 성능을 만든다"는 관점을 신경망 내부에서 다시 확인하게 된다.
