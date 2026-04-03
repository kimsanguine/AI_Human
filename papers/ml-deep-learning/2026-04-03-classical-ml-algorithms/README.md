# Daily AI Paper Recommendations

> **Date:** 2026-04-03
> **Module:** Machine Learning and Deep Learning
> **Topic:** Classical ML Algorithms and Foundations

---

## Paper 1 (Classic): Random Forests
- **Authors:** Leo Breiman
- **Year:** 2001
- **Source:** Machine Learning, 45(1), 5–32
- **PDF:** [./random-forests-breiman-2001.pdf](./random-forests-breiman-2001.pdf)
- **Citation Count:** ~120,000+

### 요약
Random Forests는 학습 시 다수의 결정 트리를 구성하고, 분류에서는 클래스의 최빈값을, 회귀에서는 개별 트리 예측의 평균을 출력하는 앙상블 학습 방법을 소개한다. 각 분할마다 랜덤 피처 선택과 배깅(bagging)을 결합하여 편향(bias)을 유지하면서 분산(variance)을 줄이는 핵심 아이디어를 제시했다.

### 핵심 기여
- 배깅과 랜덤 피처 부분공간 선택을 결합한 Random Forest 알고리즘 제안
- 트리 수가 증가함에 따라 일반화 오류가 수렴함을 증명하여 과적합 방지 특성 입증
- 순열 기반(permutation-based) 변수 중요도 측정 개념 정의 — 모델 해석력의 기초

### 이 논문이 중요한 이유
Random Forests는 단순함, 견고함, 뛰어난 기본 성능 덕분에 산업계에서 가장 널리 사용되는 ML 알고리즘 중 하나로 남아있다. 앙상블 방법의 표준을 세웠으며, 이후 등장한 거의 모든 트리 기반 알고리즘에 영향을 미쳤다.

### 사전 지식
- 결정 트리(CART)에 대한 기본 이해
- 배깅(Bootstrap Aggregating) 개념
- 편향-분산 트레이드오프(Bias-Variance Tradeoff)

### 관련 논문
- [Classification and Regression Trees / CART (Breiman et al., 1984)](https://doi.org/10.1201/9781315139470)
- [Bagging Predictors (Breiman, 1996)](https://link.springer.com/article/10.1023/A:1018054314350)
- [An Empirical Comparison of Supervised Learning Algorithms (Caruana & Niculescu-Mizil, 2006)](https://dl.acm.org/doi/10.1145/1143844.1143865)

### 실무 적용
Random Forests는 정형 데이터 기반 프로덕션 시스템에서 광범위하게 활용된다 — 이상 거래 탐지, 의료 진단, 추천 시스템, 피처 선택 파이프라인 등. 내장된 변수 중요도(feature importance)는 규제 산업에서 모델 해석력 확보에 매우 유용하다.

---

## Paper 2 (Classic): XGBoost — A Scalable Tree Boosting System
- **Authors:** Tianqi Chen, Carlos Guestrin
- **Year:** 2016
- **arXiv:** [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
- **PDF:** [./xgboost-chen-guestrin-2016.pdf](./xgboost-chen-guestrin-2016.pdf)
- **Citation Count:** ~50,000+

### 요약
XGBoost는 구조화/정형 데이터 대회와 산업 응용을 지배한 최적화된 분산 그래디언트 부스팅 시스템이다. 결측값 처리를 위한 희소성 인식(sparsity-aware) 알고리즘, 근사적 트리 학습을 위한 가중 분위수 스케치(weighted quantile sketch), 효율적 연산을 위한 캐시 인식 블록 구조를 도입했다.

### 핵심 기여
- 희소성 인식 분할 탐색을 갖춘 확장 가능한 엔드투엔드 트리 부스팅 시스템 설계
- 효율적인 근사적 분할 열거를 위한 가중 분위수 스케치 도입
- 캐시 인식 접근 패턴, 데이터 압축, 샤딩을 통한 아웃오브코어 연산 엔지니어링
- L1/L2 페널티를 결합한 정규화 학습 목적함수로 과적합 방지

### 이 논문이 중요한 이유
XGBoost는 Kaggle 대회를 석권하며 산업계 정형 데이터 ML의 사실상 표준이 되었다. 정규화, 희소성 처리, 시스템 최적화 등 설계 결정을 이해하면 프로덕션 ML에 필수적인 알고리즘과 엔지니어링 원칙을 동시에 배울 수 있다.

### 사전 지식
- 그래디언트 부스팅 개념 (Friedman, 2001)
- 결정 트리 기초
- 기본 최적화 (경사 하강법, 2차 방법)

### 관련 논문
- [Greedy Function Approximation: A Gradient Boosting Machine (Friedman, 2001)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., 2017)](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [CatBoost: Unbiased Boosting with Categorical Features (Prokhorenkova et al., 2018)](https://arxiv.org/abs/1706.09516)

### 실무 적용
XGBoost는 핀테크(신용 평가), 광고 기술(클릭률 예측), 헬스케어(위험 계층화) 등 정형 데이터가 존재하는 거의 모든 도메인에서 실시간 예측 시스템을 구동한다. 효율적인 추론 성능은 지연 시간에 민감한 프로덕션 환경에 이상적이다.

---

## Paper 3 (Recent): A Survey on Deep Tabular Learning
- **Authors:** Shriyank Somvanshi, Subasish Das, Syed Aaqib Javed, Gian Antariksa, Ahmed Hossain
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2410.12034](https://arxiv.org/abs/2410.12034)
- **PDF:** [./survey-deep-tabular-learning-somvanshi-2024.pdf](./survey-deep-tabular-learning-somvanshi-2024.pdf)
- **Citation Count:** ~50+ (빠르게 증가 중)

### 요약
이 종합 서베이는 초기 완전연결 네트워크(FCN)부터 TabNet, SAINT, TabTranSELU, MambaNet 등 최신 아키텍처까지 정형 데이터용 딥러닝 모델의 진화를 리뷰한다. 딥러닝이 트리 기반 모델(Random Forests, XGBoost)을 정형 데이터에서 마침내 능가할 수 있는지에 대한 진행 중인 논쟁을 다루며, 어텐션 메커니즘, 피처 임베딩, 하이브리드 아키텍처, 자기지도 사전학습, TabPFNv2 같은 파운데이션 모델까지 포괄한다.

### 핵심 기여
- FCN 기반, 어텐션 기반, 트리 영감, 하이브리드 아키텍처로 정형 딥러닝을 체계적 분류
- 정형 데이터를 위한 자기지도 학습 및 사전학습 전략의 역할 분석
- 소규모 데이터에서 가능성을 보이는 파운데이션 모델(TabPFNv2, TabICL) 다룸
- 데이터셋 특성에 따른 실용적인 모델 선택 가이드 제공

### 클래식 논문 대비 새로운 점
Random Forests와 XGBoost가 정형 데이터의 골드 스탠다드를 정의한 반면, 이 서베이는 딥러닝이 그 격차를 좁히고 있음을 보여준다 — 특히 어텐션 기반 모델과 파운데이션 모델에서. 그러나 트리 기반 방법은 여전히 많은 벤치마크, 특히 이질적 피처(heterogeneous features)에서 우세하다. 이 서베이는 실무자들이 언제 어떤 접근법을 사용할지 판단하는 데 도움을 준다.

### 이 논문이 중요한 이유
아키텍처 결정을 내리는 AI 엔지니어에게 클래식 트리 기반 방법 대비 딥러닝의 현재 위치를 이해하는 것은 필수적이다. 이 서베이는 딥러닝이 진정으로 우위를 보이는 경우와 그렇지 않은 경우를 포함한 가장 최신의 정형 ML 지형도를 제공한다.

### 사전 지식
- Random Forests와 그래디언트 부스팅에 대한 이해 (Paper 1 & 2)
- 어텐션 메커니즘과 트랜스포머에 대한 기본 이해
- 신경망 학습 기초

### 관련 논문
- [Why do tree-based models still outperform deep learning on tabular data? (Grinsztajn et al., 2022)](https://arxiv.org/abs/2207.08815)
- [TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, 2021)](https://arxiv.org/abs/1908.07442)
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second (Hollmann et al., 2023)](https://arxiv.org/abs/2207.01848)

### 실무 적용
이 서베이는 헬스케어, 금융, 운송 등 정형 데이터가 지배하는 도메인에서 프로덕션 정형 데이터 시스템을 위해 클래식 ML과 딥러닝 중 무엇을 선택할지 결정하는 AI 엔지니어에게 필수적인 읽을거리다.

---

## 추천 읽기 순서
1. **Random Forests** 부터 — 앙상블 접근법의 기초를 세우고 베이스라인을 확립
2. 다음으로 **XGBoost** — 부스팅으로 트리 앙상블을 발전시키며, 엔지니어링 최적화가 어떻게 실질적 임팩트를 만드는지 보여줌
3. 마지막으로 **A Survey on Deep Tabular Learning** — 전체 현대 지형도를 파악하고 이 분야가 어디로 향하는지 확인

## 핵심 테이크어웨이
- 트리 기반 앙상블 방법(Random Forest, XGBoost)은 이질적 피처에 대한 견고성, 해석력, 강한 귀납적 편향 덕분에 정형 데이터의 골드 스탠다드로 남아있다
- XGBoost의 성공은 알고리즘 혁신(정규화 부스팅, 희소성 처리)과 시스템 엔지니어링(캐시 인식 연산, 분산 처리)의 결합에서 나온다
- 정형 데이터용 딥러닝은 어텐션 기반 및 파운데이션 모델 접근법으로 빠르게 발전 중이나, 트리 기반 방법을 보편적으로 능가하지는 못했다 — 언제 무엇을 사용할지 아는 것이 핵심

## 다음 토픽과의 연결
내일의 토픽은 **Neural Network Fundamentals and Training** (Batch Normalization, Dropout)이다. 클래식 ML 알고리즘이 왜 정형 데이터에서 여전히 경쟁력이 있는지 이해하면 중요한 맥락을 갖게 된다 — 신경망은 트리 기반 방법이 구조적으로 자연스럽게 해결하는 문제들을 다루기 위해 특정 아키텍처 혁신(정규화, 정규화 기법)이 필요하기 때문이다.
