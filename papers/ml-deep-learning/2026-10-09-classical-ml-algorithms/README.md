# Daily AI Paper Recommendations

> **Date:** 2026-10-09
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Classical ML Algorithms and Foundations

> 이번 사이클에서는 Random Forest/XGBoost 등 개별 알고리즘 대신 "ML을 다루는 사고방식"과 "왜 트리 모델이 여전히 강한가"라는 질문에 집중했습니다. 이전 사이클에서 다룬 논문(RF, XGBoost, LightGBM, CatBoost, AdaBoost, Bagging, ID3, SVM, GBM, Extra-Trees, Isolation Forest, SHAP, SMOTE 등)과 겹치지 않도록 선정했습니다.

---

## Paper 1 (Classic): A Few Useful Things to Know about Machine Learning
- **Authors:** Pedro Domingos
- **Year:** 2012
- **DOI:** https://doi.org/10.1145/2347736.2347755 (Communications of the ACM, 55(10), 78–87)
- **PDF:** [./few-useful-things-ml-domingos-2012.pdf](./few-useful-things-ml-domingos-2012.pdf)
- **Citation Count:** ~5,000+

### 요약
ML 연구자와 실무자가 경험으로 체득한 12가지 교훈을 정리한 에세이형 논문입니다. "학습 = 표현(Representation) + 평가(Evaluation) + 최적화(Optimization)"라는 프레임으로 알고리즘을 분해하고, 일반화·과적합·차원의 저주·피처 엔지니어링·앙상블 등 교과서에 잘 안 나오는 "암묵지"를 명시적으로 설명합니다.

### 핵심 기여
- 모든 학습 알고리즘을 Representation / Evaluation / Optimization 세 축으로 분해하는 범용 프레임 제시
- 과적합을 Bias–Variance 관점으로 설명하고, "데이터가 많은 단순 알고리즘이 데이터가 적은 정교한 알고리즘을 이긴다"는 실무 원칙 정리
- 차원의 저주, 피처 엔지니어링의 중요성, 상관관계 ≠ 인과관계, 단일 모델보다 앙상블 등 12개 교훈 체계화

### 이 논문이 중요한 이유
AI 엔지니어가 LLM 시대에도 반복해서 만나는 질문 — "왜 검증 점수는 좋은데 프로덕션에서는 망가지지?", "모델을 바꿀까, 데이터를 더 모을까?" — 에 대한 사고의 기준점을 줍니다. 특정 알고리즘이 아니라 **판단 기준**을 배우는 논문이라 커리큘럼 첫날에 가장 잘 어울립니다.

### 사전 지식
- 지도학습의 기본 개념(학습/검증/테스트 분할, 손실 함수)
- 과적합과 정규화에 대한 직관
- 의사결정나무, 선형 모델 정도의 기초 알고리즘 이해

### 관련 논문
- [Statistical Modeling: The Two Cultures (Breiman, 2001)](https://doi.org/10.1214/ss/1009213726)
- [The Unreasonable Effectiveness of Data (Halevy, Norvig, Pereira, 2009)](https://doi.org/10.1109/MIS.2009.36)
- [Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)

### 실무 적용
- 신규 모델 실험 전 "표현/평가/최적화 중 무엇을 바꾸는 실험인가?"를 실험 설계서에 명시하면 실험 목적이 선명해집니다.
- 성능 정체 시 모델 교체보다 데이터 확보·피처 개선이 ROI가 높은지 먼저 검토하는 의사결정 기준으로 활용됩니다.
- LLM 평가 파이프라인에서도 "평가 데이터 누수", "분포 이동"을 점검하는 체크리스트의 원형이 됩니다.

---

## Paper 2 (Classic): Why do tree-based models still outperform deep learning on tabular data?
- **Authors:** Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux
- **Year:** 2022 (NeurIPS 2022 Datasets & Benchmarks)
- **arXiv:** https://arxiv.org/abs/2207.08815
- **PDF:** [./why-tree-models-outperform-deep-learning-grinsztajn-2022.pdf](./why-tree-models-outperform-deep-learning-grinsztajn-2022.pdf)
- **Citation Count:** ~1,500+

### 요약
45개의 표준화된 중간 규모(~1만 샘플) 테이블 데이터셋에서 XGBoost·Random Forest 등 트리 모델과 MLP·ResNet·FT-Transformer 등 딥러닝 모델을 대규모 하이퍼파라미터 탐색(학습기당 약 20,000 GPU 시간)으로 비교했습니다. 트리 모델이 여전히 우위이며, 그 원인을 신경망의 **귀납적 편향(inductive bias)** 차이로 실험적으로 설명합니다.

### 핵심 기여
- 테이블 데이터 벤치마크 구성 기준을 명확히 하고, 하이퍼파라미터 탐색 비용까지 고려한 공정한 비교 프로토콜 제시
- 신경망의 약점 3가지를 실험으로 규명: ① 불규칙한(irregular) 타깃 함수 학습의 어려움 ② 무의미한 피처에 대한 취약성 ③ 회전 불변성(rotation invariance)으로 인한 피처 방향 정보 손실
- 테이블 전용 신경망이 풀어야 할 설계 과제를 제시해 이후 TabM, RealMLP, TabPFN 등 연구의 출발점이 됨

### 이 논문이 중요한 이유
"딥러닝이면 다 되지 않나?"라는 가설을 데이터로 반박한 논문입니다. 실무의 상당수 데이터(결제, 이탈, CRM, 로그 집계)는 여전히 테이블이고, 모델 선택을 감이 아닌 **데이터 특성(피처 수, 노이즈 피처, 함수의 불규칙성)** 으로 판단하게 해줍니다.

### 사전 지식
- 그래디언트 부스팅/랜덤 포레스트의 기본 원리
- MLP, Transformer의 기본 구조
- 하이퍼파라미터 탐색(랜덤 서치, 베이지안 최적화) 개념

### 관련 논문
- [Revisiting Deep Learning Models for Tabular Data / FT-Transformer (Gorishniy et al., 2021)](https://arxiv.org/abs/2106.11959)
- [Tabular Data: Deep Learning is Not All You Need (Shwartz-Ziv & Armon, 2021)](https://arxiv.org/abs/2106.03253)
- [When Do Neural Nets Outperform Boosted Trees on Tabular Data? (McElfresh et al., 2023)](https://arxiv.org/abs/2305.02997)

### 실무 적용
- 이탈 예측, 리드 스코어링, 사기 탐지 같은 테이블 과제에서 **GBDT를 기본 베이스라인**으로 두는 근거가 됩니다.
- 피처가 많고 노이즈가 섞인 경우 신경망 도입 전 피처 선택을 먼저 수행해야 하는 이유를 설명해 줍니다.
- 딥러닝 모델을 써야 할 때(멀티모달 결합, 임베딩 재사용 등)와 아닐 때를 구분하는 기술 의사결정 문서의 근거 자료로 활용할 수 있습니다.

---

## Paper 3 (Recent): A Closer Look at Deep Learning Methods on Tabular Datasets
- **Authors:** Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, De-Chuan Zhan
- **Year:** 2024 (v3 revised 2025-01)
- **arXiv:** https://arxiv.org/abs/2407.00956
- **PDF:** [./closer-look-deep-tabular-talent-ye-2024.pdf](./closer-look-deep-tabular-talent-ye-2024.pdf)
- **Citation Count:** ~100+

### 요약
300개 이상의 테이블 데이터셋(TALENT 벤치마크)에서 32개의 최신 딥러닝·트리 기반 방법을 여러 기준으로 비교한 대규모 실증 연구입니다. 상위 성능 방법은 기준과 무관하게 소수의 모델에 집중되며, 데이터셋의 메타 피처(이질성 등)로 딥 테이블 모델의 학습 동학을 어느 정도 예측할 수 있음을 보였습니다. 빠른 재현 평가용 45개 코어 셋(Talent-tiny)도 함께 제공합니다.

### 핵심 기여
- 규모·피처 구성(수치/범주 혼합)·도메인·태스크 유형이 다양한 300+ 데이터셋 기반 TALENT 벤치마크와 오픈소스 툴킷 공개
- 32개 방법의 다기준 비교로 "어떤 데이터에서 어떤 모델이 이기는가"를 데이터 특성 관점에서 분석
- 데이터셋 이질성을 포착하는 메타 피처를 정의하고, 이를 이용해 학습 곡선/성능을 예측하는 접근 제시

### 이 논문이 중요한 이유
Paper 2(2022)가 "트리가 이긴다"는 결론이었다면, 이 논문은 2024년 기준 **TabR, ModernNCA, TabPFN 등 신규 딥 모델까지 포함한 업데이트된 지형도**를 보여줍니다. 즉 "여전히 트리가 이기는가?"라는 가설을 2년 뒤 더 큰 데이터로 재검증하는 논문이라, 두 논문을 함께 읽으면 연구 흐름의 변화를 직접 추적할 수 있습니다.

### 사전 지식
- Paper 2의 핵심 결론과 실험 설계
- FT-Transformer, TabR, TabPFN 등 주요 딥 테이블 모델 이름 수준의 이해
- 벤치마크 평가에서 평균 순위(average rank), 통계 검정의 의미

### 관련 논문
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data (Erickson et al., 2025)](https://arxiv.org/abs/2506.16791)
- [Revisiting Nearest Neighbor for Tabular Data / ModernNCA (Ye et al., 2024)](https://arxiv.org/abs/2407.03257)
- [TabR: Tabular Deep Learning Meets Nearest Neighbors (Gorishniy et al., 2023)](https://arxiv.org/abs/2307.14338)

### 실무 적용
- 신규 테이블 과제에서 모델 후보군을 좁힐 때, 데이터 규모·피처 구성 기반으로 "먼저 시도할 3개 모델"을 정하는 근거로 활용할 수 있습니다.
- TALENT 툴킷을 사내 AutoML/모델 비교 파이프라인의 베이스라인 세트로 재사용할 수 있습니다.
- 메타 피처 기반 성능 예측 아이디어는 "데이터만 보고 모델 추천"하는 내부 ML 플랫폼 기능 설계에 응용할 수 있습니다.

---

## 추천 읽기 순서
1. **Domingos (2012)** — ML 판단의 기준(일반화, Bias–Variance, 데이터 vs 알고리즘)을 먼저 장착합니다.
2. **Grinsztajn et al. (2022)** — 그 기준으로 "트리 vs 딥러닝"이라는 구체적 질문을 실험적으로 따라갑니다.
3. **Ye et al. (2024)** — 같은 질문을 2024년 최신 모델과 300+ 데이터셋으로 재검증한 결과를 확인합니다.

## 핵심 테이크어웨이
- **질문:** 테이블 데이터에서 딥러닝이 트리를 이길 수 있는가? → **답:** 일반적으로는 아직 GBDT가 강력한 기본값이지만, 데이터 특성(규모, 이질성, 노이즈 피처)에 따라 승자가 달라지며 그 격차는 좁혀지는 중입니다.
- **질문:** 성능이 정체되면 무엇부터 바꿔야 하나? → **답:** 알고리즘보다 데이터·피처·평가 방식을 먼저 의심하라(Domingos).
- **질문:** 모델 선택을 어떻게 데이터 기반으로 할 수 있나? → **답:** 벤치마크 결과를 데이터셋 메타 피처와 연결해 보면 "우리 데이터에 맞는 모델"을 가설로 세울 수 있습니다.

## 다음 토픽과의 연결
다음 토픽은 **Neural Network Fundamentals and Training(Batch Normalization, Dropout 등)** 입니다. 오늘 Paper 2에서 지적한 신경망의 약점(불규칙 함수 학습, 노이즈 피처 취약성)이 정규화·학습 기법으로 어디까지 보완될 수 있는지를 질문으로 가지고 넘어가면 좋습니다.
