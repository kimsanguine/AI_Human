# Daily AI Paper Recommendations

> **Date:** 2026-11-05
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Classical ML Algorithms and Foundations

---

## Paper 1 (Classic): Scikit-learn: Machine Learning in Python
- **Authors:** Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, et al.
- **Year:** 2011 (JMLR 12)
- **arXiv:** https://arxiv.org/abs/1201.0490
- **PDF:** [./scikit-learn-pedregosa-2011.pdf](./scikit-learn-pedregosa-2011.pdf)
- **Citation Count:** ~100,000+

### 요약
고전 ML 알고리즘(SVM, 랜덤 포레스트, k-means, PCA 등)을 하나의 일관된 Python API로 묶은 scikit-learn 라이브러리를 소개합니다. 6쪽짜리 짧은 논문이지만, "알고리즘을 누구나 쓸 수 있게 만드는 설계"가 무엇인지 보여 줍니다.

### 핵심 기여
- `fit / predict / transform`으로 통일된 Estimator 인터페이스 설계
- NumPy·SciPy 기반의 효율적 구현과 BSD 라이선스로 산업·학계 동시 확산
- 문서화·테스트·API 일관성을 강조한 오픈소스 개발 프로세스 정립

### 이 논문이 중요한 이유
고전 ML의 "사실상 표준 도구"입니다. 알고리즘 자체보다 **인터페이스 설계가 생태계를 만든다**는 점에서, 나중에 LLM·에이전트 프레임워크(LangChain의 Runnable 등)를 설계할 때도 같은 원리가 적용됩니다.

### 사전 지식
Python, NumPy 배열, 지도/비지도 학습의 기본 개념

### 관련 논문
- [API design for machine learning software: experiences from the scikit-learn project (Buitinck et al., 2013)](https://arxiv.org/abs/1309.0238)
- [Random Forests (Breiman, 2001)](https://doi.org/10.1023/A:1010933404324)

### 실무 적용
베이스라인 모델, 피처 전처리 파이프라인(`Pipeline`, `ColumnTransformer`), 교차 검증, 하이퍼파라미터 탐색에 지금도 매일 쓰입니다. LLM 프로젝트에서도 임베딩 + 로지스틱 회귀 같은 가벼운 분류기, 평가 지표 계산에 자주 등장합니다.

---

## Paper 2 (Classic): "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
- **Authors:** Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
- **Year:** 2016 (KDD 2016)
- **arXiv:** https://arxiv.org/abs/1602.04938
- **PDF:** [./lime-why-should-i-trust-you-ribeiro-2016.pdf](./lime-why-should-i-trust-you-ribeiro-2016.pdf)
- **Citation Count:** ~20,000+

### 요약
어떤 블랙박스 분류기든, 한 개의 예측 주변에서 입력을 조금씩 바꿔 보고 그 결과를 단순한 선형 모델로 근사해 "왜 이런 예측이 나왔는지" 설명하는 LIME 기법을 제안합니다. 여러 설명을 골라 모델 전체를 신뢰할 수 있는지 판단하는 SP-LIME도 함께 제시합니다.

### 핵심 기여
- 모델 종류와 무관한(model-agnostic) 국소 설명 프레임워크
- 해석 가능성과 충실도(fidelity)의 트레이드오프를 하나의 목적식으로 정식화
- 사용자 실험으로 "설명이 사람의 모델 신뢰·선택을 실제로 개선한다"는 것을 검증 (허스키 vs 늑대 사례)

### 이 논문이 중요한 이유
정확도만으로는 모델을 배포할 수 없다는 사실을 처음 대중화한 논문입니다. 이전 사이클에서 다룬 SHAP의 직접적 선행 연구이며, 오늘날 LLM의 근거 제시·attribution 연구의 출발점이기도 합니다.

### 사전 지식
선형 회귀, 분류기의 예측 확률, 피처 개념, 기본적인 샘플링

### 관련 논문
- [A Unified Approach to Interpreting Model Predictions / SHAP (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
- [Anchors: High-Precision Model-Agnostic Explanations (Ribeiro et al., 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11491)
- [Towards A Rigorous Science of Interpretable Machine Learning (Doshi-Velez & Kim, 2017)](https://arxiv.org/abs/1702.08608)

### 실무 적용
신용평가·이탈 예측 같은 B2B 모델에서 "왜 이 고객이 위험군인가"를 CS·영업팀에 설명할 때, 그리고 학습 데이터 누수(leakage)나 잘못된 상관(spurious correlation)을 찾는 디버깅에 사용됩니다.

---

## Paper 3 (Recent): TabDPT: Scaling Tabular Foundation Models on Real Data
- **Authors:** Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, Maksims Volkovs (Layer 6 AI)
- **Year:** 2024 (NeurIPS 2025)
- **arXiv:** https://arxiv.org/abs/2410.18164
- **PDF:** [./tabdpt-scaling-tabular-foundation-models-ma-2024.pdf](./tabdpt-scaling-tabular-foundation-models-ma-2024.pdf)
- **Citation Count:** ~50+ (빠르게 증가 중)

### 요약
합성 데이터로만 학습한 TabPFN 계열과 달리, **실제 표 데이터**에 retrieval 기반 in-context learning과 자기지도 학습을 결합해 사전학습한 표 데이터 파운데이션 모델입니다. 모델 크기와 데이터 양이 늘수록 성능이 멱법칙(power law)으로 좋아진다는 스케일링 결과를 보여 줍니다.

### 핵심 기여
- 실제 표 데이터 기반 사전학습 + 컨텍스트 retrieval로 큰 데이터셋에도 적용
- 표 데이터에서도 LLM처럼 스케일링 법칙이 성립함을 실증
- 추가 학습(파인튜닝) 없이 분류·회귀 모두 처리, 가중치·코드·학습 데이터 공개

### 이 논문이 중요한 이유
"표 데이터는 여전히 XGBoost가 왕"이라는 통념(10월 사이클의 Grinsztajn 논문)에 대한 최신 반론 흐름의 핵심입니다. 고전 ML과 파운데이션 모델 패러다임이 만나는 지점을 이해할 수 있습니다.

### 사전 지식
Transformer와 in-context learning, k-NN retrieval 개념, TabPFN의 기본 아이디어

### 관련 논문
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second (Hollmann et al., 2022)](https://arxiv.org/abs/2207.01848)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data (Erickson et al., 2025)](https://arxiv.org/abs/2506.16791)
- [Accurate predictions on small data with a tabular foundation model (Hollmann et al., 2025)](https://doi.org/10.1038/s41586-024-08328-6)

### 실무 적용
SaaS 제품의 이탈·전환 예측처럼 데이터셋은 많지만 각각 작고, 매번 튜닝할 인력이 부족한 상황에서 "학습 없이 바로 예측하는" 베이스라인으로 유용합니다. 단, 추론 비용·지연 시간은 GBDT보다 크므로 배치 예측에 먼저 적용해 보는 것을 권합니다.

---

## 추천 읽기 순서
1. **Scikit-learn (2011)** — 고전 ML 도구 생태계와 Estimator 추상화를 먼저 이해
2. **LIME (2016)** — 만든 모델을 "어떻게 믿을 것인가"로 관점 확장
3. **TabDPT (2024)** — 같은 표 데이터 문제를 파운데이션 모델로 푸는 최신 흐름 확인

## 핵심 테이크어웨이
- 좋은 알고리즘보다 **좋은 인터페이스**가 더 오래 살아남는다 (scikit-learn).
- 정확도는 배포의 필요조건일 뿐, **설명 가능성**이 신뢰를 만든다 (LIME).
- 표 데이터에서도 스케일링 법칙이 나타나기 시작했다 — 다만 GBDT와의 비용·성능 트레이드오프는 직접 벤치마크로 확인해야 한다 (TabDPT).

## 다음 토픽과의 연결
다음 주제는 **Neural Network Fundamentals and Training**입니다. TabDPT 같은 모델이 안정적으로 학습되려면 정규화(Normalization), 드롭아웃, 최적화 기법이 필수입니다. 오늘의 "고전 ML → 파운데이션 모델" 흐름이 내일 다룰 신경망 학습 기법 위에서 어떻게 가능해지는지 이어서 살펴보세요.
