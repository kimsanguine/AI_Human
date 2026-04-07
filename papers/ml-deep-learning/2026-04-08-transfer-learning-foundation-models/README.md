# Daily AI Paper Recommendations

> **Date:** 2026-04-08
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): A Survey on Transfer Learning
- **Authors:** Sinno Jialin Pan, Qiang Yang
- **Year:** 2010
- **URL:** https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf
- **PDF:** [./survey-transfer-learning-pan-2010.pdf](./survey-transfer-learning-pan-2010.pdf)
- **Citation Count:** 25,000+

### 요약
전이학습(Transfer Learning)의 정의, 분류, 그리고 주요 접근 방식을 종합적으로 정리한 가장 영향력 있는 서베이 논문이다. 소스 도메인의 지식을 타깃 도메인으로 어떻게 옮길 수 있는지에 대한 이론적 토대와 실용적 분류 체계를 제시한다.

### 핵심 기여
- Inductive / Transductive / Unsupervised Transfer Learning의 분류 체계 정립
- "What to transfer", "How to transfer", "When to transfer"라는 3가지 핵심 질문 제안
- Instance / Feature / Parameter / Relational knowledge transfer의 4가지 전이 방식 정의

### 이 논문이 중요한 이유
오늘날 모든 파인튜닝, 도메인 적응, 사전학습-전이 패러다임의 이론적 출발점이다. AI 엔지니어가 사전학습 모델을 다룰 때 왜, 무엇을, 어떻게 전이해야 하는지에 대한 사고의 프레임을 제공한다.

### 사전 지식
지도학습/비지도학습 기본 개념, 도메인과 태스크의 정의, 일반화(generalization)와 분포 이동(distribution shift)에 대한 이해.

### 관련 논문
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)
- [A Survey of Transfer Learning (Weiss et al., 2016)](https://doi.org/10.1186/s40537-016-0043-6)

### 실무 적용
파인튜닝 전략 설계, 도메인 적응이 필요한 B2B AI 제품(예: 의료/법률 NLP), LoRA/Adapter 기반 효율적 전이 학습 등 거의 모든 모던 LLM 활용 시나리오의 개념적 기초가 된다.

---

## Paper 2 (Classic): ImageNet: A Large-Scale Hierarchical Image Database
- **Authors:** Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei
- **Year:** 2009
- **URL:** https://image-net.org/static_files/papers/imagenet_cvpr09.pdf
- **PDF:** [./imagenet-deng-2009.pdf](./imagenet-deng-2009.pdf)
- **Citation Count:** 60,000+

### 요약
WordNet 계층 구조에 기반해 수천 개 클래스, 수백만 장의 라벨링된 이미지를 모은 대규모 데이터셋 ImageNet을 제안한 논문. 이후 딥러닝 혁명과 사전학습-전이 패러다임의 데이터적 토대가 되었다.

### 핵심 기여
- 22K개 클래스, 수백만 장 이미지로 구성된 사전 없이 보지 못한 규모의 이미지 데이터셋 구축
- WordNet 기반 의미적 계층 구조 도입
- 데이터 수집/검증을 위한 크라우드소싱(Amazon Mechanical Turk) 파이프라인 제시

### 이 논문이 중요한 이유
"데이터가 모델보다 중요하다"는 명제를 실증한 시초. ImageNet 사전학습 → 다운스트림 파인튜닝이라는 컴퓨터 비전의 표준 워크플로우를 만들어냈고, 오늘날 파운데이션 모델 시대의 데이터 중심 사고의 출발점이다.

### 사전 지식
이미지 분류 태스크, 데이터셋 큐레이션의 개념, 라벨 노이즈와 데이터 품질 이슈에 대한 기초적 이해.

### 관련 논문
- [ImageNet Classification with Deep CNNs / AlexNet (Krizhevsky et al., 2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Revisiting Unreasonable Effectiveness of Data in Deep Learning Era (Sun et al., 2017)](https://arxiv.org/abs/1707.02968)

### 실무 적용
AI 제품에서 사전학습 백본 선택, 데이터셋 설계, 라벨링 파이프라인 구축 시 사고의 원형이 된다. 특히 도메인 특화 파운데이션 모델을 만들 때 "ImageNet급 데이터를 어떻게 만들 것인가"가 핵심 질문이 된다.

---

## Paper 3 (Recent): A Survey of Resource-efficient LLM and Multimodal Foundation Models
- **Authors:** Mengwei Xu, Wangsong Yin, Dongqi Cai, et al.
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2401.08092
- **PDF:** [./resource-efficient-foundation-models-xu-2024.pdf](./resource-efficient-foundation-models-xu-2024.pdf)
- **Citation Count:** 300+

### 요약
LLM, ViT, 디퓨전 모델, 멀티모달 LLM 등 대규모 파운데이션 모델을 자원 효율적으로 학습/배포/추론하기 위한 최신 기법들을 알고리즘과 시스템 양 측면에서 종합한 서베이.

### 핵심 기여
- 파운데이션 모델 라이프사이클 전체(학습, 파인튜닝, 추론)에 걸친 효율화 기법의 분류 체계
- 알고리즘 레벨(양자화, 프루닝, 스파스 어텐션, MoE)과 시스템 레벨(분산 학습, 서빙 최적화) 통합 정리
- 멀티모달/온디바이스 파운데이션 모델까지 범위 확장

### 이 논문이 중요한 이유
파운데이션 모델 시대에 "어떻게 더 적은 자원으로 더 큰 모델을 굴릴 것인가"는 모든 AI 제품의 비용/지연/규모 문제와 직결된다. 이 서베이 한 편으로 효율화 기법의 지형도를 빠르게 파악할 수 있다.

### 사전 지식
트랜스포머 구조, LoRA/QLoRA, 양자화(INT8/INT4)의 기본 개념, 분산 학습(DDP/FSDP)에 대한 입문 수준 이해.

### 관련 논문
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)

### 실무 적용
Agentic AI 제품의 추론 비용 절감, 온디바이스 LLM/ASR/TTS 배포, 멀티모달 모델 서빙 아키텍처 설계 등 CPO 관점에서 비용/성능 트레이드오프를 결정할 때 직접적인 레퍼런스가 된다.

---

## 추천 읽기 순서
1. Pan & Yang 2010 — 전이학습의 개념적 프레임 잡기
2. Deng et al. 2009 — 데이터 중심 사고와 사전학습 패러다임의 기원
3. Xu et al. 2024 — 현재의 파운데이션 모델 효율화 지형 파악

## 핵심 테이크어웨이
- 전이학습은 단순한 기법이 아니라 "지식을 옮기는 문제"라는 사고 프레임이며, 모든 사전학습 모델 활용의 근간이다.
- ImageNet은 모델이 아닌 "데이터"가 패러다임을 바꾼 사례이며, 오늘날 파운데이션 모델 데이터 큐레이션 사고의 원형이다.
- 파운데이션 모델 시대의 차별화는 더 이상 단순히 "큰 모델"이 아니라 "효율적으로 운영 가능한 큰 모델"에 있다.

## 다음 토픽과의 연결
다음 모듈은 NLP/Speech 데이터로 넘어가며, 워드 임베딩과 표현학습을 다룬다. 오늘 다룬 전이학습/사전학습 사고는 Word2Vec, BERT 등 NLP 표현학습의 출발점이 되며, 이후 파운데이션 모델로 자연스럽게 확장된다.
