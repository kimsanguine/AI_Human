# Daily AI Paper Recommendations

> **Date:** 2026-06-28
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
- **Authors:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- **Year:** 2019 (JMLR 2020)
- **arXiv:** https://arxiv.org/abs/1910.10683
- **PDF:** [./t5-text-to-text-transfer-transformer-raffel-2019.pdf](./t5-text-to-text-transfer-transformer-raffel-2019.pdf)
- **Citation Count:** ~21,000+ (Google Scholar 기준)

### 요약
모든 NLP 문제(번역, 요약, 분류, QA 등)를 "텍스트를 입력받아 텍스트를 출력한다"는 단일 text-to-text 형식으로 통일한 전이학습 프레임워크다. 사전학습 목표, 아키텍처, 데이터셋, 전이 방식 등 전이학습의 거의 모든 설계 변수를 체계적인 통제 실험으로 비교하고, 그 결론을 대규모(11B 파라미터)와 새 코퍼스(C4)에 결합해 다수 벤치마크에서 SOTA를 달성했다.

### 핵심 기여
- 모든 텍스트 태스크를 동일한 입력/출력 인터페이스로 환원하는 text-to-text 패러다임 제시 (태스크별 헤드/손실 불필요)
- 사전학습 목표·모델 구조·코퍼스·파인튜닝 전략을 대규모 ablation으로 정량 비교한 "전이학습 설계 지침서"
- Colossal Clean Crawled Corpus(C4)라는 정제된 대규모 사전학습 데이터셋 공개

### 이 논문이 중요한 이유
오늘날 LLM이 "하나의 모델로 모든 태스크를 프롬프트로 처리한다"는 사고의 뿌리가 되는 논문이다. AI 엔지니어는 사전학습-파인튜닝 전이의 어떤 선택이 실제로 성능을 좌우하는지를 데이터로 배울 수 있고, instruction tuning·멀티태스크 학습 설계의 직관을 얻는다.

### 사전 지식
- Transformer 인코더-디코더 구조와 attention (Vaswani et al., 2017)
- 사전학습(pre-training)과 파인튜닝(fine-tuning)의 차이
- BERT 류 masked language modeling과 autoregressive LM의 차이

### 관련 논문
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

### 실무 적용
요약·분류·추출 등 여러 NLP 기능을 하나의 모델로 서빙할 때 태스크별 모델을 두지 않고 프롬프트/프리픽스로 분기하는 설계의 근거가 된다. FLAN-T5 등 파생 모델은 지금도 경량 사내 NLP 파이프라인의 기본 백본으로 쓰인다.

---

## Paper 2 (Classic): A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
- **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2002.05709
- **PDF:** [./simclr-contrastive-learning-chen-2020.pdf](./simclr-contrastive-learning-chen-2020.pdf)
- **Citation Count:** ~25,000+ (Google Scholar 기준)

### 요약
특수한 아키텍처나 메모리 뱅크 없이, 같은 이미지의 두 증강 뷰를 가깝게·다른 이미지를 멀게 학습하는 단순한 대조학습(contrastive learning) 프레임워크다. 라벨 없이 학습한 표현 위에 선형 분류기만 얹어도 ImageNet top-1 76.5%를 달성해, 지도학습 ResNet-50에 필적함을 보였다.

### 핵심 기여
- 데이터 증강 "조합"이 대조학습 태스크의 난이도와 표현 품질을 결정하는 핵심 변수임을 규명
- 표현과 대조손실 사이에 학습 가능한 비선형 projection head를 두면 품질이 크게 향상됨을 발견
- 대조학습은 지도학습보다 큰 배치와 긴 학습에서 더 큰 이득을 본다는 점을 정량화

### 이 논문이 중요한 이유
"라벨 없는 대규모 데이터로 범용 표현을 학습한다"는 자기지도학습(self-supervised learning) 사고의 대표작으로, 비전 파운데이션 모델의 사전학습 레시피의 기초가 된다. 라벨 비용이 큰 실무 환경에서 표현학습 전략을 설계할 때 필독이다.

### 사전 지식
- CNN(특히 ResNet)과 이미지 분류 기본기
- 데이터 증강(crop, color jitter, blur 등)의 개념
- InfoNCE/대조손실의 기본 직관

### 관련 논문
- [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo, He et al., 2019)](https://arxiv.org/abs/1911.05722)
- [Masked Autoencoders Are Scalable Vision Learners (MAE, He et al., 2021)](https://arxiv.org/abs/2111.06377)

### 실무 적용
라벨이 적은 도메인(의료영상, 제조 결함, 위성영상 등)에서 비라벨 데이터로 백본을 사전학습한 뒤 소량 라벨로 파인튜닝하는 전략의 표준 근거다. 임베딩 기반 검색·유사도 시스템의 표현학습에도 직접 응용된다.

---

## Paper 3 (Recent): The Llama 3 Herd of Models
- **Authors:** Aaron Grattafiori, Abhimanyu Dubey, et al. (Llama Team, Meta AI)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.21783
- **PDF:** [./llama-3-herd-of-models-grattafiori-2024.pdf](./llama-3-herd-of-models-grattafiori-2024.pdf)
- **Citation Count:** ~3,500+ (Google Scholar 기준, 2024년 발표 대비 매우 높음)

### 요약
다국어·코딩·추론·도구 사용을 기본 지원하는 파운데이션 모델 군(8B/70B/405B)을 공개한 기술 보고서다. 최대 모델은 405B 파라미터의 dense Transformer로 128K 컨텍스트를 지원하며, GPT-4급 품질을 다수 태스크에서 보인다. 사전학습 데이터·스케일링·사후학습(SFT/DPO) 파이프라인과 이미지·비디오·음성을 결합하는 compositional 멀티모달 접근을 상세히 기술한다.

### 핵심 기여
- 데이터 큐레이션·스케일링 법칙·인프라까지 포함한 프런티어급 모델의 "엔드투엔드 제작 레시피"를 투명하게 공개
- 단순하지만 견고한 설계(dense Transformer + SFT + DPO)로 복잡한 RLHF 없이 고품질 정렬 달성
- 이미지/비디오/음성 인코더를 결합하는 모듈식 멀티모달 확장 실험 공유

### 이 논문이 중요한 이유
2024년 이후 오픈 가중치 LLM 생태계의 사실상 기준점으로, 파운데이션 모델을 "어떻게 실제로 만드는가"를 가장 구체적으로 보여주는 1차 자료다. 데이터·정렬·평가·서빙 전반의 의사결정을 학습하려는 AI 엔지니어에게 필독이다.

### 사전 지식
- Transformer/LLM 기본 구조와 스케일링 법칙(Kaplan et al., 2020)
- SFT(지도 파인튜닝)와 선호 정렬(DPO/RLHF)의 개념
- 사전학습 데이터 파이프라인과 평가 벤치마크 개념

### 관련 논문
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

### 실무 적용
온프레미스/프라이빗 환경에서 상용 API 대체로 오픈 가중치 모델을 도입할 때의 모델 선택·파인튜닝·정렬 기준이 된다. 405B의 합성데이터 생성으로 소형 모델을 distillation하는 등, 비용 최적화 제품 설계의 실전 레퍼런스다.

---

## 추천 읽기 순서
1. **T5** — 전이학습의 설계 변수를 통제 실험으로 이해하며 "사전학습→파인튜닝" 사고의 토대를 잡는다.
2. **SimCLR** — 라벨 없이 표현을 학습하는 자기지도 패러다임으로 시야를 확장한다.
3. **Llama 3** — 위 두 흐름(전이학습 + 대규모 표현학습)이 오늘날 파운데이션 모델 제작으로 어떻게 수렴했는지 확인한다.

## 핵심 테이크어웨이
- 전이학습의 성패는 모델 크기 못지않게 **사전학습 목표·데이터 품질·전이 전략의 조합**에서 갈린다 (T5).
- **라벨 없는 데이터 + 잘 설계된 증강/목적함수**만으로 지도학습에 필적하는 범용 표현을 얻을 수 있다 (SimCLR).
- 프런티어 파운데이션 모델은 화려한 기법보다 **데이터 큐레이션·스케일·견고한 정렬 파이프라인의 공학적 완성도**가 핵심이다 (Llama 3).

## 다음 토픽과의 연결
오늘의 주제는 Module 3의 마지막인 전이학습/파운데이션 모델로, Module 4(NLP)의 단어 임베딩·Transformer·BERT로 자연스럽게 이어진다. 특히 T5의 text-to-text 사고와 Llama 3의 LLM 제작 레시피는 이후 Module 6(LLM)·Module 7(프롬프트)·Module 9(RAG)에서 반복적으로 재등장하는 기반 개념이다.
