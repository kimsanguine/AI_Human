# Daily AI Paper Recommendations

> **Date:** 2026-08-21
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): CNN Features off-the-shelf: an Astounding Baseline for Recognition
- **Authors:** Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, Stefan Carlsson
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1403.6382
- **PDF:** [./cnn-features-off-the-shelf-razavian-2014.pdf](./cnn-features-off-the-shelf-razavian-2014.pdf)
- **Citation Count:** ~9,000+

### 요약
ImageNet으로 학습된 OverFeat CNN에서 뽑아낸 중간 레이어 활성값을 아무런 파인튜닝 없이 그대로 특징 벡터로 쓰고, 그 위에 선형 SVM 하나만 얹었다. 이 단순한 조합이 객체 분류, 장면 인식, 세밀 분류(fine-grained), 속성 검출, 이미지 검색까지 전혀 다른 태스크들에서 당시 각 분야의 정교하게 튜닝된 SOTA 시스템을 일관되게 앞질렀다. "특징을 설계하는 시대는 끝났고, 사전학습 표현을 재사용하는 시대가 왔다"는 것을 실증한 논문이다.

### 핵심 기여
- 대규모 사전학습 CNN의 표현이 **태스크 비종속적(generic)**임을 여러 도메인에 걸쳐 체계적으로 입증
- 파인튜닝 없는 "frozen feature + 선형 분류기" 파이프라인을 표준 베이스라인으로 정착시킴
- 수작업 특징(SIFT, HOG, Fisher Vector) 기반 컴퓨터 비전 파이프라인의 종식을 사실상 선언
- 데이터가 적은 도메인에서도 전이만으로 강력한 성능을 낼 수 있음을 보여 전이학습의 실용성을 대중화

### 이 논문이 중요한 이유
오늘날 "파운데이션 모델을 임베딩 추출기로 쓴다"는 관행의 원형이 이 논문이다. CLIP 임베딩으로 검색을 만들고, 텍스트 임베딩 모델로 RAG를 만들고, 사전학습 백본을 얼려둔 채 헤드만 학습시키는 모든 작업이 이 논문이 제시한 논리 위에 서 있다. AI 엔지니어에게 "직접 학습할 것인가, 남의 표현을 재사용할 것인가"는 매일 마주치는 결정이며, 이 논문은 후자가 왜 대부분의 경우 합리적인 출발점인지에 대한 최초의 강력한 증거다.

### 사전 지식
- CNN의 레이어별 표현 구조(저수준 엣지 → 고수준 의미)
- ImageNet/ILSVRC 벤치마크와 AlexNet, OverFeat 아키텍처
- 선형 SVM, mAP 등 분류/검색 평가 지표
- feature extraction vs. fine-tuning의 차이

### 관련 논문
- [DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition (Donahue et al., 2013)](https://arxiv.org/abs/1310.1531)
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)
- [Rich feature hierarchies for accurate object detection and semantic segmentation / R-CNN (Girshick et al., 2013)](https://arxiv.org/abs/1311.2524)
- [OverFeat: Integrated Recognition, Localization and Detection using CNNs (Sermanet et al., 2013)](https://arxiv.org/abs/1312.6229)

### 실무 적용
이미지 중복 제거, 상품 유사도 검색, 콘텐츠 모더레이션 같은 기능에서 여전히 첫 번째 접근은 "사전학습 백본 → 임베딩 → 인덱싱"이다. 학습 데이터가 수천 장 수준일 때 풀 파인튜닝보다 frozen feature + 경량 분류기가 비용·시간·과적합 측면에서 유리한 경우가 많으므로, 제품 초기 단계의 ML 기능은 이 베이스라인부터 측정하고 시작하는 것이 원칙이다.

---

## Paper 2 (Classic): Masked Autoencoders Are Scalable Vision Learners
- **Authors:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2111.06377
- **PDF:** [./masked-autoencoders-he-2021.pdf](./masked-autoencoders-he-2021.pdf)
- **Citation Count:** ~11,000+

### 요약
이미지 패치의 75%를 가리고 원본 픽셀을 복원하도록 학습시키는 자기지도 사전학습 방법(MAE)을 제안했다. 인코더는 가려지지 않은 25% 패치만 처리하고, 가벼운 디코더가 마스크 토큰과 함께 전체를 복원하기 때문에 학습 비용이 3배 이상 줄어든다. 이렇게 라벨 없이 사전학습한 ViT-Huge가 ImageNet-1K만으로 87.8%를 달성하며, 검출·분할 등 다운스트림 전이에서도 지도학습 사전학습을 능가했다.

### 핵심 기여
- **비대칭 인코더-디코더 구조**: 인코더가 마스크 토큰을 보지 않게 하여 연산량과 메모리를 대폭 절감
- **높은 마스킹 비율(75%)** 이 비전에서 오히려 유의미한 자기지도 과제를 만든다는 반직관적 발견
- 라벨 없는 데이터만으로 대규모 비전 모델을 확장 가능하게(scalable) 학습하는 경로 제시
- NLP의 BERT식 마스크 예측 패러다임을 비전으로 성공적으로 이식하여 모달리티 간 사전학습 원리를 통합

### 이 논문이 중요한 이유
파운데이션 모델의 본질은 "라벨이 아니라 데이터 자체로부터 표현을 배운다"는 것이다. MAE는 그 원리가 텍스트뿐 아니라 이미지에서도 통하며, 심지어 더 저렴하게 통한다는 것을 보여줬다. 오늘날 멀티모달 파운데이션 모델의 비전 인코더 상당수가 MAE 계열 또는 그 아이디어를 흡수한 대조학습 하이브리드다. 데이터는 많은데 라벨은 없는 상황(대부분의 실제 기업 데이터)에서 무엇을 할 수 있는지를 정의한 논문이다.

### 사전 지식
- Vision Transformer(ViT)의 패치 임베딩과 self-attention
- BERT의 Masked Language Modeling 목적함수
- 자기지도학습의 두 계열: 대조학습(SimCLR, MoCo) vs. 생성/복원 기반
- linear probing과 fine-tuning 평가의 차이

### 관련 논문
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale / ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [BEiT: BERT Pre-Training of Image Transformers (Bao et al., 2021)](https://arxiv.org/abs/2106.08254)
- [Emerging Properties in Self-Supervised Vision Transformers / DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (Woo et al., 2023)](https://arxiv.org/abs/2301.00808)

### 실무 적용
도메인 특화 이미지(의료 영상, 제조 결함, 위성 사진, 문서 스캔)처럼 공개 사전학습 모델의 분포와 다른 데이터를 다룰 때, 라벨링 예산을 쓰기 전에 보유한 비라벨 데이터로 MAE 사전학습을 먼저 수행하는 전략이 표준적으로 쓰인다. 라벨링 비용을 수분의 일로 줄이면서 다운스트림 정확도를 올릴 수 있어, 데이터 확보 전략과 모델 로드맵을 함께 설계해야 하는 PM/엔지니어에게 직접적인 의사결정 근거가 된다.

---

## Paper 3 (Recent): Qwen3 Technical Report
- **Authors:** Qwen Team (An Yang et al.)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2505.09388
- **PDF:** [./qwen3-technical-report-qwen-2025.pdf](./qwen3-technical-report-qwen-2025.pdf)
- **Citation Count:** ~1,000+ (2025년 공개 이후 급증)

### 요약
0.6B부터 235B(MoE)까지 이어지는 Qwen3 모델 패밀리의 학습·정렬·평가 전 과정을 공개한 기술 보고서다. 가장 큰 특징은 복잡한 추론을 수행하는 **thinking mode**와 즉답하는 **non-thinking mode**를 하나의 모델에 통합하고, 추론 시 사용자가 **thinking budget**을 지정해 지연시간과 성능을 조절할 수 있게 한 점이다. 다국어 지원은 Qwen2.5의 29개에서 119개 언어·방언으로 확장되었으며, 전 모델이 Apache 2.0으로 공개되었다.

### 핵심 기여
- 추론 모드와 일반 모드를 **단일 모델에 통합**하여 모델 스위칭 없이 태스크 난이도에 대응
- **Thinking budget** 메커니즘으로 추론 연산량을 런타임에 제어 가능한 파라미터로 전환
- 대형 모델의 지식을 소형 모델로 옮기는 **strong-to-weak 증류**로, 소형 모델의 학습 비용을 크게 절감
- MoE 구조로 활성 파라미터를 줄이면서도 코드·수학·에이전트 벤치마크에서 SOTA급 성능 달성
- 오픈 웨이트 파운데이션 모델이 프론티어 폐쇄형 모델과의 격차를 좁히고 있음을 실증

### 이 논문이 중요한 이유
파운데이션 모델을 "가져다 쓰는" 입장에서 이 보고서는 사실상 제품 설계 문서다. thinking budget은 **추론 비용을 제품 레버로 다루는 방식**의 구체적 사례이고, strong-to-weak 증류는 대형 모델을 서비스에 그대로 태우기 어려운 팀이 어떻게 품질을 유지하며 비용을 낮추는지를 보여준다. 전이학습의 현대적 형태가 "파인튜닝"에서 "증류 + 프롬프트 + 어댑터"로 이동했음을 가장 명확히 보여주는 최신 문서다.

### 사전 지식
- Transformer 디코더 구조와 Mixture-of-Experts(MoE) 라우팅
- 사전학습 → SFT → 선호 정렬(RLHF/DPO/GRPO)로 이어지는 LLM 학습 파이프라인
- Knowledge distillation의 기본 개념
- MMLU, GPQA, LiveCodeBench, AIME 등 LLM 평가 벤치마크
- Chain-of-Thought와 추론 시간 스케일링(test-time scaling)의 개념

### 관련 논문
- [Qwen2.5 Technical Report (Qwen Team, 2024)](https://arxiv.org/abs/2412.15115)
- [The Llama 3 Herd of Models (Grattafiori et al., 2024)](https://arxiv.org/abs/2407.21783)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)

### 실무 적용
자체 SaaS에서 LLM을 서빙할 때 Qwen3 계열은 "오픈 웨이트 + Apache 2.0 + 사이즈 스펙트럼"이라는 조건 덕에 온프레미스/VPC 배포의 현실적 후보다. 특히 thinking budget은 요금제 설계와 직결된다 — 무료 티어는 non-thinking으로 빠르게, 유료 티어는 thinking budget을 늘려 정확도를 높이는 식의 **가격-품질 계단**을 모델 레벨에서 구현할 수 있다. 또한 대형 모델로 데이터를 만들어 소형 모델을 증류하는 패턴은 응답 지연이 중요한 에이전트 워크플로에서 비용을 한 자릿수 배 줄이는 표준 전술이다.

---

## 추천 읽기 순서

1. **Razavian et al. (2014)** — 가장 짧고(6페이지) 개념이 단순하다. "사전학습 표현을 재사용한다"는 전이학습의 원형을 먼저 체화한다.
2. **He et al. (2021, MAE)** — 그 표현을 라벨 없이 어떻게 만들 것인가로 질문을 옮긴다. 사전학습 목적함수 설계의 관점을 얻는다.
3. **Qwen3 Technical Report (2025)** — 위 두 원리가 수천억 파라미터 규모에서 어떻게 엔지니어링되고, 어떤 제품 레버로 변환되는지 확인한다.

이 순서는 "표현 재사용 → 표현 생성 → 표현 배포"라는 흐름을 따른다.

## 핵심 테이크어웨이

- **전이학습의 본질은 데이터 효율이다.** 라벨이 부족한 상황에서 남이 학습한 표현을 빌려오는 것은 우회로가 아니라 정공법이다.
- **사전학습 과제의 난이도가 표현의 품질을 결정한다.** MAE의 75% 마스킹처럼, 너무 쉬운 과제는 쓸모없는 표현을 만든다.
- **파운데이션 모델 시대의 전이는 파인튜닝만이 아니다.** 프롬프트, 어댑터(LoRA), 증류, 임베딩 추출이 모두 전이의 형태이며 비용·성능·운영 난이도가 각각 다르다.
- **추론 비용은 이제 설계 변수다.** Qwen3의 thinking budget처럼, 모델의 "생각하는 양"을 제품이 제어할 수 있게 되면서 아키텍처 결정이 곧 가격 결정이 된다.
- **비라벨 데이터는 자산이다.** 자기지도 사전학습이 실용화된 이상, 쌓아둔 로그·이미지·음성은 라벨링 예산보다 먼저 검토해야 할 자원이다.

## 다음 토픽과의 연결

Module 3의 마지막 토픽인 오늘 내용은 Module 4(NLP and Speech Data)의 **Word Embeddings and Representation Learning**으로 직접 이어진다. 오늘 다룬 "사전학습된 표현을 재사용한다"는 아이디어가 비전에서 언어로 옮겨가면 Word2Vec/GloVe의 정적 임베딩이 되고, 다시 문맥을 반영하면 BERT 계열의 문맥 임베딩이 된다. 특히 MAE의 마스크 복원 목적함수는 원래 BERT에서 왔으므로, 다음 토픽을 읽을 때 "같은 원리가 모달리티만 바꿔 반복된다"는 점을 확인하며 읽으면 좋다.
