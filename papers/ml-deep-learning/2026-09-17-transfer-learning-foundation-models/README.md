# Daily AI Paper Recommendations

> **Date:** 2026-09-17
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)
- **Authors:** Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
- **Year:** 2019 (CVPR 2020)
- **arXiv:** https://arxiv.org/abs/1911.05722
- **PDF:** [./moco-momentum-contrast-he-2019.pdf](./moco-momentum-contrast-he-2019.pdf)
- **Citation Count:** 약 18,000+

### 요약
MoCo는 대조 학습(contrastive learning)을 "동적 사전(dynamic dictionary) 조회" 문제로 재정의한 자기지도 학습 방법입니다. 큐(queue) 형태의 negative sample 저장소와 momentum으로 천천히 업데이트되는 key encoder를 도입해, 거대한 배치 없이도 매우 많은 negative를 사용할 수 있게 했습니다. 그 결과 라벨 없이 학습한 표현이 ImageNet 지도학습 사전학습 가중치를 여러 downstream 태스크(detection, segmentation)에서 처음으로 능가했습니다.

### 핵심 기여
- Negative sample을 큐에 저장해 배치 크기와 dictionary 크기를 분리 — GPU 메모리 제약에서 대조 학습을 해방
- Momentum encoder(EMA 업데이트)로 key 표현의 일관성을 유지, 학습 붕괴 방지
- 라벨 없는 사전학습이 지도학습 사전학습보다 전이(transfer) 성능이 좋을 수 있음을 7개 downstream 태스크에서 실증

### 이 논문이 중요한 이유
"사전학습 = 대량의 라벨 데이터"라는 전제를 깨뜨린 전환점입니다. 오늘날 모든 foundation model의 전제 조건인 "라벨 없는 데이터로 전이 가능한 표현을 만든다"를 시각 도메인에서 최초로 설득력 있게 증명했습니다. 또한 momentum encoder / 메모리 큐 패턴은 이후 SimCLR, BYOL, DINO, CLIP 계열 학습 레시피와 오늘날의 임베딩 모델 학습(대조 학습 기반 retrieval 모델)에 그대로 살아 있습니다. AI 엔지니어가 "우리 도메인 데이터는 라벨이 없다"는 현실적 제약을 다룰 때 가장 먼저 참조해야 하는 원전입니다.

### 사전 지식
- CNN 기본 구조와 ImageNet 사전학습/파인튜닝 워크플로
- InfoNCE loss와 대조 학습의 기본 개념 (positive/negative pair)
- Exponential moving average(EMA) 파라미터 업데이트
- Linear probing과 downstream fine-tuning 평가 방식의 차이

### 관련 논문
- [A Simple Framework for Contrastive Learning of Visual Representations / SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- [Improved Baselines with Momentum Contrastive Learning / MoCo v2 (Chen et al., 2020)](https://arxiv.org/abs/2003.04297)
- [Bootstrap Your Own Latent / BYOL (Grill et al., 2020)](https://arxiv.org/abs/2006.07733)
- [Representation Learning with Contrastive Predictive Coding / CPC (van den Oord et al., 2018)](https://arxiv.org/abs/1807.03748)

### 실무 적용
대조 학습 기반 사전학습은 라벨 비용이 큰 도메인(의료 영상, 제조 불량 검출, 리테일 상품 이미지)에서 표준 접근법이 되었습니다. 또한 MoCo의 negative queue 아이디어는 텍스트 임베딩 모델(E5, GTE, BGE 등) 학습에서 in-batch negative를 확장하는 기법으로 재사용되며, RAG 파이프라인의 retriever 품질과 직결됩니다. 자체 SaaS에서 도메인 특화 검색/유사도 기능을 만들 때, 라벨 없는 로그 데이터로 임베딩을 후속 학습(continued pretraining)하는 전략의 이론적 근거가 이 논문입니다.

---

## Paper 2 (Classic): Emerging Properties in Self-Supervised Vision Transformers (DINO)
- **Authors:** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
- **Year:** 2021 (ICCV 2021)
- **arXiv:** https://arxiv.org/abs/2104.14294
- **PDF:** [./dino-self-supervised-vision-transformers-caron-2021.pdf](./dino-self-supervised-vision-transformers-caron-2021.pdf)
- **Citation Count:** 약 11,000+

### 요약
DINO는 negative sample도, 라벨도 없이 "자기 증류(self-distillation with no labels)" 방식으로 Vision Transformer를 학습합니다. student 네트워크가 teacher(student의 EMA)의 출력 분포를 맞추도록 학습하며, centering과 sharpening으로 붕괴를 방지합니다. 놀라운 점은 학습 과정에서 명시적으로 가르치지 않은 능력 — 객체 경계를 담은 attention map과 k-NN만으로도 강력한 분류 성능 — 이 자연 발생(emerge)했다는 것입니다.

### 핵심 기여
- Negative pair 없이 teacher-student 자기 증류만으로 고품질 표현 학습 (centering + temperature sharpening으로 collapse 회피)
- ViT의 self-attention이 라벨 없이도 의미적 객체 분할(semantic segmentation) 정보를 학습함을 시각적으로 입증
- Frozen feature + k-NN 분류기만으로 ImageNet 80%대 성능 — "파인튜닝 없이 쓸 수 있는 범용 feature"의 등장

### 이 논문이 중요한 이유
Foundation model의 정의적 특성인 "창발적 능력(emergent capability)"을 시각 도메인에서 최초로 명확히 보여준 논문입니다. 또한 DINO → DINOv2 → DINOv3로 이어지는 계보는 오늘날 vision foundation model의 사실상 표준 백본이 되었고, 다운스트림 태스크에서 별도 파인튜닝 없이 frozen feature를 그대로 쓰는 실무 패턴을 정착시켰습니다. AI 엔지니어 관점에서 "모델을 학습시키는 것"과 "이미 학습된 표현을 조합하는 것" 사이의 무게중심이 후자로 이동한 분기점입니다.

### 사전 지식
- Vision Transformer(ViT)의 patch embedding, self-attention, [CLS] token 구조
- Knowledge distillation과 teacher-student 프레임워크
- Multi-crop augmentation, EMA teacher, softmax temperature
- Representation collapse 문제와 이를 막는 정규화 기법들

### 관련 논문
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [DINOv3 (Siméoni et al., 2025)](https://arxiv.org/abs/2508.10104)
- [Masked Autoencoders Are Scalable Vision Learners / MAE (He et al., 2021)](https://arxiv.org/abs/2111.06377)
- [An Image is Worth 16x16 Words / ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)

### 실무 적용
DINOv2/v3 frozen feature + 얇은 head 조합은 이미지 검색, 중복 제거, 품질 이상 탐지, 콘텐츠 모더레이션을 학습 데이터 거의 없이 구축하는 표준 레시피입니다. AI Avatar/더빙 파이프라인에서는 얼굴·프레임 유사도 판정, 장면 전환 감지, 생성 결과의 신원 일관성(identity consistency) 검증 지표로 활용할 수 있습니다. attention map이 별도 라벨 없이 객체 영역을 잡아주므로, 자동 마스킹·크롭 전처리를 저비용으로 구현하는 데도 적합합니다.

---

## Paper 3 (Recent): Scaling Native Multimodal Pre-Training From Scratch
- **Authors:** Haoyuan Wu, Aoqi Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, Bei Yu
- **Year:** 2026
- **arXiv:** https://arxiv.org/abs/2607.22043
- **PDF:** [./scaling-native-multimodal-pretraining-wu-2026.pdf](./scaling-native-multimodal-pretraining-wu-2026.pdf)
- **Citation Count:** 신규 논문 (2026-07 공개, 인용 집계 초기 단계)

### 요약
텍스트로 먼저 학습한 LLM에 비전 인코더를 붙이는 late-fusion 방식이 아니라, 처음부터 멀티모달 입력으로 학습하는 "native multimodal pre-training"의 스케일링 법칙을 체계적으로 규명한 논문입니다. 고정된 컴퓨트 예산에서 최적 모델 크기와 토큰 수가 power law를 따르며, 언어 목적함수와 멀티모달 목적함수가 서로 다른 스케일링 거동을 보인다는 것을 실증했습니다. 특히 멀티모달 할당 법칙은 데이터 혼합 비율에 매우 민감해, 텍스트 비중이 높은 혼합은 모델이 충분히 커진 뒤에야 컴퓨트 효율적이 됩니다.

### 핵심 기여
- Native multimodal 사전학습의 compute law와 allocation law를 분리 도출 — 모델 크기 / 토큰 수 / 데이터 혼합비의 efficiency frontier 제시
- 언어 할당 법칙은 데이터 구성에 거의 불변, 멀티모달 할당 법칙은 구성에 민감하다는 비대칭성 발견
- Native 멀티모달 학습이 순수 텍스트 공간 추론(spatial reasoning)까지 향상시키는 긍정적 cross-modal transfer를 downstream 평가로 확인
- 멀티모달 in-context learning이 late-fusion보다 견고하게 나타남을 관찰

### 이 논문이 중요한 이유
지금까지의 멀티모달 모델 개발은 "일단 붙여보고 튜닝한다"는 경험칙에 의존했습니다. 이 논문은 Kaplan/Chinchilla가 LLM에 제공한 것과 같은 예측 가능한 자원 배분 근거를 멀티모달 영역에 제공합니다. 즉, 학습 전에 "어느 크기 모델에, 몇 토큰을, 어떤 이미지:텍스트 비율로 줘야 하는가"를 계산할 수 있게 만든 실무적 의미가 큽니다. 또한 cross-modal transfer가 텍스트 능력까지 끌어올린다는 결과는 "멀티모달은 비용"이라는 통념을 뒤집는 전이 학습의 새 근거입니다.

### 사전 지식
- Scaling law의 기본 형태 (Kaplan et al. 2020, Chinchilla compute-optimal 개념)
- Late fusion(비전 인코더 + LLM 어댑터) vs. native/early fusion 아키텍처 차이
- Vision-language model의 학습 목적함수 구성 (언어 모델링 + 이미지-텍스트 정렬)
- IsoFLOP 분석과 compute-optimal 곡선 해석 방법

### 관련 논문
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)
- [Test-Time Scaling in Multimodal Foundation Models: A Comprehensive Survey (2026)](https://arxiv.org/abs/2606.08231)

### 실무 적용
자체 멀티모달 모델을 사전학습하거나 continued pretraining할 때, 데이터 큐레이션 예산을 어디에 쓸지 결정하는 계산 근거로 직접 사용할 수 있습니다. 예를 들어 소형 모델에서 텍스트 비중을 과도하게 높이는 것은 컴퓨트 낭비라는 결론은, 제한된 GPU 예산으로 도메인 특화 VLM을 만드는 팀의 데이터 믹스 설계를 바꿉니다. 또한 cross-modal transfer 효과는 "문서 이해 + 스크린샷 이해"가 필요한 Agentic AI 제품에서 멀티모달 학습 투자를 정당화하는 근거가 됩니다.

---

## 추천 읽기 순서

1. **MoCo (2019)** — 먼저 읽으세요. "라벨 없이 전이 가능한 표현을 만들 수 있는가?"라는 질문과 대조 학습의 메커니즘을 가장 명확하게 이해할 수 있습니다. 6페이지 분량으로 부담도 적습니다.
2. **DINO (2021)** — 다음으로 읽으세요. MoCo의 negative sample 의존을 제거하고 ViT로 넘어가면서, 표현 학습이 어떻게 "창발적 능력"을 갖게 되는지 확인합니다. Attention map 시각화(Figure 1)만 봐도 얻는 것이 큽니다.
3. **Scaling Native Multimodal Pre-Training (2026)** — 마지막으로 읽으세요. 앞 두 논문이 "어떻게 사전학습하는가"를 다뤘다면, 이 논문은 "얼마나, 어떤 비율로 사전학습해야 하는가"라는 자원 배분 문제를 다룹니다. 스케일링 법칙 사전 지식이 있으면 2배 빠르게 읽힙니다.

시간이 부족하면: DINO의 Figure 1~3 + 오늘 논문의 efficiency frontier 절만 읽어도 핵심 흐름은 잡힙니다.

## 핵심 테이크어웨이

- **전이 학습의 병목은 라벨이 아니라 목적함수 설계로 이동했다.** MoCo와 DINO는 라벨 없는 데이터로도 지도학습을 능가하는 표현을 만들 수 있음을 증명했고, 이후 모든 foundation model이 이 전제 위에 서 있습니다.
- **Negative sample은 필수가 아니다.** MoCo(대조) → DINO(자기 증류)의 전환은, 표현 붕괴를 막는 방법이 여러 갈래라는 것을 보여줍니다. 실무에서는 데이터 규모와 배치 크기 제약에 따라 선택이 달라집니다.
- **좋은 표현은 가르치지 않은 능력을 만든다.** DINO의 attention map은 segmentation 라벨 없이 객체 경계를 찾습니다. Foundation model 평가 시 "학습 목표 지표"만 보면 실제 가치를 놓칩니다.
- **Frozen feature 전략을 먼저 검토하라.** 파인튜닝보다 frozen feature + 얇은 head가 저비용·고안정성인 경우가 많습니다. 실험 순서는 항상 (1) frozen + linear probe → (2) 부분 파인튜닝 → (3) full 파인튜닝입니다.
- **멀티모달 학습은 이제 계산 가능한 의사결정이다.** 2026년 스케일링 연구는 모델 크기·토큰 수·데이터 혼합비의 최적 조합을 사전에 추정할 수 있게 만들었습니다. 감으로 데이터 믹스를 정하던 시대가 끝나가고 있습니다.
- **Cross-modal transfer는 순방향으로 작동한다.** 이미지를 함께 학습시키면 텍스트 공간 추론까지 좋아집니다. 멀티모달 투자는 비용이 아니라 성능 레버일 수 있습니다.

## 다음 토픽과의 연결

오늘 다룬 self-supervised 표현 학습은 "어떻게 라벨 없이 의미 있는 벡터를 만드는가"의 시각 도메인 버전입니다. 다음 모듈(Module 4: NLP and Speech Data)의 첫 토픽인 **Word Embeddings and Representation Learning**은 동일한 질문을 텍스트에서 다룹니다 — Word2Vec의 negative sampling은 MoCo의 InfoNCE와 본질적으로 같은 아이디어이며, 이 계보는 이후 BERT의 masked language modeling(오늘 언급한 MAE의 텍스트 원형)과 RAG의 임베딩 모델로 이어집니다. 오늘 잡아둔 "사전학습 → 전이 → frozen feature 활용"이라는 사고 틀을 그대로 가져가면 다음 토픽이 훨씬 빠르게 정리됩니다.
