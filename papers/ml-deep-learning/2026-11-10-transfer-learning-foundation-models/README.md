# Daily AI Paper Recommendations

> **Date:** 2026-11-10
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): Taskonomy: Disentangling Task Transfer Learning
- **Authors:** Amir R. Zamir, Alexander Sax, William Shen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
- **Year:** 2018 (CVPR 2018 Best Paper)
- **arXiv:** https://arxiv.org/abs/1804.08328
- **PDF:** [./taskonomy-task-transfer-zamir-2018.pdf](./taskonomy-task-transfer-zamir-2018.pdf)
- **Citation Count:** ~1,800+

### 요약
"어떤 태스크에서 배운 표현이 어떤 다른 태스크에 도움이 되는가?"를 처음으로 **계산 가능한 지도(taxonomy)** 로 만든 논문입니다. 깊이 추정, 표면 법선, 엣지, 세그멘테이션 등 26개 시각 태스크에 대해 수천 개의 전이 조합을 실험해, 태스크 간 1차·고차 전이 관계를 그래프로 정리했습니다. 이 지도를 활용하면 10개 태스크를 풀 때 필요한 라벨 데이터를 약 2/3 줄이면서도 성능을 거의 유지할 수 있었습니다.

### 핵심 기여
- 26개 2D/2.5D/3D/시맨틱 태스크 간 전이 가능성(transferability)을 정량적으로 측정하는 프레임워크 제시
- 소스 태스크 조합을 선택하는 문제를 이진 정수 계획(BIP)으로 풀어, 주어진 라벨 예산 내 최적 전이 경로를 찾는 방법 제안
- 400만 장 규모의 실내 장면 데이터셋과 사전학습 모델 공개 — 이후 멀티태스크·전이 연구의 표준 벤치마크

### 이 논문이 중요한 이유
전이 학습을 "ImageNet으로 사전학습하고 파인튜닝하면 된다"는 관성에서 꺼내, **소스 태스크 선택 자체가 설계 결정**이라는 관점을 심어줍니다. 파운데이션 모델 시대에도 "어떤 사전학습 목표·데이터가 내 다운스트림 태스크에 잘 전이되는가?"는 여전히 핵심 질문이며, Taskonomy는 이를 실험으로 답하는 방법론의 원형입니다.

### 사전 지식
- 인코더-디코더 구조와 Feature Extraction / Fine-tuning의 차이
- 멀티태스크 학습(Multi-task Learning) 기본 개념
- 컴퓨터 비전 태스크 유형(분류, 세그멘테이션, 깊이 추정 등)

### 관련 논문
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)
- [Which Tasks Should Be Learned Together in Multi-task Learning? (Standley et al., 2019)](https://arxiv.org/abs/1905.07553)
- [Task2Vec: Task Embedding for Meta-Learning (Achille et al., 2019)](https://arxiv.org/abs/1902.03545)

### 실무 적용
- 신규 도메인 모델 개발 시, 여러 사전학습 체크포인트를 소량 데이터로 빠르게 프로빙(linear probe)해 "전이 적합도"를 먼저 측정하는 습관
- 멀티태스크 헤드를 하나의 백본에 얹을 때, 서로 방해하는 태스크(negative transfer)를 분리하는 근거
- 라벨링 예산이 제한된 B2B 프로젝트에서 "어떤 보조 태스크 라벨을 먼저 모을지" 우선순위 결정에 활용

---

## Paper 2 (Classic): Domain-Adversarial Training of Neural Networks (DANN)
- **Authors:** Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky
- **Year:** 2016 (JMLR, vol. 17; arXiv 2015)
- **arXiv:** https://arxiv.org/abs/1505.07818
- **PDF:** [./domain-adversarial-training-dann-ganin-2016.pdf](./domain-adversarial-training-dann-ganin-2016.pdf)
- **Citation Count:** ~10,000+

### 요약
학습 데이터(소스 도메인)와 실제 서비스 데이터(타깃 도메인)의 분포가 다를 때, 라벨 없는 타깃 데이터만으로 모델을 적응시키는 **비지도 도메인 적응(Unsupervised Domain Adaptation)** 의 대표 논문입니다. 본 태스크는 잘 풀면서도 "이 입력이 어느 도메인에서 왔는지"는 구분할 수 없는 특징을 학습하도록, **Gradient Reversal Layer(GRL)** 하나를 네트워크에 추가하는 간단한 방법을 제안했습니다.

### 핵심 기여
- 도메인 적응 이론(H-divergence, Ben-David et al.)을 신경망 학습 목표로 직접 연결
- 역전파 시 그래디언트 부호만 뒤집는 GRL로, 일반 SGD만으로 적대적 학습을 구현 — 몇 줄 코드로 기존 모델에 적용 가능
- 감성 분석, 이미지 분류, 사람 재식별(Re-ID) 등 다양한 도메인에서 당시 SOTA 달성

### 이 논문이 중요한 이유
실무에서 모델이 실패하는 가장 흔한 이유는 "학습 때 본 데이터와 운영 데이터가 다르다"는 **분포 이동(distribution shift)** 입니다. DANN은 이 문제를 "표현 공간에서 도메인 차이를 지운다"는 명확한 직관으로 풀었고, 이후 적대적 학습·공정성(fairness)·도메인 일반화 연구 전반에 GRL 패턴이 퍼졌습니다.

### 사전 지식
- 역전파(Backpropagation)와 계산 그래프
- GAN의 적대적 학습(min-max) 개념
- 공변량 이동(Covariate Shift)과 도메인 적응의 정의

### 관련 논문
- [Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)](https://arxiv.org/abs/1409.7495)
- [Adversarial Discriminative Domain Adaptation / ADDA (Tzeng et al., 2017)](https://arxiv.org/abs/1702.05464)
- [Learning Transferable Features with Deep Adaptation Networks / DAN (Long et al., 2015)](https://arxiv.org/abs/1502.02791)

### 실무 적용
- 스튜디오 녹음으로 학습한 음성·더빙 모델을 실제 사용자 환경(잡음, 마이크 차이)에 적응시킬 때 라벨 없는 운영 데이터 활용
- 화자·언어·채널 정보를 표현에서 제거하고 싶을 때(예: 화자 독립 감정 인식, 공정성 제약) GRL로 "지우고 싶은 속성" 분류기를 적대적으로 붙이는 패턴
- 운영 데이터 드리프트 모니터링에서 "도메인 분류기 정확도"를 분포 이동 지표로 사용

---

## Paper 3 (Recent): Gemma 3 Technical Report
- **Authors:** Gemma Team, Google DeepMind (Aishwarya Kamath et al.)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.19786
- **PDF:** [./gemma-3-technical-report-kamath-2025.pdf](./gemma-3-technical-report-kamath-2025.pdf)
- **Citation Count:** ~1,000+

### 요약
1B~27B 파라미터 규모의 경량 오픈 파운데이션 모델 Gemma 3의 기술 보고서입니다. SigLIP 기반 비전 인코더를 붙여 이미지를 "소프트 토큰"으로 받는 멀티모달 모델로 확장했고, 로컬:글로벌 어텐션 레이어를 5:1로 섞어 128K 컨텍스트에서도 KV 캐시 메모리를 크게 줄였습니다. 모든 크기의 모델을 **지식 증류(knowledge distillation)** 로 학습해, 작은 모델로도 이전 세대 대형 모델에 근접한 성능을 냈습니다.

### 핵심 기여
- 로컬(슬라이딩 윈도우) 1024 토큰 레이어 5개당 글로벌 레이어 1개를 두는 인터리브 구조로 긴 컨텍스트의 메모리 비용 절감 (글로벌 레이어 RoPE base 10k → 1M)
- 고정 해상도 비전 인코더의 한계를 보완하는 Pan & Scan 추론 기법으로 다양한 종횡비 이미지 처리
- 대형 교사 모델로부터의 증류 + 수학·추론·다국어·멀티모달 중심 후처리 학습으로, 27B가 LMArena에서 훨씬 큰 모델과 경쟁하는 수준 달성

### 이 논문이 중요한 이유
"파운데이션 모델 → 다운스트림 전이"가 오늘날 실제로 어떻게 이뤄지는지 보여주는 최신 레퍼런스입니다. 사전학습된 비전 인코더(전이)를 언어 모델에 결합하고, 큰 모델의 지식을 작은 모델로 옮기는(증류 = 모델 간 전이) 두 축이 한 문서에 정리되어 있습니다. 온디바이스·저비용 서빙을 고민하는 AI 엔지니어에게 아키텍처 선택의 근거를 줍니다.

### 사전 지식
- Transformer 디코더 구조, RoPE 위치 인코딩, KV 캐시
- 지식 증류(Hinton et al., 2015)의 기본 개념
- 비전-언어 모델(VLM)에서 이미지 인코더와 LLM을 연결하는 방식

### 관련 논문
- [Gemma 2: Improving Open Language Models at a Practical Size (Gemma Team, 2024)](https://arxiv.org/abs/2408.00118)
- [Sigmoid Loss for Language Image Pre-Training / SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)

### 실무 적용
- 4B/12B 모델을 LoRA로 파인튜닝해 도메인 특화 멀티모달 어시스턴트(문서 이해, 화면 캡처 QA) 구축
- 128K 컨텍스트 + 낮은 KV 메모리 덕분에 단일 GPU에서 긴 문서 RAG·에이전트 메모리 실험 가능
- 1B 모델은 온디바이스·엣지 시나리오(모바일 요약, 오프라인 분류)의 베이스라인으로 활용

---

## 추천 읽기 순서
1. **DANN** — 전이 학습의 근본 문제인 "분포 차이"를 가장 명확하게 정의하고 다룹니다. 이론 섹션(3장)은 가볍게, GRL 구조(Figure 1)는 꼼꼼히.
2. **Taskonomy** — 도메인이 아닌 "태스크" 간 전이로 관점을 넓힙니다. 결과 그래프(Figure 1, 태스크 전이 지도) 중심으로 읽기.
3. **Gemma 3** — 위 두 개념(표현 전이, 모델 간 지식 전이)이 2025년 파운데이션 모델에서 어떻게 구현되는지 확인. 2~3장(아키텍처·사전학습)이 핵심.

## 핵심 테이크어웨이
- **전이의 두 축:** 무엇에서(소스 태스크·데이터, Taskonomy) 무엇으로(타깃 도메인, DANN) 옮기는지를 분리해서 생각해야 실패 원인을 진단할 수 있습니다.
- **좋은 표현 = 필요한 것만 남긴 표현:** DANN은 도메인 정보를 지우고, Taskonomy는 태스크 간 공유 구조를 찾습니다. 둘 다 "표현에 무엇을 담을지"의 설계 문제입니다.
- **파운데이션 모델 시대의 전이:** 오늘날 전이는 "사전학습 인코더 결합 + 증류 + 파인튜닝"의 조합이며, Gemma 3는 이를 효율(메모리·크기) 관점에서 최적화한 사례입니다.
- **질문으로 남기기:** 내 제품의 운영 데이터는 학습 데이터와 얼마나 다른가? 그 차이를 측정하는 지표를 갖고 있는가?

## 다음 토픽과의 연결
다음 모듈(Module 4: NLP and Speech Data)은 **Word Embeddings and Representation Learning**에서 시작합니다. 오늘 다룬 "전이 가능한 표현"이라는 아이디어가 텍스트에서는 Word2Vec·GloVe 같은 단어 임베딩으로 처음 꽃피었고, 이것이 BERT·GPT 같은 언어 파운데이션 모델로 이어집니다. 비전에서 본 전이 학습의 원리가 언어에서 어떻게 반복되는지 비교하며 읽어보세요.
