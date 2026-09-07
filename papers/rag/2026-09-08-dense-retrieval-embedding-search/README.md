# Daily AI Paper Recommendations

> **Date:** 2026-09-08
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Dense Retrieval and Embedding Search

---

## Paper 1 (Classic): Latent Retrieval for Weakly Supervised Open Domain Question Answering (ORQA)
- **Authors:** Kenton Lee, Ming-Wei Chang, Kristina Toutanova
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1906.00300
- **PDF:** [./orqa-latent-retrieval-lee-2019.pdf](./orqa-latent-retrieval-lee-2019.pdf)
- **Citation Count:** 약 1,500회 이상

### 요약
기존 오픈 도메인 QA는 BM25 같은 블랙박스 검색기에 의존하거나, 어느 문서가 정답 근거인지 사람이 직접 라벨링해야 했다. 이 논문은 "질문-정답 문자열 쌍"만으로 검색기(retriever)와 리더(reader)를 end-to-end로 함께 학습하는 ORQA를 최초로 제시했다. 핵심 장치는 Inverse Cloze Task(ICT)라는 비지도 사전학습으로, 문장을 유사 질문으로, 그 주변 문맥을 유사 근거 문서로 삼아 검색기를 미리 학습시킨다.

### 핵심 기여
- 검색 라벨 없이 QA 쌍만으로 retriever + reader를 공동 학습하는 latent variable 프레임워크 제안
- Inverse Cloze Task(ICT): 검색 전용 비지도 사전학습 과제를 최초로 정의 — 이후 DPR·Contriever·REALM 계열의 출발점
- 사용자가 실제로 답을 모르고 묻는 데이터셋에서 BM25 대비 Exact Match 최대 19점 향상 — "학습된 검색"이 필요한 조건을 실증
- 전체 위키피디아 코퍼스를 사전 인코딩해 두고 MIPS(Maximum Inner Product Search)로 조회하는 dense retrieval 파이프라인의 원형 확립

### 이 논문이 중요한 이유
오늘 우리가 쓰는 RAG 파이프라인의 "왜 dense retrieval인가"에 대한 최초의 실증적 답변이다. DPR(2020)이 dense retrieval의 표준 레시피를 만들었다면, ORQA는 그 전제 — 검색기는 학습 가능하고, 학습된 검색기가 lexical 매칭을 이긴다 — 를 증명했다. 특히 ICT는 라벨이 없는 상황에서 임베딩 모델을 부트스트랩하는 방법론의 원조로, 사내 문서처럼 정답 라벨이 전혀 없는 도메인에 RAG를 도입할 때 지금도 그대로 쓰이는 사고방식이다.

### 사전 지식
- BERT 구조와 [CLS] 토큰 기반 문장 표현
- BM25 등 sparse lexical 검색의 동작 원리와 vocabulary mismatch 문제
- MIPS(Maximum Inner Product Search)와 근사 최근접 이웃 탐색의 개념
- Latent variable 모델과 marginal likelihood 학습 (근거 문서를 관측되지 않은 변수로 취급)

### 관련 논문
- [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al., 2020)](https://arxiv.org/abs/2002.08909)
- [Unsupervised Dense Information Retrieval with Contrastive Learning / Contriever (Izacard et al., 2021)](https://arxiv.org/abs/2112.09118)

### 실무 적용
정답 라벨이 없는 신규 도메인(사내 위키, 제품 매뉴얼, 고객 티켓)에 RAG를 붙일 때, ICT 스타일의 self-supervised 페어 생성 — 문단에서 문장 하나를 뽑아 질문 역할로, 나머지를 근거로 — 은 여전히 가장 저렴한 도메인 적응 전략이다. 또한 "리트리버를 고정된 부품이 아니라 학습 대상으로 본다"는 관점은 RAG 품질이 정체됐을 때 LLM을 바꾸기 전에 리트리버부터 파인튜닝하라는 실무 우선순위로 이어진다.

---

## Paper 2 (Classic): Matryoshka Representation Learning (MRL)
- **Authors:** Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi
- **Year:** 2022 (NeurIPS 2022)
- **arXiv:** https://arxiv.org/abs/2205.13147
- **PDF:** [./matryoshka-representation-learning-kusupati-2022.pdf](./matryoshka-representation-learning-kusupati-2022.pdf)
- **Citation Count:** 약 700회 이상

### 요약
임베딩 차원은 보통 학습 시점에 고정되며, 더 작은 벡터가 필요하면 모델을 다시 학습하거나 PCA로 압축해야 했다. MRL은 하나의 임베딩 안에 여러 granularity의 정보를 중첩해(러시아 인형처럼) 학습시켜, 앞쪽 d차원만 잘라 써도 그 차원에 맞춰 따로 학습한 모델과 비슷한 성능을 내게 만든다. 손실 함수에 여러 차원의 목적항을 동시에 거는 것만으로 구현되며, 추론·배포 시 추가 비용이 없다.

### 핵심 기여
- 하나의 모델로 8·16·…·2048차원을 동시에 지원하는 nested representation 학습 기법 제안
- ImageNet-1K 분류에서 동일 정확도 기준 임베딩 크기 최대 14배 축소
- 대규모 검색에서 최대 14배 실제 속도 향상 — 저차원으로 후보를 추리고 고차원으로 재정렬하는 adaptive retrieval 구성
- ViT/ResNet(비전), ALIGN(비전+언어), BERT(언어) 등 모달리티와 아키텍처를 가리지 않고 적용 가능함을 입증

### 이 논문이 중요한 이유
OpenAI text-embedding-3, Nomic Embed, Gemini Embedding 등 현재 상용 임베딩 API가 "차원을 골라 쓰세요"라고 말할 수 있는 이유가 바로 이 논문이다. RAG 시스템의 비용은 결국 벡터 DB의 저장 용량과 ANN 탐색 지연에 지배되는데, MRL은 재학습 없이 이 두 축을 조절할 수 있는 다이얼을 제공한다. AI 엔지니어에게는 "정확도 vs 비용" 트레이드오프를 아키텍처 수준이 아니라 런타임 설정 수준에서 다룰 수 있게 해 주는 도구다.

### 사전 지식
- Contrastive learning과 표현 학습의 기본 목적함수
- 임베딩 차원이 벡터 DB 저장 비용과 ANN 탐색 지연에 미치는 영향
- PCA·product quantization 등 기존 차원 축소/압축 기법과의 차이
- Coarse-to-fine (shortlist → rerank) 검색 파이프라인 개념

### 관련 논문
- [Text and Code Embeddings by Contrastive Pre-Training (Neelakantan et al., 2022)](https://arxiv.org/abs/2201.10005)
- [MTEB: Massive Text Embedding Benchmark (Muennighoff et al., 2022)](https://arxiv.org/abs/2210.07316)
- [Nomic Embed: Training a Reproducible Long Context Text Embedder (Nussbaum et al., 2024)](https://arxiv.org/abs/2402.01613)

### 실무 적용
벡터 DB 비용이 문제일 때 가장 먼저 검토할 카드다. 1536차원을 256차원으로 잘라 인덱싱하면 저장·메모리가 약 6분의 1로 줄고, 정확도 손실은 상당수 도메인에서 1~2%p 수준에 그친다. 실전 패턴은 2단계 구성이다 — 저차원 벡터로 top-100을 빠르게 뽑고, 동일 임베딩의 전체 차원으로 재정렬해 top-10을 확정한다. 인덱스를 두 벌 만들 필요 없이 같은 벡터를 잘라 쓰기만 하면 되므로 운영 복잡도도 늘지 않는다.

---

## Paper 3 (Recent): Gemini Embedding: Generalizable Embeddings from Gemini
- **Authors:** Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, et al. (Google DeepMind)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.07891
- **PDF:** [./gemini-embedding-lee-2025.pdf](./gemini-embedding-lee-2025.pdf)
- **Citation Count:** 약 100회 이상 (2026년 9월 기준, 빠르게 증가 중)

### 요약
Gemini LLM을 초기화 지점으로 삼아 단일 임베딩 모델을 학습하고, 다국어·코드·검색·분류·클러스터링을 하나의 모델로 커버한다. 250개 이상 언어, 100개 이상 태스크로 구성된 MMTEB에서 기존 SOTA를 큰 폭으로 앞섰으며, 특정 도메인에 특화된 전용 모델들도 함께 상회했다. 합성 데이터 생성, 학습 데이터 필터링, hard negative 마이닝에 LLM 자체를 활용하는 파이프라인이 핵심이다.

### 핵심 기여
- LLM 백본(Gemini)에서 임베딩 모델을 초기화하는 접근의 대규모 검증 — 다국어·코드 이해 능력이 임베딩으로 전이됨을 확인
- LLM 기반 데이터 정제 파이프라인: 합성 쿼리 생성, 품질 필터링, hard negative 선별을 자동화
- MMTEB(다국어·영어·코드) 전 영역에서 단일 통합 모델이 SOTA 달성 — 태스크별 전용 모델 운영 필요성을 약화
- Matryoshka 방식의 가변 차원 출력을 지원해 배포 환경에 맞춘 크기 조절 가능

### 이 논문이 중요한 이유
"임베딩 모델은 작은 BERT 계열"이라는 전제가 무너진 시점을 보여주는 문서다. NV-Embed·E5-Mistral·Qwen3-Embedding과 함께, LLM 백본 + 대조학습 파인튜닝이 표준 레시피가 되었음을 확정한다. 한국어를 포함한 다국어 RAG를 만들 때 모델 선택 기준이 어떻게 바뀌었는지, 그리고 데이터 파이프라인(합성·필터링·negative 마이닝)이 모델 구조보다 성능을 좌우한다는 최근 흐름을 한 편으로 파악할 수 있다.

### 사전 지식
- 대조학습(InfoNCE)과 in-batch negative / hard negative의 차이
- Decoder-only LLM을 임베딩 인코더로 바꾸는 pooling 전략 (last-token, mean pooling 등)
- MTEB / MMTEB 벤치마크의 태스크 구성과 지표 해석법
- Matryoshka Representation Learning (Paper 2) — 가변 차원 출력의 배경

### 관련 논문
- [NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models (Lee et al., 2024)](https://arxiv.org/abs/2405.17428)
- [Improving Text Embeddings with Large Language Models / E5-Mistral (Wang et al., 2023)](https://arxiv.org/abs/2401.00368)
- [MMTEB: Massive Multilingual Text Embedding Benchmark (Enevoldsen et al., 2025)](https://arxiv.org/abs/2502.13595)

### 실무 적용
다국어 또는 한국어+영어 혼용 코퍼스를 다루는 RAG라면 언어별 임베딩 모델을 따로 두는 구성 대신 통합 모델 하나로 인덱스를 단순화할 수 있다. 더 실무적인 시사점은 데이터 쪽이다 — 사내 문서로 임베딩을 파인튜닝할 때 LLM으로 문서마다 가상 질문을 생성하고, 유사하지만 오답인 문단을 hard negative로 뽑는 파이프라인은 소규모 팀도 그대로 재현할 수 있으며, 모델 교체보다 효과가 큰 경우가 많다.

---

## 추천 읽기 순서

1. **ORQA (2019)** — 왜 학습된 검색기가 필요한가. RAG의 문제 정의를 세운다.
2. **Gemini Embedding (2025)** — 그 질문에 대한 2025년의 답. 6년 사이 무엇이 바뀌었는지 대조하며 읽는다.
3. **MRL (2022)** — 성능이 아니라 비용의 관점. 앞의 두 편을 실제 서비스에 올릴 때 부딪히는 제약을 다룬다.

시간이 없다면 MRL → Gemini Embedding 순으로 읽고, ORQA는 ICT 섹션(3장)만 발췌해도 충분하다.

## 핵심 테이크어웨이

- **검색기는 부품이 아니라 학습 대상이다.** ORQA 이후 dense retrieval의 성능 향상 대부분은 모델 구조가 아니라 학습 신호(ICT → in-batch negative → LLM 생성 hard negative)의 개선에서 나왔다.
- **임베딩 차원은 이제 하이퍼파라미터가 아니라 런타임 설정이다.** MRL 덕분에 정확도-비용 트레이드오프를 재학습 없이 조절할 수 있고, 상용 임베딩 API의 차원 옵션은 대부분 이 기법에 기반한다.
- **임베딩 모델의 백본이 BERT급에서 LLM급으로 이동했다.** 다만 성능 격차의 상당 부분은 백본 크기가 아니라 데이터 파이프라인(합성 쿼리, 필터링, hard negative)에서 발생한다 — 이는 소규모 팀이 모방할 수 있는 부분이다.
- **벤치마크 점수와 우리 도메인 성능은 다르다.** MMTEB 1위 모델이 사내 문서에서도 1위라는 보장은 없다. 자체 평가셋 구축이 모델 선택보다 우선한다.

## 다음 토픽과의 연결

다음 토픽은 **RAG Architecture and Optimization**(Lewis et al. RAG, REALM)이다. 오늘 다룬 세 편은 모두 "무엇을 가져올 것인가"에 집중했다면, 다음은 "가져온 것을 생성 모델과 어떻게 결합할 것인가"의 문제다. 특히 REALM은 오늘 읽은 ORQA의 ICT 아이디어를 사전학습 단계 전체로 확장한 논문이므로, ORQA를 먼저 읽어 두면 자연스럽게 이어진다. MRL에서 본 차원-비용 트레이드오프는 이후 **Vector Databases and Indexing** 토픽의 HNSW·양자화 논의와 직접 연결된다.
