# Daily AI Paper Recommendations

> **Date:** 2026-10-08
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Vector Databases and Indexing

> 이번 사이클에서는 이전에 다룬 FAISS·HNSW·ScaNN·DiskANN·PQ·SPANN·NSG·LSH(Andoni)·ANN-Benchmarks·Manu와 겹치지 않도록, **LSH의 원조 논문**과 **그래프 기반 ANN 종합 서베이**를 고전으로, **필터 결합 하이브리드 검색(ACORN)**을 최신 논문으로 골랐습니다.

---

## Paper 1 (Classic): Similarity Search in High Dimensions via Hashing
- **Authors:** Aristides Gionis, Piotr Indyk, Rajeev Motwani
- **Year:** 1999 (VLDB '99, pp. 518–529)
- **URL:** https://vldb.org/dblp/db/conf/vldb/GionisIM99.html (arXiv 없음)
- **PDF:** [./lsh-similarity-search-high-dimensions-gionis-1999.pdf](./lsh-similarity-search-high-dimensions-gionis-1999.pdf) (Princeton 강의 아카이브 사본)
- **Citation Count:** ~5,000+

### 요약
고차원에서 정확한 최근접 이웃 검색은 "차원의 저주" 때문에 사실상 선형 탐색과 다를 바 없어집니다. 이 논문은 가까운 점들이 같은 버킷에 충돌할 확률이 높도록 설계된 해시 함수(Locality-Sensitive Hashing)를 여러 개 사용해, 정확도를 약간 포기하는 대신 쿼리 시간을 데이터 크기에 대해 준선형(sublinear)으로 줄이는 실용적인 방법을 제시합니다. 최대 64차원의 색상 히스토그램·텍스처 데이터에서 SR-tree 같은 트리 인덱스보다 훨씬 빠름을 실험으로 보였습니다.

### 핵심 기여
- Indyk–Motwani(1998)의 이론적 LSH를 Hamming 공간 임베딩 기반의 **실제 구현 가능한 알고리즘**으로 정리
- 해시 테이블 수(L)와 해시 길이(k)로 **정확도–속도 트레이드오프를 조절**하는 프레임워크 제시
- 디스크(외부 메모리) 환경으로의 확장과 트리 기반 인덱스 대비 실증 비교

### 이 논문이 중요한 이유
"근사(approximate)를 받아들이면 고차원 검색이 가능해진다"는 ANN 분야의 출발점입니다. 이후 PQ, 그래프 인덱스, 학습 기반 해싱 모두 이 문제 정의 위에 서 있습니다. 벡터 DB의 recall–latency 튜닝 감각을 이해하려면 반드시 읽어야 할 원류입니다.

### 사전 지식
k-NN 기본 개념, L1/L2·Hamming 거리, 확률 기초(충돌 확률), 해시 테이블, kd-tree 등 공간 분할 인덱스의 한계

### 관련 논문
- [Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality (Indyk & Motwani, 1998)](https://doi.org/10.1145/276698.276876)
- [Practical and Optimal LSH for Angular Distance (Andoni et al., 2015)](https://arxiv.org/abs/1509.02897)
- [Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search (Lv et al., 2007)](https://www.vldb.org/conf/2007/papers/research/p950-lv.pdf)

### 실무 적용
현재 대규모 RAG에서는 HNSW·IVF-PQ가 주류지만, LSH는 **중복 문서 제거(MinHash/SimHash)**, 학습 데이터 near-dedup, 스트리밍 환경의 빠른 후보 필터링에 여전히 널리 쓰입니다. 데이터셋 정제 파이프라인에서 특히 유용합니다.

---

## Paper 2 (Classic): A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search
- **Authors:** Mengzhao Wang, Xiaoliang Xu, Qiang Yue, Yuxiang Wang
- **Year:** 2021 (PVLDB 14)
- **arXiv:** https://arxiv.org/abs/2101.12631
- **PDF:** [./graph-based-ann-survey-wang-2021.pdf](./graph-based-ann-survey-wang-2021.pdf)
- **Citation Count:** ~400+

### 요약
HNSW, NSG, NSW, KGraph, DPG, Vamana 등 13개 대표 그래프 기반 ANN 알고리즘을 새로운 분류 체계와 "세분화된 파이프라인"(초기화 → 후보 이웃 획득 → 이웃 선택 → 진입점 → 연결성 보장 → 라우팅)으로 해체해 비교합니다. 8개 실데이터·12개 합성 데이터에서 동일 환경으로 실험하고, 컴포넌트를 조합한 최적화 알고리즘과 실무자용 선택 가이드를 제시합니다.

### 핵심 기여
- 그래프 ANN을 **4가지 기반 그래프(DG, RNG, KNNG, MST)** 관점으로 정리한 분류 체계
- 알고리즘을 컴포넌트 단위로 분해해 **"어떤 설계 요소가 성능을 좌우하는가"**를 실험으로 규명
- 데이터 특성(규모, 차원, LID)별 알고리즘 추천 규칙(rule of thumb) 제시

### 이 논문이 중요한 이유
벡터 DB 대부분이 그래프 인덱스를 기본으로 쓰는데, 개별 논문만 읽으면 "왜 HNSW가 이 데이터에서 느린가"를 설명하기 어렵습니다. 이 서베이는 인덱스 파라미터(M, efConstruction, 이웃 선택 휴리스틱)의 의미를 구조적으로 이해하게 해 주는 지도 역할을 합니다.

### 사전 지식
HNSW·NSG 기본 원리, 그래프 탐색(greedy/beam search), recall@k와 QPS 지표, 고유 차원(LID) 개념

### 관련 논문
- [Efficient and Robust ANN Search using HNSW Graphs (Malkov & Yashunin, 2016)](https://arxiv.org/abs/1603.09320)
- [Fast ANN Search With The Navigating Spreading-out Graph (Fu et al., 2017)](https://arxiv.org/abs/1707.00143)
- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (Subramanya et al., 2019)](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)

### 실무 적용
pgvector, Milvus, Qdrant, Weaviate에서 HNSW 파라미터를 튜닝하거나 DiskANN 계열로 전환할지 판단할 때 근거가 됩니다. 임베딩 모델 교체로 데이터 분포(LID)가 바뀌면 인덱스 성능도 달라진다는 점을 이해하고 벤치마크 설계에 반영할 수 있습니다.

---

## Paper 3 (Recent): ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data
- **Authors:** Liana Patel, Peter Kraft, Carlos Guestrin, Matei Zaharia
- **Year:** 2024 (ACM SIGMOD 2024, DOI 10.1145/3654923)
- **arXiv:** https://arxiv.org/abs/2403.04871
- **PDF:** [./acorn-predicate-agnostic-hybrid-search-patel-2024.pdf](./acorn-predicate-agnostic-hybrid-search-patel-2024.pdf)
- **Citation Count:** ~100+

### 요약
"2024년 이후 작성된, 카테고리가 '정책'인 문서 중 질문과 가장 유사한 것"처럼 벡터 유사도와 구조화된 필터(predicate)를 함께 거는 하이브리드 검색은 기존 pre-filter/post-filter 방식으로는 느리거나 recall이 무너집니다. ACORN은 HNSW를 확장해, 필터를 만족하는 노드들만으로 이루어진 **predicate subgraph**를 탐색 시점에 근사적으로 순회하는 방식을 제안합니다. 인덱스를 필터 종류와 무관하게(predicate-agnostic) 구축하면서도 기존 방법 대비 0.9 recall에서 2~1,000배 높은 QPS를 보였습니다.

### 핵심 기여
- 이상적인 "필터별 전용 인덱스"를 근사하는 **predicate subgraph traversal** 탐색 전략
- 이웃 리스트를 확장(γ배)·압축하는 **predicate-agnostic 인덱스 구축** — 범위, 정규식, 임의 조건 필터 지원
- 기존 HNSW 라이브러리 위에 구현 가능한 단순한 설계 (SIFT1M, LAION, TripClick 등에서 검증)

### 이 논문이 중요한 이유
실제 RAG 서비스의 검색은 거의 항상 "권한·테넌트·날짜·문서 유형" 필터와 함께 일어납니다. 필터 선택도(selectivity)가 낮을 때 HNSW가 왜 망가지는지, 그리고 이를 인덱스 수준에서 어떻게 해결하는지 보여 주는 대표 논문으로, 2025년 이후 Weaviate 등 벡터 DB의 필터 검색 구현에도 영향을 주었습니다.

### 사전 지식
HNSW 구조와 탐색, pre-filtering vs post-filtering의 차이, 필터 선택도(selectivity) 개념, 메타데이터 필터링 기반 RAG

### 관련 논문
- [Filtered-DiskANN: Graph Algorithms for ANN Search with Filters (Gollapudi et al., 2023)](https://doi.org/10.1145/3543507.3583552)
- [Survey of Filtered Approximate Nearest Neighbor Search over the Vector-Scalar Hybrid Data (Lin et al., 2025)](https://arxiv.org/abs/2505.06501)
- [Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based ANN Search (Zhang et al., 2024)](https://arxiv.org/abs/2410.20381)

### 실무 적용
멀티테넌트 B2B SaaS에서 "고객사별 문서만 검색" 같은 권한 필터를 걸 때, 테넌트별 인덱스를 따로 만들지(비용↑) 단일 인덱스에 필터를 걸지(recall↓) 고민하게 됩니다. ACORN 방식은 단일 인덱스로 두 문제를 동시에 줄이는 선택지를 주며, 벡터 DB 선정 시 "저선택도 필터에서의 recall/QPS"를 평가 항목에 넣어야 하는 이유를 설명해 줍니다.

---

## 추천 읽기 순서
1. **Gionis et al. (1999)** — "근사 검색"이라는 문제 정의와 정확도–속도 트레이드오프의 원리부터 잡습니다.
2. **Wang et al. (2021)** — 오늘날 주류인 그래프 인덱스를 컴포넌트 단위로 이해합니다. 섹션 3(분류)과 섹션 5(컴포넌트 평가)를 우선 읽으세요.
3. **Patel et al. (2024, ACORN)** — 그래프 인덱스 위에 실무의 필터 조건이 얹혔을 때의 문제와 해법을 봅니다.

## 핵심 테이크어웨이
- 고차원 벡터 검색은 **"정확도를 조금 포기하고 속도를 얻는"** 근사 문제이며, 모든 인덱스는 이 트레이드오프를 조절하는 다이얼을 가진다.
- 그래프 인덱스 성능은 알고리즘 이름보다 **이웃 선택·진입점·연결성 같은 설계 요소와 데이터 특성(LID)**에 더 크게 좌우된다.
- 실무 RAG의 병목은 순수 ANN이 아니라 **메타데이터 필터와 결합된 하이브리드 검색**인 경우가 많다 — 벡터 DB를 평가할 때 필터 선택도별 recall을 반드시 측정하자.
- 질문해 볼 것: 우리 서비스의 검색 쿼리 중 필터가 걸리는 비율은? 그 필터의 평균 선택도는? 현재 인덱스는 저선택도에서 recall을 유지하는가?

## 다음 토픽과의 연결
이번 날짜로 Module 9(RAG)의 한 사이클이 마무리되고, 다음 날부터 다시 **Module 3: Classical ML Algorithms and Foundations**로 돌아갑니다. LSH의 확률적 근사와 그래프 탐색의 아이디어는 랜덤 포레스트·부스팅의 "약한 학습기 앙상블" 직관과 함께, **ML 전반의 근사·앙상블 사고**를 다시 짚어 보는 좋은 연결 고리가 됩니다.
