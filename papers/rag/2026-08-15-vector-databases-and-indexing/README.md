# Daily AI Paper Recommendations

> **Date:** 2026-08-15
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Vector Databases and Indexing

---

## Paper 1 (Classic): Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph (NSG)
- **Authors:** Cong Fu, Chao Xiang, Changxu Wang, Deng Cai
- **Year:** 2017 (VLDB 2019)
- **arXiv:** https://arxiv.org/abs/1707.00143
- **PDF:** [./nsg-navigating-spreading-out-graph-fu-2017.pdf](./nsg-navigating-spreading-out-graph-fu-2017.pdf)
- **Citation Count:** 900+ (approximate)

### 요약
그래프 기반 ANN(근사 최근접 이웃) 탐색에서 이상적인 그래프인 MRNG(Monotonic Relative Neighborhood Graph)를 정의하고, 이를 실용적으로 근사한 NSG(Navigating Spreading-out Graph)를 제안한 논문이다. 그래프의 간선 수를 줄이면서도 "단조 탐색 경로(monotonic search path)"를 보장해, 탐색 시 불필요한 거리 계산을 크게 줄인다. 1억 규모 데이터셋에서 당시 SOTA 대비 인덱스 메모리와 탐색 지연을 동시에 개선했다.

### 핵심 기여
- 그래프 인덱스가 만족해야 할 4가지 요건(연결성, 낮은 평균 차수, 짧은 탐색 경로, 낮은 인덱스 크기)을 명시적으로 정리
- MRNG라는 이론적 이상 그래프를 정의하고, 단조 경로 존재성을 증명
- Navigating Node(진입점 고정) + 근사 MRNG 간선 선택으로 O(n log n)에 가까운 실용적 인덱스 구축 알고리즘 제시
- Taobao 전자상거래 검색 시스템에 실제 배포된 사례 보고

### 이 논문이 중요한 이유
오늘날 벡터 DB 대부분(Milvus, Weaviate, Qdrant, Faiss의 NSG/NSSG 인덱스)이 "그래프 기반 인덱스"를 기본으로 삼는다. HNSW가 계층 구조로 문제를 풀었다면 NSG는 "간선을 얼마나 잘 고르느냐"로 풀었고, 이 간선 프루닝(pruning) 아이디어는 DiskANN·Vamana 등 후속 인덱스의 직접적 기반이 되었다. AI 엔지니어가 인덱스 파라미터(R, L, C)를 튜닝할 때 무엇을 조절하고 있는지 이해하려면 이 논문의 프레임이 필요하다.

### 사전 지식
- 최근접 이웃 탐색(NN search)과 근사 탐색의 recall@k 개념
- 그래프 탐색(greedy best-first search)과 kNN 그래프
- Delaunay 그래프, RNG(Relative Neighborhood Graph) 같은 근접성 그래프의 기본 개념
- 고차원 공간의 차원의 저주(curse of dimensionality)

### 관련 논문
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016)](https://arxiv.org/abs/1603.09320)
- [DiskANN / Vamana: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (Subramanya et al., 2019)](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [High Dimensional Similarity Search with Satellite System Graph / NSSG (Fu et al., 2019)](https://arxiv.org/abs/1907.06146)

### 실무 적용
RAG 파이프라인에서 문서 청크 수가 수백만 건을 넘어가면 flat 검색은 비용이 감당되지 않는다. 이때 NSG 계열 그래프 인덱스를 쓰면 recall 95% 수준을 유지하면서 지연을 수십 배 줄일 수 있다. 실무에서는 "인덱스 빌드 시간 vs 쿼리 지연 vs 메모리" 3자 트레이드오프를 조율해야 하는데, NSG는 HNSW보다 인덱스가 작아 메모리 제약이 큰 서빙 환경(예: 단일 노드에 여러 테넌트의 인덱스를 올리는 B2B SaaS)에서 특히 유리하다.

---

## Paper 2 (Classic): Practical and Optimal LSH for Angular Distance
- **Authors:** Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig Schmidt
- **Year:** 2015 (NeurIPS 2015)
- **arXiv:** https://arxiv.org/abs/1509.02897
- **PDF:** [./practical-optimal-lsh-angular-distance-andoni-2015.pdf](./practical-optimal-lsh-angular-distance-andoni-2015.pdf)
- **Citation Count:** 700+ (approximate)

### 요약
코사인 유사도(각도 거리) 기준의 LSH(Locality-Sensitive Hashing)에서 이론적으로 최적인 cross-polytope LSH가 실제로도 빠르게 동작함을 보인 논문이다. 저자들은 pseudo-random rotation(Hadamard 변환 기반)으로 해시 계산 비용을 낮추고, multiprobe 기법을 결합해 메모리 사용량을 줄였다. 그 결과 이론적 최적성과 실용성을 동시에 달성한 FALCONN 라이브러리의 토대가 되었다.

### 핵심 기여
- 각도 거리에 대해 최적 지수(optimal exponent) ρ를 달성하는 cross-polytope LSH의 실용적 구현 제시
- Fast Hadamard Transform 기반 유사 랜덤 회전으로 해시 계산을 O(d log d)로 축소
- Multiprobe LSH를 cross-polytope에 적용해 필요한 해시 테이블 수를 대폭 절감
- 하이퍼플레인 LSH 대비 실측 성능 우위를 벤치마크로 입증 (FALCONN으로 오픈소스화)

### 이 논문이 중요한 이유
현대 임베딩 검색은 사실상 대부분 코사인 유사도 기반이다. 그래프 인덱스가 주류가 된 지금도 LSH는 (1) 이론적 보장이 필요한 상황, (2) 스트리밍/온라인 업데이트가 빈번한 상황, (3) 중복 제거·near-duplicate detection 같은 작업에서 여전히 쓰인다. 무엇보다 "왜 근사 검색이 정확도를 포기하고도 안전한가"에 대한 확률적 근거를 제공하는 것이 LSH 계열이며, AI 엔지니어가 recall 목표를 수치로 설계하려면 이 사고 틀이 필요하다.

### 사전 지식
- 코사인 유사도와 L2 정규화된 벡터에서 각도 거리 ≈ 유클리드 거리라는 관계
- LSH의 기본 정의: (r1, r2, p1, p2)-sensitive 해시 패밀리와 ρ = log(1/p1)/log(1/p2)
- SimHash / 하이퍼플레인 LSH의 동작 방식
- 기초 확률론과 고차원 구면 위 균등 분포에 대한 직관

### 관련 논문
- [Similarity Search in High Dimensions via Hashing (Gionis, Indyk, Motwani, 1999)](https://www.vldb.org/conf/1999/P49.pdf)
- [Optimal Data-Dependent Hashing for Approximate Near Neighbors (Andoni & Razenshteyn, 2015)](https://arxiv.org/abs/1501.01062)
- [Similarity Estimation Techniques from Rounding Algorithms / SimHash (Charikar, 2002)](https://dl.acm.org/doi/10.1145/509907.509965)

### 실무 적용
대규모 콘텐츠 플랫폼에서 중복 문서 제거, 유사 이미지·오디오 클러스터링, 실시간 추천 후보군 생성 등에 LSH가 직접 투입된다. RAG 관점에서는 인덱싱 파이프라인 전처리 단계에서 near-duplicate 청크를 제거해 컨텍스트 낭비를 줄이는 데 유용하다. 또한 데이터가 초당 수천 건씩 삽입/삭제되는 워크로드에서는 그래프 인덱스 재구축 비용이 크기 때문에, LSH 기반 인덱스가 운영상 더 현실적인 선택지가 되기도 한다.

---

## Paper 3 (Recent): Survey of Filtered Approximate Nearest Neighbor Search over the Vector-Scalar Hybrid Data
- **Authors:** Yanjun Lin, Kai Zhang, Zhenying He, Yinan Jing, X. Sean Wang
- **Year:** 2025 (submitted to The VLDB Journal)
- **arXiv:** https://arxiv.org/abs/2505.06501
- **PDF:** [./filtered-ann-search-survey-lin-2025.pdf](./filtered-ann-search-survey-lin-2025.pdf)
- **Citation Count:** 초기 인용 단계 (2025년 5월 공개)

### 요약
벡터와 스칼라 속성(메타데이터)이 함께 있는 데이터에서 필터 조건을 만족하는 최근접 이웃을 찾는 FANNS(Filtered ANNS) 문제를 최초로 체계적으로 정리한 서베이다. 저자들은 하이브리드 데이터셋·하이브리드 쿼리·평가 지표를 형식적으로 정의하고, 기존 알고리즘을 "프루닝 방식" 중심의 분류 체계로 재정리했다. 나아가 데이터와 쿼리의 분포 관계 관점에서 쿼리 난이도를 분석하고, 실무자를 위한 선택 가이드를 제시한다.

### 핵심 기여
- FANNS 문제의 정의가 논문마다 달랐던 문제를 해소하는 형식적 정의와 평가 지표 정립
- pre-filtering / post-filtering / in-filtering(hybrid)을 아우르는 프루닝 중심의 세분화된 분류 프레임워크 제안
- 대표적 하이브리드 데이터셋을 정리하고, selectivity(선택도)와 속성-벡터 상관관계에 따른 쿼리 난이도 분석
- 데이터셋 다운로드 및 난이도 분석 코드 공개 (https://github.com/lyj-fdu/FANNS)

### 이 논문이 중요한 이유
실무 RAG는 "순수 벡터 검색"으로 끝나지 않는다. 거의 항상 `tenant_id = X AND created_at > Y AND doc_type IN (...)` 같은 필터가 붙는다. 그런데 필터 선택도가 낮아지면 그래프 인덱스의 recall이 급격히 무너지는데, 이 현상이 왜 발생하는지 대부분의 팀은 경험적으로만 안다. 이 서베이는 그 실패 모드를 분포 관점에서 설명하고 어떤 전략을 언제 써야 하는지 판단 기준을 준다. 2024~2025년 벡터 DB 경쟁의 핵심 전장이 바로 이 필터드 검색이다.

### 사전 지식
- HNSW, IVF, DiskANN 등 기본 ANN 인덱스 구조
- pre-filtering vs post-filtering의 개념과 각각의 recall/지연 트레이드오프
- 쿼리 선택도(selectivity) 개념과 DB 옵티마이저의 기본 사고방식
- recall@k, QPS, 인덱스 빌드 비용 등 ANN 벤치마크 지표

### 관련 논문
- [Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters (Gollapudi et al., 2023)](https://dl.acm.org/doi/10.1145/3543507.3583552)
- [ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data (Patel et al., 2024)](https://arxiv.org/abs/2403.04871)
- [Survey of Vector Database Management Systems (Pan, Wang, Li, 2023)](https://arxiv.org/abs/2310.14021)
- [Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based ANN Search (Zhang et al., 2024)](https://arxiv.org/abs/2410.20381)

### 실무 적용
멀티테넌트 B2B SaaS에서 RAG를 운영한다면 모든 쿼리에 테넌트 필터가 붙고, 테넌트별 문서 비중이 작으면 선택도가 0.1% 이하로 떨어진다. 이때 post-filtering은 recall이 붕괴하고 pre-filtering은 brute-force에 가까워진다. 실무적으로는 (1) 테넌트별 인덱스 분리, (2) Filtered-DiskANN/ACORN 계열 predicate-aware 인덱스 도입, (3) 선택도에 따라 전략을 동적으로 전환하는 라우팅 중 하나를 택하게 되는데, 이 서베이의 난이도 분석이 그 의사결정의 근거 자료가 된다.

---

## 추천 읽기 순서

1. **NSG (Fu et al., 2017)** — 먼저 그래프 인덱스가 왜 지금의 표준이 되었는지 그 원리를 잡는다. 4가지 요건과 간선 프루닝만 이해해도 HNSW·DiskANN 파라미터가 전부 읽힌다.
2. **Practical and Optimal LSH (Andoni et al., 2015)** — 다음으로 그래프와 대비되는 해싱 계열의 사고방식을 본다. 확률적 보장이라는 다른 축을 알면 인덱스 선택 판단이 입체적으로 바뀐다.
3. **FANNS Survey (Lin et al., 2025)** — 마지막으로 두 접근이 "필터가 붙는 현실 워크로드"에서 어떻게 깨지는지 확인한다. 앞의 두 논문을 읽은 뒤에 봐야 분류 체계가 몸에 들어온다.

이론 수식이 부담스럽다면 NSG는 Section 3(그래프 요건)과 Section 5(실험), LSH는 Section 1~2와 실험 파트만 먼저 읽어도 실무 의사결정에는 충분하다.

## 핵심 테이크어웨이

- **벡터 인덱스는 "정확도를 얼마나 싸게 포기할 것인가"의 공학이다.** recall 100%를 목표로 하는 순간 비용이 지수적으로 튄다. 제품 요구사항에서 허용 recall을 먼저 정의하는 것이 인덱스 선택보다 앞선다.
- **그래프 계열(NSG/HNSW/DiskANN)과 해싱 계열(LSH)은 경쟁이 아니라 다른 제약 조건에 대한 답이다.** 정적·읽기 중심이면 그래프, 갱신 빈번·이론적 보장 필요면 해싱이 유리하다.
- **간선 프루닝이 그래프 인덱스 성능의 본질이다.** NSG의 MRNG 근사, DiskANN의 Vamana 프루닝 모두 "긴 간선 하나가 짧은 간선 여러 개보다 낫다"는 같은 통찰을 공유한다.
- **필터드 검색(FANNS)은 벤치마크와 프로덕션의 격차가 가장 큰 지점이다.** ann-benchmarks에서 잘 나오던 인덱스가 테넌트 필터 하나 붙는 순간 recall이 무너질 수 있다. 반드시 자사 워크로드의 선택도 분포로 재측정해야 한다.
- **선택도(selectivity)를 관측 지표로 만들어라.** 쿼리 로그에서 필터 선택도 분포를 대시보드화하면, 인덱스 전략 변경 시점을 데이터로 판단할 수 있다.

## 다음 토픽과의 연결

오늘로 Module 9(RAG) 사이클이 마무리된다. 벡터 DB와 인덱싱은 RAG 스택의 최하단 인프라 레이어로, 임베딩 품질(Day 24)·검색 아키텍처(Day 25)·자기 교정 검색(Day 26)에서 논의한 모든 상위 전략이 결국 이 인덱스의 recall과 지연 위에 얹힌다. 다음 사이클은 Module 3(Classical ML Algorithms)로 돌아가는데, 트리 앙상블과 벡터 인덱스는 "고차원 공간을 어떻게 분할할 것인가"라는 동일한 질문을 각각 지도학습과 유사도 검색의 관점에서 다룬다는 점에서 이어서 보면 흥미롭다.
