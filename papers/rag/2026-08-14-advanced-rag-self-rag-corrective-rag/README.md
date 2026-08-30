# Daily AI Paper Recommendations

> **Date:** 2026-08-14
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG — Self-RAG, Corrective RAG

---

## Paper 1 (Classic): REPLUG: Retrieval-Augmented Black-Box Language Models
- **Authors:** Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2301.12652
- **PDF:** [./replug-shi-2023.pdf](./replug-shi-2023.pdf)
- **Citation Count:** approx. 900+ (Google Scholar 기준)

### 요약
REPLUG은 LLM 내부를 전혀 건드리지 않고(블랙박스로 취급) 외부 검색기(retriever)만 학습시켜 성능을 끌어올리는 검색증강 프레임워크다. 검색된 문서를 입력 앞에 단순히 이어붙이는(prepend) 방식이라 GPT-3 같은 API 전용 모델에도 곧바로 적용할 수 있다. 핵심은 "LM이 더 정확한 예측을 하도록 돕는 문서"가 무엇인지를 LM 자신의 예측 신호로 역으로 학습(REPLUG LSR)하여 검색기를 최적화한다는 점이다.

### 핵심 기여
- LM을 재학습하지 않는 블랙박스 전제에서 검색기만 튜닝하는 실용적 아키텍처를 제시
- 여러 문서를 각각 처리한 뒤 출력 확률을 앙상블(ensemble)하여 컨텍스트 길이 제약을 우회
- LM의 다음 토큰 확률을 지도 신호로 삼아 검색기를 학습하는 REPLUG LSR(LM-Supervised Retrieval) 기법

### 이 논문이 중요한 이유
상용 API 기반 LLM(폐쇄형 모델)이 지배적인 현실에서, 모델 가중치에 접근하지 못해도 검색 파이프라인만으로 정확도와 사실성을 높일 수 있음을 보여준다. 오늘날 대부분의 프로덕션 RAG가 "프리징된 LLM + 튜닝 가능한 리트리버" 구조를 따르는데, REPLUG는 그 정당성과 학습 방법을 이론적·실험적으로 뒷받침한 기준점이다.

### 사전 지식
- Dense retrieval과 임베딩 유사도 검색(DPR 계열)의 기본 개념
- 언어모델의 next-token 확률 분포와 perplexity
- 앙상블(확률 평균)과 KL divergence 기반 학습 손실

### 관련 논문
- [In-Context Retrieval-Augmented Language Models (Ram et al., 2023)](https://arxiv.org/abs/2302.00083)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

### 실무 적용
GPT-4o, Claude 등 가중치에 접근할 수 없는 폐쇄형 모델을 쓰는 서비스에서, 도메인 특화 리트리버만 별도로 파인튜닝해 답변 품질을 올리는 전형적 패턴이 REPLUG의 직계 후손이다. 문서별 확률 앙상블은 긴 컨텍스트를 못 넣을 때 top-k 문서를 나눠 병렬 추론하고 결과를 합치는 실전 최적화로 이어진다.

---

## Paper 2 (Classic): In-Context Retrieval-Augmented Language Models
- **Authors:** Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2302.00083
- **PDF:** [./in-context-ralm-ram-2023.pdf](./in-context-ralm-ram-2023.pdf)
- **Citation Count:** approx. 700+ (Google Scholar 기준)

### 요약
In-Context RALM은 LM 아키텍처와 파라미터를 그대로 둔 채, 생성 시점에 관련 문서를 입력 컨텍스트 앞에 붙이기만 해도 언어모델링 성능이 크게 향상됨을 보인 연구다. 값비싼 재학습 없이도 RAG의 이득 대부분을 얻을 수 있으며, 어떤 문서를 언제 얼마나 자주 검색해 넣을지(검색 주기·재랭킹)가 성능을 좌우함을 체계적으로 분석했다.

### 핵심 기여
- 아무 학습 없이 문서를 프리펜드하는 것만으로 RALM 이득의 상당 부분을 확보할 수 있음을 실증
- 생성 도중 일정 토큰 간격마다 재검색하는 전략(retrieval stride)이 품질을 크게 좌우함을 규명
- 범용/제로샷 재랭커로 검색 결과를 다시 정렬하면 추가 이득이 있음을 제시

### 이 논문이 중요한 이유
"RAG는 곧 프롬프트에 문서를 넣는 것"이라는 오늘날의 상식적 구현이 왜 잘 동작하는지를 처음으로 깔끔하게 정리했다. 또한 한 번 검색하고 끝내는 단순 RAG의 한계와, 생성 중 주기적 재검색의 필요성을 보여줌으로써 이후 FLARE·Self-RAG 같은 적응형/반복형 RAG로 가는 다리를 놓았다.

### 사전 지식
- 언어모델의 컨텍스트 윈도우와 프롬프트 구성 방식
- BM25 등 희소 검색과 dense 검색의 차이
- 재랭킹(re-ranking)의 개념

### 관련 논문
- [REPLUG: Retrieval-Augmented Black-Box Language Models (Shi et al., 2023)](https://arxiv.org/abs/2301.12652)
- [Active Retrieval Augmented Generation / FLARE (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)

### 실무 적용
LangChain·LlamaIndex의 기본 RAG 체인이 바로 이 "프리펜드 + 재랭킹" 구조다. 긴 생성 태스크(리포트 작성 등)에서 중간중간 다시 검색해 컨텍스트를 갱신하는 iterative RAG 파이프라인 설계의 근거를 제공한다.

---

## Paper 3 (Recent): Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity
- **Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
- **Year:** 2024 (NAACL 2024)
- **arXiv:** https://arxiv.org/abs/2403.14403
- **PDF:** [./adaptive-rag-jeong-2024.pdf](./adaptive-rag-jeong-2024.pdf)
- **Citation Count:** approx. 300+ (Google Scholar 기준)

### 요약
Adaptive-RAG는 질문의 복잡도에 따라 RAG 전략 자체를 동적으로 바꾸는 프레임워크다. 간단한 질문은 검색 없이(no-retrieval), 중간 난도는 단일 검색(single-step), 복잡한 다단계 질문은 반복 검색(multi-step)으로 처리한다. 작은 분류기가 질문 복잡도를 예측해 최적 경로를 라우팅함으로써, 정확도와 지연시간·비용 사이의 균형을 자동으로 맞춘다.

### 핵심 기여
- 질문 복잡도를 3단계로 예측하는 경량 분류기 기반 라우팅 메커니즘 제안
- 단순 질문의 불필요한 검색 오버헤드와 복잡 질문의 검색 부족 문제를 동시에 해결
- 정확도를 유지하면서 평균 검색 횟수·응답 지연을 크게 줄이는 효율성 입증
