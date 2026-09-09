# Daily AI Paper Recommendations

> **Date:** 2026-09-10
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG — Self-RAG, Corrective RAG

---

## Paper 1 (Classic): Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models
- **Authors:** Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin Ma, Hongwei Wang, Dong Yu
- **Year:** 2023 (EMNLP 2024)
- **arXiv:** https://arxiv.org/abs/2311.09210
- **PDF:** [./chain-of-note-yu-2023.pdf](./chain-of-note-yu-2023.pdf)
- **Citation Count:** 약 400+

### 요약
RAG는 검색된 문서가 정확할 때만 잘 동작하며, 노이즈가 섞인 문서가 들어오면 모델이 자기 내부 지식조차 무시하고 잘못된 답을 생성한다. Chain-of-Note(CoN)는 답변을 바로 생성하지 않고, 검색된 문서 각각에 대해 "읽기 노트(reading note)"를 순차적으로 작성한 뒤 그 노트를 근거로 최종 답을 만드는 방식을 제안한다. 이 중간 단계가 문서의 관련성을 명시적으로 평가하게 만들어, 노이즈 문서 환경에서 EM +7.9, 지식 범위 밖 질문에 대한 거절률 +10.5를 달성했다.

### 핵심 기여
- 검색 결과와 최종 답변 사이에 "노트 생성"이라는 명시적 중간 추론 단계를 삽입한 프레임워크 제안
- 문서 유형을 (1) 직접 답 포함 (2) 간접 단서 제공 (3) 무관 의 3가지로 구분해 노트를 작성하도록 학습시키는 데이터 생성 방식 설계
- "모르면 모른다고 답하기(rejection)"를 RAG 파이프라인 안에서 정량적으로 개선 — 환각 억제의 실증적 근거 제시

### 이 논문이 중요한 이유
실무 RAG의 실패는 대부분 "검색기가 못 찾아서"가 아니라 "애매하게 관련된 문서를 LLM이 과신해서" 발생한다. CoN은 리랭커나 임베딩을 바꾸지 않고도 생성 단계의 프롬프트/파인튜닝 설계만으로 이 문제를 완화할 수 있다는 것을 보여준다. Self-RAG의 reflection token, CRAG의 retrieval evaluator와 함께 "생성기가 검색 결과를 비판적으로 읽게 만든다"는 Advanced RAG의 핵심 계보를 이룬다.

### 사전 지식
- 기본 RAG 파이프라인 (Lewis et al., 2020)
- Chain-of-Thought 프롬프팅과 중간 추론 토큰의 역할
- Open-domain QA 평가 지표 (EM, F1)와 NQ / TriviaQA / WebQ 벤치마크

### 관련 논문
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

### 실무 적용
사내 문서 QA 봇에서 "관련은 있지만 답은 없는 문서"가 상위 랭크될 때 발생하는 그럴듯한 오답을 줄이는 데 바로 쓸 수 있다. 파인튜닝 없이도 생성 프롬프트를 "각 문서별 노트 → 종합 판단 → 답변 또는 '정보 없음'" 구조로 바꾸는 것만으로 효과를 볼 수 있고, 생성된 노트 자체가 사용자에게 보여줄 근거(citation rationale)와 내부 품질 로그로 재활용된다. 다만 노트 생성만큼 토큰과 지연시간이 늘어나므로, 검색 신뢰도가 낮은 쿼리에만 선택적으로 적용하는 설계가 현실적이다.

---

## Paper 2 (Classic): Making Retrieval-Augmented Language Models Robust to Irrelevant Context
- **Authors:** Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.01558
- **PDF:** [./ret-robust-yoran-2023.pdf](./ret-robust-yoran-2023.pdf)
- **Citation Count:** 약 500+

### 요약
"검색이 도움이 될 때는 성능을 올리고, 도움이 안 될 때는 성능을 깎지 않아야 한다"는 RAG의 기본 요구조건이 실제로는 지켜지지 않는다는 점을 5개 open-domain QA 벤치마크에서 정량 분석한 논문이다. 저자들은 두 가지 처방을 제시한다. 첫째, NLI(자연어 추론) 모델로 "이 문서가 질문-답변 쌍을 함의(entail)하는가"를 판정해 관련 없는 문서를 필터링한다. 둘째, 학습 시 관련 문서와 무관 문서를 섞은 데이터를 자동 생성해 파인튜닝하면 단 1,000개 예시만으로도 무관한 컨텍스트에 견고해진다는 것을 보인다.

### 핵심 기여
- 검색이 오히려 정확도를 떨어뜨리는 실패 케이스를 벤치마크 단위로 체계적으로 분류 (다단계 추론 질문에서 특히 심각)
- 학습 불필요한 방어책으로 NLI 기반 사후 필터링 제안 — 리트리버 교체 없이 즉시 적용 가능
- 소량(1K)의 혼합 컨텍스트 데이터만으로 견고성을 확보하는 데이터 생성 레시피 제시. 견고성이 비용이 큰 문제가 아님을 입증

### 이 논문이 중요한 이유
Corrective RAG(CRAG)의 retrieval evaluator, Self-RAG의 critique token이 모두 "검색 결과를 신뢰할지 판단하는 별도 모듈"이라는 동일한 아이디어를 공유한다. 이 논문은 그 아이디어를 가장 단순한 형태(NLI 필터)와 가장 저렴한 형태(1K 파인튜닝)로 검증해, Advanced RAG 설계 시 "얼마나 복잡한 장치가 실제로 필요한가"에 대한 기준선을 제공한다. 새 기법을 도입하기 전 반드시 비교해야 할 베이스라인이다.

### 사전 지식
- Natural Language Inference(NLI)와 entailment 개념
- Multi-hop QA (HotpotQA, StrategyQA)와 self-ask 계열 프롬프팅
- 파인튜닝 데이터 자동 생성(synthetic data generation)의 기본 개념

### 관련 논문
- [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Measuring and Narrowing the Compositionality Gap / Self-Ask (Press et al., 2022)](https://arxiv.org/abs/2210.03350)
- [RECOMP: Improving RAG with Compression and Selective Augmentation (Xu et al., 2023)](https://arxiv.org/abs/2310.04408)

### 실무 적용
프로덕션 RAG에서 "리랭커 점수 threshold"만으로 문서를 거르면 도메인이 바뀔 때마다 임계값을 다시 튜닝해야 한다. NLI 기반 필터는 의미적 함의를 직접 검증하므로 임계값 민감도가 낮고, 작은 크로스인코더로도 돌아가 지연시간 부담이 적다. 또한 "1K 예시 파인튜닝" 결과는 자체 SLM을 RAG 생성기로 쓰는 팀에게 특히 실용적이다 — 대규모 도메인 학습 없이 소량의 하드 네거티브 혼합 데이터만으로 노이즈 내성을 확보할 수 있다는 뜻이기 때문이다.

---

## Paper 3 (Recent): AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning
- **Authors:** Jingbo Sun, Wenyue Chong, Songjun Tu, Qichao Zhang, Yaocheng Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao
- **Year:** 2026
- **arXiv:** https://arxiv.org/abs/2604.17337
- **PDF:** [./autosearch-sun-2026.pdf](./autosearch-sun-2026.pdf)
- **Citation Count:** 신규 논문 (2026-04 공개, 인용 축적 중)

### 요약
Agentic RAG는 LLM이 검색 도구를 여러 번 호출하며 문제를 푸는 구조지만, 실제로는 불필요한 검색 단계가 반복되어 비용과 지연시간이 폭증한다. 기존 해법은 검색 횟수 상한을 고정하는 것이었는데, 이는 어려운 질문에서 탐색 부족을 초래한다. AutoSearch는 각 검색 단계마다 "지금까지의 정보로 답을 내면 무엇인가"를 스스로 생성(self-answering)해 그 단계의 가치를 평가하고, 질문 난이도에 따라 달라지는 "최소 충분 검색 깊이(minimal sufficient search depth)"에 도달하면 보상하고 과잉 검색은 벌점을 주는 RL 프레임워크다.

### 핵심 기여
- 검색 깊이와 정확도의 관계를 분석해 "최소 충분 검색 깊이"라는 개념을 정의하고, 이것이 질문 복잡도와 에이전트 능력에 의해 함께 결정됨을 실증
- 중간 답변 자가생성을 단계별 보상 신호로 사용하는 RL 설계 — 별도 프로세스 보상 모델(PRM) 없이 step-level credit assignment 구현
- 과잉 검색(over-searching) 억제와 탐색 행동 안정화를 위한 보상 항 설계로 정확도-효율 트레이드오프 곡선 자체를 개선

### 이 논문이 중요한 이유
Self-RAG와 CRAG가 "언제 검색할까 / 검색 결과를 믿을까"를 다뤘다면, AutoSearch는 에이전트 시대의 후속 질문인 "언제 검색을 멈출까"를 다룬다. Agentic RAG를 실제 서비스에 올릴 때 병목은 정확도가 아니라 단위 쿼리당 토큰 비용과 P95 지연시간이며, 이 논문은 그 비용을 학습된 정책으로 제어한다. 2024년의 Adaptive-RAG(휴리스틱 라우팅)에서 2026년의 RL 기반 정책 학습으로 이어지는 흐름을 대표한다.

### 사전 지식
- ReAct 스타일 도구 사용 루프와 multi-step 검색 에이전트
- RLVR / GRPO 등 검증 가능 보상 기반 LLM 강화학습의 기본 개념
- Adaptive-RAG(Jeong et al., 2024)의 쿼리 복잡도 라우팅 아이디어
- Multi-hop QA 벤치마크 (HotpotQA, 2WikiMultiHopQA, Musique)

### 관련 논문
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity (Jeong et al., 2024)](https://arxiv.org/abs/2403.14403)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (Singh et al., 2025)](https://arxiv.org/abs/2501.09136)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

### 실무 적용
Deep Research류 기능이나 에이전트형 검색 제품을 운영할 때 가장 흔한 불만은 "쉬운 질문에도 20초씩 걸린다"이다. AutoSearch의 self-answering 평가는 RL 학습을 하지 않더라도 런타임 조기 종료(early-stopping) 휴리스틱으로 이식할 수 있다 — 매 검색 후 임시 답변을 생성해 직전 답변과 수렴했는지 보고 루프를 끊는 식이다. 또한 "최소 충분 검색 깊이" 개념은 제품 지표로 직결된다: 쿼리별 평균 검색 횟수, 과잉 검색 비율, 깊이 대비 정확도 곡선을 대시보드에 올리면 비용 최적화와 품질 회귀를 한 화면에서 관리할 수 있다.

---

## 추천 읽기 순서

1. **Yoran et al. (2310.01558)** — 먼저 "검색이 언제 해를 끼치는가"라는 문제 정의를 정확히 잡는다. 실패 사례 분석 파트만 읽어도 자기 RAG 시스템의 로그를 보는 눈이 달라진다.
2. **Chain-of-Note (2311.09210)** — 같은 문제를 생성기 쪽에서 푸는 방법. 필터링(입력 차단) vs 노트 작성(비판적 읽기)의 대비를 의식하면서 읽는다.
3. **AutoSearch (2604.17337)** — 단일 턴 RAG에서 다중 턴 에이전트로 문제가 확장됐을 때, 앞의 두 아이디어가 "단계별 자기평가"라는 형태로 어떻게 재등장하는지 확인한다.

## 핵심 테이크어웨이

- **Advanced RAG의 공통 뼈대는 '자기 평가 루프'다.** Self-RAG의 reflection token, CRAG의 evaluator, CoN의 노트, Yoran의 NLI 필터, AutoSearch의 self-answering은 모두 "생성 전에 현재 근거의 충분성을 판정한다"는 하나의 패턴을 각기 다른 지점에 배치한 변형이다.
- **저렴한 베이스라인을 먼저 검증하라.** Yoran et al.은 NLI 필터와 1K 파인튜닝만으로 상당한 견고성을 얻었다. 복잡한 멀티에이전트 RAG를 도입하기 전, 이 수준의 처방으로 문제가 해결되는지부터 측정하는 것이 PM/엔지니어 모두에게 이득이다.
- **품질 지표만으로는 부족하다.** AutoSearch가 보여주듯 에이전트형 RAG의 실질 KPI는 정확도 × 비용 × 지연시간의 결합이다. 검색 깊이, 과잉 검색 비율 같은 프로세스 지표를 처음부터 계측해야 최적화가 가능하다.
- **"모른다"고 답하는 능력은 기능이다.** CoN의 rejection rate 개선은 환각 억제의 핵심이며, 사용자 신뢰와 직결되는 제품 요구사항으로 다뤄야 한다.

## 다음 토픽과의 연결

다음은 **Vector Databases and Indexing**이다. 오늘 다룬 논문들은 모두 "검색 결과의 품질이 나쁠 수 있다"는 전제 위에서 생성 단계의 방어를 설계했다. 다음 토픽은 그 전제 자체를 개선하는 인프라 계층 — FAISS의 대규모 유사도 검색, HNSW의 근사 최근접 이웃 그래프 인덱스 — 을 다룬다. 검색 품질(인덱싱), 검색 판단(corrective/adaptive), 생성 견고성(CoN)의 세 층이 어떻게 맞물려 하나의 RAG 시스템을 이루는지 연결해서 정리해두면 좋다.
