# Daily AI Paper Recommendations

> **Date:** 2026-11-03
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG Self-RAG Corrective RAG

> 이번 사이클에서는 이전 사이클에서 다룬 Self-RAG, FLARE, CRAG, Speculative RAG, GraphRAG, HyDE, IRCoT, RAPTOR, LightRAG, Adaptive-RAG, REPLUG, In-Context RALM, Chain-of-Note, Ret-Robust, Rewrite-Retrieve-Read, RGB, Search-R1 등과 겹치지 않도록 **"검색 결과를 LLM이 어떻게 '읽는가'"**에 초점을 맞췄습니다. 흐름은 **"긴 컨텍스트의 중간 정보는 무시된다(Lost in the Middle)" → "그렇다면 넣기 전에 압축·선별하자(RECOMP)" → "추론 모델이 스스로 검색하고, 문서를 따로 정제해서 읽게 하자(Search-o1)"** 입니다.

---

## Paper 1 (Classic): Lost in the Middle: How Language Models Use Long Contexts
- **Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
- **Year:** 2023 (TACL 2024)
- **arXiv:** https://arxiv.org/abs/2307.03172
- **PDF:** [./lost-in-the-middle-liu-2023.pdf](./lost-in-the-middle-liu-2023.pdf)
- **Citation Count:** ~2,500+ (approximate)

### 요약
여러 개의 검색 문서를 넣는 멀티 문서 QA와 key-value 검색 실험을 통해, LLM이 긴 입력을 **균일하게 활용하지 못한다**는 사실을 보였습니다. 정답 문서가 컨텍스트의 **맨 앞이나 맨 뒤에 있을 때 성능이 가장 높고, 중간에 있을 때 크게 떨어지는 U자형 곡선**이 나타나며, 이는 롱컨텍스트 전용 모델에서도 마찬가지였습니다. 또한 검색 문서 수를 늘려도 일정 수준 이후에는 성능이 거의 오르지 않았습니다.

### 핵심 기여
- 정답 문서의 **위치(position)**만 바꾸는 통제 실험 설계로 "컨텍스트 활용도"를 측정하는 방법론 제시
- GPT-3.5-Turbo, Claude-1.3, MPT, LongChat 등에서 공통적으로 나타나는 **U자형 성능 곡선(primacy/recency bias)** 발견
- 오픈도메인 QA에서 retriever recall은 계속 오르지만 reader 정확도는 **k≈20 근처에서 포화**됨을 보여, "더 많이 넣기"의 한계를 정량화

### 이 논문이 중요한 이유
"컨텍스트 윈도우가 크니 다 넣으면 된다"는 직관을 깨뜨린 논문입니다. Self-RAG·CRAG 같은 Advanced RAG 기법들이 왜 **검색 결과를 걸러내고, 재정렬하고, 압축**하는지에 대한 근본적인 실험적 근거를 제공합니다. 2026년 현재 1M 토큰급 모델이 흔해졌지만 "Needle-in-a-Haystack을 통과한다 ≠ 여러 근거를 잘 종합한다"는 문제의식은 여전히 유효하며, 롱컨텍스트 평가 설계의 기준점이 되었습니다.

### 사전 지식
- 기본 RAG 파이프라인(retriever → top-k 문서 → LLM reader)
- Transformer의 positional encoding(RoPE, ALiBi 등)과 attention 개념
- 오픈도메인 QA 평가(NaturalQuestions, Exact Match/Accuracy)

### 관련 논문
- [Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization (Hsieh et al., 2024)](https://arxiv.org/abs/2406.16008)
- [RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654)
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (Jiang et al., 2023)](https://arxiv.org/abs/2310.06839)
- [Retrieval meets Long Context Large Language Models (Xu et al., 2023)](https://arxiv.org/abs/2310.03025)

### 실무 적용
- **리랭킹 후 배치 전략:** 가장 관련도 높은 문서를 프롬프트의 맨 앞과 맨 뒤에 두는 "양끝 배치"(LangChain `LongContextReorder`가 이 논문에서 출발)
- **top-k 튜닝:** k를 무작정 키우지 말고, 오프라인 평가에서 정확도 포화 지점을 찾아 비용·지연과 함께 최적화
- **에이전트 메모리 설계:** 긴 대화 로그를 그대로 넣기보다 요약/구조화해 핵심 정보를 앞쪽에 고정(system 영역)하는 패턴

---

## Paper 2 (Classic): RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
- **Authors:** Fangyuan Xu, Weijia Shi, Eunsol Choi
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.04408
- **PDF:** [./recomp-xu-2023.pdf](./recomp-xu-2023.pdf)
- **Citation Count:** ~400+ (approximate)

### 요약
검색된 문서를 통째로 LLM에 넣는 대신, **작은 압축기(compressor)로 먼저 요약한 뒤** 넣는 RECOMP를 제안합니다. 문장을 골라내는 **추출형(extractive) 압축기**와 여러 문서를 종합 요약하는 **생성형(abstractive) 압축기** 두 가지를 학습하며, 문서가 도움이 안 될 때는 **빈 문자열을 출력해 아예 검색 결과를 넣지 않는 "선택적 증강(selective augmentation)"**이 가능합니다. 언어모델링과 오픈도메인 QA에서 입력 토큰을 약 6% 수준까지 줄이면서도 성능 손실을 최소화했습니다.

### 핵심 기여
- 압축기의 학습 신호를 "요약 품질"이 아니라 **최종 LM의 태스크 성능(downstream perplexity/정답률)**으로 정의
- 추출형(문장 선택, dual encoder)과 생성형(T5 기반 요약, GPT-3.5 distillation) 압축기를 모두 제시하고 trade-off 비교
- 검색 결과가 무관하거나 해로울 때 **"아무것도 넣지 않는" 결정을 학습**하게 해, Self-RAG/CRAG의 "검색 결과 평가·교정" 아이디어와 연결

### 이 논문이 중요한 이유
Lost in the Middle이 "많이 넣으면 오히려 못 읽는다"는 문제를 보였다면, RECOMP는 그에 대한 **가장 직접적인 엔지니어링 해법**입니다. 압축은 비용(토큰), 지연(latency), 정확도(노이즈 제거)를 동시에 개선하는 드문 레버이며, 이후 LLMLingua, xRAG, Provence 같은 컨텍스트 압축 연구의 기준선(baseline)이 되었습니다. "검색 결과를 쓸지 말지 판단한다"는 점에서 Corrective RAG 계열과도 같은 문제의식을 공유합니다.

### 사전 지식
- 추출 요약 vs 생성 요약의 차이
- Dual encoder(Contriever, DPR) 기반 문장 임베딩과 contrastive 학습
- Knowledge distillation(큰 모델 출력으로 작은 모델 학습)

### 관련 논문
- [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models (Jiang et al., 2023)](https://arxiv.org/abs/2310.05736)
- [Learning to Filter Context for Retrieval-Augmented Generation / FILCO (Wang et al., 2023)](https://arxiv.org/abs/2311.08377)
- [xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token (Cheng et al., 2024)](https://arxiv.org/abs/2405.13792)
- [Corrective Retrieval Augmented Generation / CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)

### 실무 적용
- **2단계 파이프라인:** 작은 모델(또는 저렴한 LLM)로 top-k 문서를 질의 관련 문장만 추출 → 비싼 메인 LLM에는 압축본만 전달해 토큰 비용 절감
- **"근거 없음" 경로:** 압축 결과가 비면 검색 없이 답하거나 "모르겠다/추가 정보 필요"로 분기하는 가드레일
- **B2B SaaS 관점:** 고객사별 긴 매뉴얼·계약서 RAG에서 응답 지연과 API 비용을 낮추는 핵심 최적화 포인트(멀티테넌트 환경에서 단위 경제성 개선)

---

## Paper 3 (Recent): Search-o1: Agentic Search-Enhanced Large Reasoning Models
- **Authors:** Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, Zhicheng Dou
- **Year:** 2025 (EMNLP 2025)
- **arXiv:** https://arxiv.org/abs/2501.05366
- **PDF:** [./search-o1-li-2025.pdf](./search-o1-li-2025.pdf)
- **Code:** https://github.com/RUC-NLPIR/Search-o1
- **Citation Count:** ~400+ (approximate)

### 요약
o1/QwQ 같은 대형 추론 모델(LRM)은 긴 사고 과정 중 **지식이 부족한 지점에서 "아마도…"라며 불확실하게 추론을 이어가는** 문제가 있습니다. Search-o1은 추론 도중 모델이 스스로 `<|begin_search_query|>` 토큰으로 검색을 호출하는 **에이전틱 RAG**를 붙이고, 가져온 문서를 추론 체인에 그대로 넣지 않고 **별도의 "Reason-in-Documents" 모듈로 먼저 분석·정제**해 필요한 지식만 주입합니다. 과학(GPQA), 수학, 코딩 같은 고난도 추론 과제와 6개 오픈도메인 QA 벤치마크에서 성능 향상을 보고했습니다.

### 핵심 기여
- 학습 없이(training-free) 추론 모델에 **"필요할 때 스스로 검색"**하는 능력을 부여하는 에이전틱 검색 워크플로
- **Reason-in-Documents:** 검색 문서를 메인 추론과 분리된 단계에서 읽고 정제해, 긴 문서가 추론 흐름을 깨뜨리는 노이즈 문제 완화
- 단발성 RAG(검색 1회 후 생성) 대비, **다단계·반복 검색이 장문 추론에서 갖는 이점**을 체계적으로 실증

### 이 논문이 중요한 이유
오늘의 세 논문을 하나로 묶는 "현재형 답"입니다. Lost in the Middle이 지적한 **"긴 문서는 잘 못 읽는다"**, RECOMP가 제시한 **"넣기 전에 정제하라"**는 원칙이, 추론 모델 시대에는 **"추론 루프 안에서 검색하고, 검색 결과는 별도 서브루틴으로 소화한 뒤 결론만 돌려준다"**는 에이전트 설계로 진화했음을 보여줍니다. 이는 Deep Research 류 제품, 그리고 서브에이전트에 탐색을 위임하고 요약만 받는 멀티에이전트 패턴과 구조적으로 동일합니다. 이전 사이클의 Search-R1(RL로 검색 학습)과 비교하면 "프롬프트/워크플로 기반 vs 학습 기반"의 두 축을 이해할 수 있습니다.

### 사전 지식
- ReAct(Reason + Act) 패턴과 tool calling 개념
- 대형 추론 모델(o1, QwQ, DeepSeek-R1)의 long CoT 특성
- 멀티홉 QA(HotpotQA, 2WikiMultihopQA, MuSiQue)와 반복 검색(IRCoT, FLARE)

### 관련 논문
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (Jin et al., 2025)](https://arxiv.org/abs/2503.09516)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning (Chen et al., 2025)](https://arxiv.org/abs/2503.19470)
- [WebThinker: Empowering Large Reasoning Models with Deep Research Capability (Li et al., 2025)](https://arxiv.org/abs/2504.21776)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

### 실무 적용
- **Deep Research형 기능:** 추론 모델 + 검색 도구 + "문서 소화용 서브 호출"로 리서치 리포트 생성 에이전트 구현
- **컨텍스트 위생(context hygiene):** 웹/사내 문서 원문을 메인 에이전트 컨텍스트에 붓지 말고, 서브에이전트가 읽고 요약한 결과만 반환하는 아키텍처
- **평가 포인트:** 검색 호출 횟수, 호출 시점(불확실성 신호), 정제 단계 전후 정답률을 로그로 남겨 비용-정확도 곡선을 관리

---

## 추천 읽기 순서
1. **Lost in the Middle** — 문제 정의부터. Figure 1(U자 곡선)과 오픈도메인 QA의 k별 포화 실험만 봐도 핵심을 얻을 수 있습니다.
2. **RECOMP** — "그럼 어떻게 줄일까?"에 대한 고전적 해법. 압축기 학습 목표(downstream 성능 기반)와 selective augmentation 부분에 집중하세요.
3. **Search-o1** — 앞의 두 원칙이 에이전틱 추론 시대에 어떻게 재조합되는지 확인. Reason-in-Documents 모듈과 단발 RAG 대비 실험을 중점적으로 읽으세요.

## 핵심 테이크어웨이
- **Q. 컨텍스트 윈도우가 크면 검색 결과를 다 넣어도 되지 않나?**
  A. 아니요. 위치에 따라 활용도가 다르고(U자형), 문서 수를 늘려도 정확도는 포화됩니다. "넣는 양"보다 "넣는 질과 배치"가 중요합니다.
- **Q. Advanced RAG의 공통 원리는?**
  A. 검색 결과를 **그대로 믿지 않고** 평가(Self-RAG/CRAG) → 정제·압축(RECOMP) → 필요한 만큼만 반복 검색(Search-o1)하는 것입니다.
- **Q. 에이전트 시대에 이 원리는 어떻게 바뀌었나?**
  A. 검색이 "파이프라인의 한 단계"에서 "추론 중 호출하는 도구"로 바뀌었고, 문서 읽기는 메인 루프와 분리된 서브 단계로 위임됩니다.
- **Q. 제품 관점에서 가장 먼저 해볼 실험은?**
  A. 현재 RAG의 top-k와 문서 배치를 바꿔가며 정답률·토큰비용·지연을 측정하고, 압축 단계를 넣었을 때의 비용 대비 효과를 A/B로 확인해보세요.

## 다음 토픽과의 연결
다음 토픽은 **Vector Databases and Indexing**입니다. 오늘 다룬 논문들이 "검색 결과를 어떻게 읽을 것인가"였다면, 다음은 그 앞단 — **수많은 문서 중 후보를 얼마나 빠르고 정확하게 찾아올 것인가**(ANN 인덱스, HNSW/IVF, 하이브리드 검색)입니다. 특히 Search-o1처럼 추론 중 검색을 여러 번 호출하는 에이전트 구조에서는 **검색 지연이 곧 전체 응답 지연**이 되므로, 인덱스 설계가 에이전트 UX에 직접적인 영향을 줍니다.
