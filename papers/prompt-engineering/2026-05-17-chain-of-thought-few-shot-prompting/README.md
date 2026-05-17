# Daily AI Paper Recommendations

> **Date:** 2026-05-17
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting (2회차)

---

## Paper 1 (Classic): Large Language Models are Zero-Shot Reasoners
- **Authors:** Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa
- **Year:** 2022 (NeurIPS 2022)
- **arXiv:** https://arxiv.org/abs/2205.11916
- **PDF:** [./zero-shot-reasoners-kojima-2022.pdf](./zero-shot-reasoners-kojima-2022.pdf)
- **Citation Count:** 약 5,500회 이상 (2026년 기준)

### 요약
거대 언어 모델(LLM)이 "Let's think step by step"라는 단 한 문장을 프롬프트 앞에 붙이는 것만으로 별도의 예시(exemplar) 없이도 단계적 추론(Zero-shot CoT)을 수행할 수 있음을 처음으로 보여준 논문이다. 별도의 fine-tuning이나 task-specific few-shot 예제 없이도 산술/상식/기호 추론 벤치마크에서 성능이 비약적으로 향상된다는 것이 핵심 발견이다.

### 핵심 기여
- "Let's think step by step"라는 단일 트리거 프롬프트로 zero-shot 환경에서도 추론 능력이 드러난다는 것을 입증했다. MultiArith 정답률을 17.7% → 78.7%, GSM8K를 10.4% → 40.7%로 끌어올렸다.
- Few-shot CoT(Wei et al. 2022)가 "예시를 통해" 추론을 유도한 반면, 본 논문은 "지시문만으로" 추론을 유도할 수 있음을 보였다. 즉, 추론 능력은 모델 내부에 이미 잠재되어 있다는 것을 시사한다.
- 2단계 프롬프팅(① reasoning extraction "Let's think step by step" → ② answer extraction "Therefore, the answer is...") 패턴을 정형화하여, 후속 자동 프롬프팅·에이전트 연구의 기본 빌딩 블록을 만들었다.

### 이 논문이 중요한 이유
AI 엔지니어 입장에서 이 논문은 "예시 없이도 LLM의 추론 능력을 끌어낼 수 있다"는 사실을 가장 단순한 형태로 증명했다는 점에서 의미가 크다. 프로덕트에서 prompt token budget이 빠듯하거나, 정답 예시를 수집하기 어려운 도메인(법률·의료·고객 문의)에서 few-shot exemplar 없이도 reasoning을 유도하는 기준선(baseline)을 제공한다. 또한 o1/o3/DeepSeek-R1과 같은 reasoning model이 등장하기 이전, "추론은 학습이 아니라 elicitation"이라는 패러다임을 정착시켰다.

### 사전 지식
- Few-shot prompting과 in-context learning의 기본 개념 (GPT-3 논문)
- Chain-of-Thought few-shot 버전(Wei et al. 2022)과의 차이점
- LLM scaling laws — 추론 능력은 모델 크기 일정 임계점 이상에서 발현되는 emergent ability라는 관점

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682)
- [Automatic Chain of Thought Prompting / Auto-CoT (Zhang et al., 2022)](https://arxiv.org/abs/2210.03493)

### 실무 적용
- **고객 지원 챗봇**: 답변 전에 "단계별로 생각해보겠습니다"를 system prompt에 삽입하여 multi-step troubleshooting 정확도 향상
- **B2B SaaS 분석 에이전트**: SQL 생성·데이터 해석 시 zero-shot CoT를 default mode로 두고, 복잡 케이스에 한해 few-shot으로 augment
- **AI Dubbing/Avatar 시나리오 생성**: 캐릭터별 reasoning을 강제하여 일관된 대본 생성
- **온디바이스/소형 모델 한계**: 소형 모델(7B 이하)에서는 zero-shot CoT 효과가 미미하므로, 모델 크기에 따라 strategy를 다르게 가져가야 함

---

## Paper 2 (Classic): Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **Year:** 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2203.11171
- **PDF:** [./self-consistency-wang-2022.pdf](./self-consistency-wang-2022.pdf)
- **Citation Count:** 약 3,800회 이상 (2026년 기준)

### 요약
Chain-of-Thought 프롬프팅의 단일 greedy decoding 결과 대신, 다양한 reasoning path를 샘플링한 뒤 최종 답변에 대해 다수결(marginalization)을 취하는 self-consistency 디코딩 전략을 제안했다. 동일한 문제에 대해 다양한 추론 경로가 같은 정답으로 수렴하는 경향이 있다는 직관에 기반하며, 별도의 학습이나 supervision 없이 추론 정확도를 큰 폭으로 끌어올린다.

### 핵심 기여
- 단일 그리디 디코딩 → 다중 sampling + majority vote 라는 단순하지만 강력한 추론 안정화 기법을 정립했다. GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4% 등 거의 모든 reasoning 벤치마크에서 새로운 SOTA를 달성했다.
- "Reasoning이 다양해도 정답은 수렴한다"는 통찰을 정량적으로 보여, 이후 verifier-free ensemble, test-time compute 확장의 이론적 기반이 되었다.
- Temperature·top-p 같은 stochastic decoding 파라미터를 reasoning 품질 향상의 도구로 본격 활용한 첫 사례 중 하나.

### 이 논문이 중요한 이유
AI 엔지니어가 가장 즉시 적용할 수 있는 "test-time compute" 기법이다. 모델을 재학습하지 않고도 같은 모델로 호출 횟수만 늘려 reasoning 품질을 크게 끌어올릴 수 있고, 이는 곧 "추론 비용 ↔ 정답률 ↔ 응답 시간"이라는 운영 트레이드오프를 PM이 직접 다이얼링할 수 있는 levers를 만들어준다. OpenAI o1/o3 같은 reasoning model의 internal 자기검증, test-time scaling laws의 시초이기도 하다.

### 사전 지식
- Sampling 기반 디코딩 (temperature, top-k, top-p, nucleus sampling)
- Few-shot Chain-of-Thought 프롬프팅 (Wei et al. 2022)
- Marginalization, ensembling, majority voting의 기본 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)
- [Universal Self-Consistency (Chen et al., 2023)](https://arxiv.org/abs/2311.17311)
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)
- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

### 실무 적용
- **Agentic 워크플로의 critical-path 호출**: 의사결정 노드(예: tool 선택, 라우팅)에서 N=5~10 samples로 self-consistency를 적용하면 환각/오라우팅을 크게 줄일 수 있다.
- **데이터 드리븐 검증**: A/B 테스트에서 single-call vs self-consistency의 정답률·비용·지연 곡선을 그려, 사용 케이스별 최적 N을 결정
- **AI Dubbing/Avatar 시나리오 평가**: 동일 대본에 대해 여러 번 평가·요약을 돌려 일관된 결과만 채택
- **비용 관리**: N이 늘수록 호출 비용·지연이 선형 증가 → high-stakes 단계에만 selective하게 적용해야 함. 일반 응답은 N=1 + 일관성 모니터링.

---

## Paper 3 (Recent): Buffer of Thoughts — Thought-Augmented Reasoning with Large Language Models
- **Authors:** Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E. Gonzalez, Bin Cui
- **Year:** 2024 (NeurIPS 2024 Spotlight)
- **arXiv:** https://arxiv.org/abs/2406.04271
- **PDF:** [./buffer-of-thoughts-yang-2024.pdf](./buffer-of-thoughts-yang-2024.pdf)
- **Citation Count:** 약 400회 이상 (2026년 기준, NeurIPS 2024 Spotlight)

### 요약
"thought-template"이라는 high-level 추론 패턴을 meta-buffer에 저장해 두고, 새 문제가 들어오면 가장 관련성 높은 템플릿을 검색해 인스턴스화하여 추론을 수행하는 thought-augmented reasoning 프레임워크다. Tree-of-Thought이나 Graph-of-Thought 같은 multi-query 방식 대비 평균 12% 비용으로 동등 이상의 reasoning 정확도를 달성하며, Game of 24 +11%, Geometric Shapes +20%, Checkmate-in-One +51%로 큰 격차의 SOTA를 기록했다.

### 핵심 기여
- **Meta-buffer 도입**: 과거 문제 해결 과정에서 distill한 thought-template들을 retrieval 가능한 형태로 축적하여, RAG와 유사한 reasoning-template retrieval 패러다임을 정립했다.
- **Buffer-manager**: 새 task가 들어올 때마다 meta-buffer를 동적으로 업데이트하여, 시간이 지날수록 reasoning 품질과 효율이 함께 향상되는 self-improving 구조를 제시했다.
- **비용 효율성**: ToT/GoT 대비 평균 12% 비용으로 SOTA reasoning 성능을 달성 — 즉 "추론 품질 vs. 추론 비용" 트레이드오프 곡선 자체를 끌어올렸다.

### 이 논문이 중요한 이유
Agentic AI 시대의 가장 큰 운영 이슈인 "추론 비용 폭증" 문제에 대한 시스템 아키텍처 차원의 해법이다. 비용 측면에서 ToT/GoT는 프로덕션 적용이 어려운 반면, BoT는 retrieval 기반이라 latency·비용 모두 통제 가능하다. 또한 thought-template이라는 추상화는 "조직의 reasoning 노하우를 데이터셋으로 축적한다"는 새로운 자산화 관점을 제공한다 — PM 관점에서는 "사용자 문제 해결 패턴을 자산으로 만드는" data-driven moat 설계와 직결된다.

### 사전 지식
- Tree-of-Thoughts (Yao et al., 2023), Graph-of-Thoughts (Besta et al., 2023)
- RAG와 dense retrieval 기본 개념 (embedding, similarity search)
- Self-Consistency, ReAct, Reflexion 등 기존 reasoning augmentation 기법
- In-context learning에서 exemplar 선택의 중요성

### 관련 논문
- [Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts: Solving Elaborate Problems with LLMs (Besta et al., 2023)](https://arxiv.org/abs/2308.09687)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Automatic Chain of Thought Prompting / Auto-CoT (Zhang et al., 2022)](https://arxiv.org/abs/2210.03493)

### 실무 적용
- **Agentic 제품의 thought-template 자산화**: 고객 문제 해결 로그에서 잘 동작한 reasoning trace를 distill해서 사내 thought-buffer에 적재 → 신규 케이스에 retrieval로 재활용
- **AI 영업/CS 에이전트**: 과거 성공 케이스의 reasoning template을 retrieval하여 신규 유저 케이스에 인스턴스화 → 적은 비용으로 일관된 품질
- **B2B 워크플로 자동화**: 회사별/도메인별 thought-template을 vertical SaaS의 핵심 자산으로 운영 가능 (Notion·Linear·HubSpot 통합 자동화)
- **모니터링 지표 설계**: meta-buffer hit rate, template-당 평균 reasoning step 수, 비용·정확도 곡선을 핵심 PM 지표로 관리

---

## 추천 읽기 순서

1. **Kojima 2022 (Zero-Shot Reasoners)** 먼저 — "예시 없이도 reasoning이 elicit 된다"는 가장 단순한 형태부터 이해해야 이후의 정교한 기법들이 왜 등장했는지 맥락이 잡힌다.
2. **Wang 2022 (Self-Consistency)** — 단일 추론 → 다중 추론 + 집계 라는 "test-time compute scaling"의 출발점. Kojima와 함께 읽으면 "추론 elicitation"과 "추론 ensembling"이라는 두 축이 명확해진다.
3. **Yang 2024 (Buffer of Thoughts)** 마지막 — 위 두 흐름을 retrieval 기반으로 통합·확장하는 최신 시스템. NeurIPS 2024 Spotlight인 이유는 단순 정확도 개선이 아니라 비용·운영 관점에서 production-friendly한 reasoning을 제시했기 때문이다.

## 핵심 테이크어웨이

- **Q. Reasoning은 학습되는 것인가, 끌어내는 것인가?**
  - A. Kojima 2022는 "이미 모델에 내재되어 있고 단지 elicitation의 문제"라는 입장을 정립했다. 이후 모든 prompting 연구의 출발점이 되었다.

- **Q. 같은 모델로 추론 정확도를 어떻게 끌어올리는가?**
  - A. Self-Consistency가 보여주듯, 같은 모델을 N번 호출 → majority vote만으로도 10~20%p 향상이 가능하다. 이것이 "test-time compute scaling"의 시초다.

- **Q. ToT/GoT처럼 비싼 reasoning을 production에 어떻게 적용하나?**
  - A. Buffer of Thoughts가 답이다. retrieval 기반으로 thought-template을 재사용하면 12% 비용으로 동급 SOTA에 도달한다. "reasoning 비용 ↔ 품질" 곡선 자체를 끌어올리는 것이 핵심.

- **Q. PM 관점의 시사점은?**
  - A. 세 논문 모두 "모델을 더 키우지 않고도 prompting/decoding/retrieval만으로 reasoning 품질을 끌어올릴 수 있다"는 일관된 메시지를 준다. 즉 cost·UX·정확도 trade-off는 product/infra 레이어에서 컨트롤 가능한 levers다.

## 다음 토픽과의 연결

- **Day 19: Advanced Prompting (ToT, ReAct, Self-Consistency 심화)** — 오늘 Self-Consistency를 살펴봤다면, 다음 토픽에서는 Tree-of-Thoughts와 ReAct로 reasoning을 명시적인 search/tool-use로 확장하는 단계로 이어진다. Buffer of Thoughts는 ToT의 "비용 문제"를 푼 후속이므로 자연스러운 다리 역할을 한다.
- **Module 8: LangChain & Orchestration** — Buffer of Thoughts의 meta-buffer는 결국 reasoning template을 위한 retrieval 시스템이므로, LangGraph·LlamaIndex 같은 orchestration framework에서 어떻게 구현되는지로 확장된다.
- **Module 9: RAG** — thought-template retrieval은 dense embedding 기반 retrieval의 reasoning 확장형이다. 다음 RAG 토픽에서 다룰 임베딩·검색 기법이 BoT의 meta-buffer 성능을 직접 결정한다.
