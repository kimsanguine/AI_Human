# Daily AI Paper Recommendations

> **Date:** 2026-10-27
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting ToT ReAct Self-Consistency

> 이번 사이클에서는 이전에 다룬 ToT·ReAct·Self-Consistency·Reflexion·Least-to-Most·Self-Refine·PAL·Self-Ask·Decomposed Prompting·Plan-and-Solve·LATS·CoVe·Universal Self-Consistency와 겹치지 않도록, **"추상화(Step-Back)" → "다수 모델 토론(Debate)" → "다수 트리 합의(Forest-of-Thought)"** 로 이어지는 새로운 축을 골랐습니다.

---

## Paper 1 (Classic): Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
- **Authors:** Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V. Le, Denny Zhou
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.06117
- **PDF:** [./step-back-prompting-zheng-2023.pdf](./step-back-prompting-zheng-2023.pdf)
- **Citation Count:** ~600+

### 요약
구체적인 질문에 바로 답하게 하지 않고, 먼저 "이 질문의 바탕이 되는 원리·개념은 무엇인가?"라는 한 단계 위의 질문(step-back question)을 던지게 한 뒤, 그 추상화된 원리를 근거로 원래 질문을 추론하게 하는 2단계 프롬프팅 기법입니다. PaLM-2L 기준 MMLU 물리 +7%, 화학 +11%, TimeQA +27%, MuSiQue +7% 개선을 보고했습니다.

### 핵심 기여
- **Abstraction → Reasoning** 2단계 구조 제안: 세부 사항에 매몰되어 중간 단계에서 틀리는 CoT의 약점을 "원리 먼저 떠올리기"로 보완
- STEM, 지식 QA(TimeQA, SituatedQA), 멀티홉 추론(MuSiQue, StrategyQA) 등 이질적인 과제에서 일관된 개선을 실증
- 오류 분석을 통해 "추상화 단계는 대부분 쉽게 성공하며, 남은 오류의 대부분은 추론 단계에서 발생한다"는 병목 위치를 정량적으로 제시

### 이 논문이 중요한 이유
CoT가 "어떻게 생각할지"를 길게 펼친다면, Step-Back은 "무엇을 기준으로 생각할지"를 먼저 고정합니다. 프롬프트 한두 줄 추가로 구현 가능하면서도 RAG(원리 질문으로 검색 → 검색 품질 향상)와 결합하기 쉬워, 실무 투입 대비 효과(ROI)가 매우 높은 기법입니다. 또한 "질문 재작성(query rewriting)"이라는 오늘날 RAG 파이프라인의 표준 단계를 이론적으로 뒷받침합니다.

### 사전 지식
- Chain-of-Thought, Few-shot 프롬프팅의 기본 개념 (Day 18)
- 멀티홉 QA와 RAG의 기본 구조
- "Take a deep breath" 같은 프롬프트 트릭과 구조적 프롬프팅의 차이

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models (Zhou et al., 2022)](https://arxiv.org/abs/2205.10625)
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures (Zhou et al., 2024)](https://arxiv.org/abs/2402.03620)

### 실무 적용
- **RAG 쿼리 확장:** 사용자의 구체적 질문("2019년 X사의 Y 정책은?")에서 원리/배경 질문("X사의 정책 변천사")을 생성해 함께 검색 → recall 향상. LangChain의 Step-Back 템플릿이 이 논문을 그대로 구현한 것입니다.
- **CS/기술지원 에이전트:** 증상(에러 메시지)에서 바로 해결책을 내지 말고, "이 에러가 속한 범주와 일반 원인"을 먼저 정리하게 하면 엉뚱한 해결책 비율이 줄어듭니다.
- **평가 포인트:** 추상화 단계 출력을 로그로 남기면, 답이 틀렸을 때 "원리를 잘못 잡았는지 vs 적용을 잘못했는지" 분리해 디버깅할 수 있습니다.

---

## Paper 2 (Classic): Improving Factuality and Reasoning in Language Models through Multiagent Debate
- **Authors:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch
- **Year:** 2023 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2305.14325
- **PDF:** [./multiagent-debate-du-2023.pdf](./multiagent-debate-du-2023.pdf)
- **Citation Count:** ~1,000+

### 요약
같은 질문에 대해 여러 LLM 인스턴스가 각자 답을 낸 뒤, 서로의 답과 근거를 읽고 비판하며 자신의 답을 고쳐 나가는 과정을 여러 라운드 반복해 최종적으로 합의된 답을 도출하는 "Society of Minds" 방식입니다. 산술·GSM8K·체스 수 예측 같은 추론 과제와 인물 전기·MMLU 같은 사실성 과제에서 단일 모델 대비 정확도를 높이고 환각을 줄였습니다.

### 핵심 기여
- Self-Consistency(독립 샘플 다수결)를 넘어 **샘플끼리 상호작용**하게 만든 프레임워크 — 블랙박스 API만으로 동작
- 에이전트 수·라운드 수가 늘수록 성능이 올라가는 "토론 스케일링" 경향과, 처음엔 모두 틀렸던 답이 토론 중 정답으로 수렴하는 사례를 제시
- 서로 다른 모델(ChatGPT + Bard) 간 토론도 효과가 있음을 보이며, 이종 모델 앙상블의 가능성을 열었음

### 이 논문이 중요한 이유
오늘날 멀티에이전트 시스템(AutoGen, CAMEL, LLM-as-a-Judge 패널, "검증 에이전트 분리" 패턴)의 핵심 아이디어인 **"모델끼리 서로 검토하게 하라"**를 가장 단순하고 재현 가능한 형태로 정리한 논문입니다. 프롬프트 엔지니어링이 단일 호출 최적화에서 **"여러 호출의 상호작용 설계"**로 넘어가는 분기점이라는 점에서 필독입니다.

### 사전 지식
- Self-Consistency (Wang et al., 2022)의 다수결 원리
- 환각(hallucination)과 사실성 평가 방법
- 추론 비용(토큰 × 에이전트 수 × 라운드 수)에 대한 감각

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate (Liang et al., 2023)](https://arxiv.org/abs/2305.19118)
- [ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate (Chan et al., 2023)](https://arxiv.org/abs/2308.07201)

### 실무 적용
- **고위험 답변 검증:** 법률·의료·금융 요약처럼 틀리면 비싼 답변에 한해 2~3개 에이전트 × 2라운드 토론을 적용하고, 합의 실패 시 사람에게 에스컬레이션.
- **LLM-as-a-Judge 패널:** 단일 Judge 대신 다수 Judge가 서로의 평가 근거를 읽고 조정하게 하면 평가 분산이 줄어듭니다.
- **비용 설계:** 비용이 에이전트×라운드에 비례하므로, 모든 요청이 아니라 "불확실도가 높은 요청"에만 토론을 트리거하는 라우팅 설계가 핵심입니다.

---

## Paper 3 (Recent): Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning
- **Authors:** Zhenni Bi, Kai Han, Chuanjian Liu, Yehui Tang, Yunhe Wang
- **Year:** 2024 (ICML 2025)
- **arXiv:** https://arxiv.org/abs/2412.09078
- **PDF:** [./forest-of-thought-bi-2024.pdf](./forest-of-thought-bi-2024.pdf)
- **Citation Count:** ~50+ (빠르게 증가 중)

### 요약
Tree-of-Thought가 한 그루의 추론 트리를 탐색하는 데 그쳤다면, Forest-of-Thought(FoT)는 여러 그루의 추론 트리를 병렬로 키우고 그 결과를 합의(consensus)로 통합하는 프레임워크입니다. 관련성 높은 경로만 활성화하는 희소 활성화(sparse activation), 실시간 오류를 바로잡는 동적 자기교정(dynamic self-correction), 합의 기반 의사결정 전략을 결합해 테스트 타임 컴퓨트를 효율적으로 늘려 추론 성능을 끌어올립니다.

### 핵심 기여
- ToT(탐색) + Self-Consistency(다수결) + Self-Correction(교정)을 **하나의 테스트 타임 스케일링 프레임워크로 통합**
- 희소 활성화로 모든 트리를 끝까지 키우지 않고 유망한 경로만 확장 → 정확도-비용 트레이드오프 개선
- Game of 24, GSM8K, MATH 등에서 트리 수를 늘릴수록 성능이 올라가는 "포레스트 스케일링" 경향 실증 (ICML 2025 채택)

### 이 논문이 중요한 이유
2024~2025년 추론 연구의 핵심 화두는 "모델을 키우는 대신 추론 시점에 더 많이 생각하게 하자(test-time compute scaling)"입니다. FoT는 이번 토픽에서 배운 ToT·Self-Consistency·Debate의 아이디어가 최신 연구에서 어떻게 **조합되고 스케일링되는지**를 보여주는 좋은 사례로, 고전 기법을 "부품"으로 보고 시스템을 설계하는 관점을 길러 줍니다.

### 사전 지식
- Tree of Thoughts (Yao et al., 2023)의 BFS/DFS 탐색 구조
- Self-Consistency의 다수결과 Universal Self-Consistency
- Test-time compute scaling 개념 (Snell et al., 2024)

### 관련 논문
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models (Besta et al., 2023)](https://arxiv.org/abs/2308.09687)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

### 실무 적용
- **난이도 적응형 추론:** 쉬운 질문은 단일 CoT, 어려운 질문만 여러 트리를 키우는 "적응형 포레스트"로 라우팅하면 평균 비용을 억제하면서 어려운 케이스 정확도를 올릴 수 있습니다.
- **코드/수학 에이전트:** 여러 풀이 경로를 병렬로 탐색하고 테스트·검증기로 경로를 가지치기하는 구조는 코딩 에이전트의 "여러 패치 후보 생성 → 테스트로 선택" 패턴과 동일합니다.
- **제품 관점 질문:** "정확도 +X%를 위해 지연시간과 비용을 몇 배까지 감수할 것인가?" — FoT는 이 질문을 실험으로 답하게 만드는 프레임워크입니다.

---

## 추천 읽기 순서
1. **Step-Back Prompting** — 가장 짧고 구현이 쉬움. 단일 호출 안에서 "추론의 기준점"을 바꾸는 법을 먼저 익힙니다.
2. **Multiagent Debate** — 단일 호출을 넘어 "여러 호출이 서로 검토하는" 구조로 확장합니다. Self-Consistency와 비교하며 읽어 보세요.
3. **Forest-of-Thought** — 앞의 두 아이디어(더 나은 추론 경로, 다수의 합의)가 ToT와 결합되어 테스트 타임 스케일링으로 발전하는 모습을 확인합니다.

## 핵심 테이크어웨이
- **Q. 고급 프롬프팅은 결국 무엇을 바꾸는가?** → ① 추론의 *출발점*(Step-Back: 원리 먼저), ② 추론의 *참여자 수*(Debate: 여러 모델), ③ 추론의 *탐색 폭*(FoT: 여러 트리)을 바꿉니다.
- **Q. 더 많이 호출하면 무조건 좋은가?** → 아닙니다. 세 논문 모두 성능 향상의 대가로 토큰·지연시간이 늘어납니다. 실무의 핵심은 "언제 비싼 추론을 켤지"를 정하는 라우팅입니다.
- **Q. 가설로 세워볼 만한 것은?** → "우리 서비스의 오답 중 상당수는 지식 부족이 아니라 *잘못된 프레이밍*에서 온다" — Step-Back을 A/B 테스트해 검증해 볼 수 있습니다.
- **Q. 상호 검토는 왜 효과가 있는가?** → 독립 샘플은 같은 실수를 반복할 수 있지만, 서로의 근거를 읽으면 오류가 드러날 기회가 생깁니다. 단, 다수가 틀린 답으로 동조할 위험도 있어 이종 모델·역할 분리가 도움이 됩니다.

## 다음 토픽과의 연결
다음 토픽 **Automatic Prompt Optimization**에서는 사람이 Step-Back 질문이나 토론 프롬프트를 손으로 설계하는 대신, APE·DSPy·TextGrad처럼 **프롬프트 자체를 자동으로 탐색·최적화**하는 방법을 다룹니다. 오늘의 기법들은 그 최적화 대상이 되는 "추론 모듈"이며, Debate와 FoT의 다단계 구조는 DSPy가 다루는 "멀티 스테이지 LM 프로그램"의 대표적인 예입니다.
