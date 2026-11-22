# Daily AI Paper Recommendations

> **Date:** 2026-11-23
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting ToT ReAct Self-Consistency

> 이번 사이클에서는 이전에 다룬 ToT·ReAct·Self-Consistency·Reflexion·Least-to-Most·Self-Refine·PAL·Self-Ask·Decomposed Prompting·Plan-and-Solve·Let's Verify·LATS·CoVe·Universal Self-Consistency·Step-Back·Multiagent Debate와 겹치지 않도록, 세 기법 계열을 각각 한 단계씩 "확장"한 논문을 골랐습니다.
> **ToT 계열 → RAP(세계 모델 + MCTS)**, **ReAct 계열 → ReWOO(관찰과 추론의 분리)**, **Self-Consistency 계열 → Large Language Monkeys(반복 샘플링의 스케일링 법칙)**.

---

## Paper 1 (Classic): Reasoning with Language Model is Planning with World Model
- **Authors:** Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
- **Year:** 2023 (EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2305.14992
- **PDF:** [./reasoning-via-planning-rap-hao-2023.pdf](./reasoning-via-planning-rap-hao-2023.pdf)
- **Citation Count:** ~700+

### 요약
LLM이 어려운 계획·추론 문제에서 실수하는 근본 원인을 "행동의 결과로 세상이 어떻게 바뀔지 예측하는 내부 세계 모델(world model)이 없기 때문"으로 진단합니다. 이를 해결하기 위해 같은 LLM을 **추론 에이전트**와 **세계 모델** 두 역할로 동시에 사용하고, Monte Carlo Tree Search(MCTS)로 추론 경로를 탐색하는 RAP(Reasoning via Planning) 프레임워크를 제안합니다. Blocksworld 계획 생성, GSM8K 수학, PrOntoQA 논리 추론에서 CoT·Self-Consistency·Least-to-Most를 능가했고, LLaMA-33B + RAP가 GPT-4 + CoT보다 계획 생성에서 33% 상대 개선을 보였습니다.

### 핵심 기여
- **LLM = 에이전트 + 세계 모델**: 에이전트가 "행동(다음 추론 단계)"을 제안하면, 세계 모델이 "그 결과 상태(중간 변수 값, 블록 배치 등)"를 예측하게 하는 구조를 정립
- **MCTS 기반 탐색**: ToT의 BFS/DFS 대신 탐색(exploration)과 활용(exploitation)의 균형을 잡는 MCTS를 도입하고, 행동 가능성·자기 평가·상태 신뢰도·과제별 휴리스틱을 조합한 보상 설계를 제시
- **오픈소스 추론 라이브러리 `llm-reasoners`** 공개: 이후 많은 트리 탐색 추론 연구(LATS, rStar 등)의 기준 구현이 됨

### 이 논문이 중요한 이유
ToT가 "여러 생각을 가지로 펼쳐 본다"는 아이디어였다면, RAP는 "가지를 펼칠 때 **다음 상태를 시뮬레이션**하고 **보상으로 가지를 고른다**"는 강화학습/계획 이론의 언어로 추론을 재정의했습니다. 2024~2025년 o1·R1류 추론 모델, Process Reward Model, MCTS 기반 test-time compute 연구를 이해하는 데 필요한 개념적 다리 역할을 합니다. "작은 모델 + 좋은 탐색 알고리즘이 큰 모델의 단순 CoT를 이긴다"는 메시지는 추론 비용 설계에서 매우 실용적인 시사점입니다.

### 사전 지식
- Chain-of-Thought, Self-Consistency, Tree of Thoughts의 기본 개념 (Day 18~19 이전 사이클)
- MCTS의 4단계(Selection → Expansion → Simulation → Back-propagation)와 UCT 공식
- 강화학습의 상태(state)·행동(action)·보상(reward) 개념

### 관련 논문
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models (Zhou et al., 2023)](https://arxiv.org/abs/2310.04406)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers / rStar (Qi et al., 2024)](https://arxiv.org/abs/2408.06195)

### 실무 적용
- **다단계 작업 에이전트**: 여행 일정, 배포 절차, 데이터 파이프라인 구성처럼 "순서가 틀리면 실패하는" 작업에서, 각 단계 실행 전 LLM에게 "이 행동 후 상태는?"을 먼저 예측시키면 실패 경로를 사전에 가지치기할 수 있습니다.
- **비용-품질 트레이드오프**: 고가의 대형 모델 1회 호출 대신 소형 모델 + 탐색으로 동급 품질을 내는 구조를 검토할 때 기준점이 됩니다(단, 호출 횟수·지연 시간 증가를 함께 측정해야 함).
- **평가/디버깅**: 세계 모델이 예측한 중간 상태를 로그로 남기면 "어느 단계에서 상태 예측이 틀렸는지"를 추적할 수 있어, 에이전트 실패 분석이 훨씬 쉬워집니다.

---

## Paper 2 (Classic): ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models
- **Authors:** Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, Dongkuan Xu
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.18323
- **PDF:** [./rewoo-decoupling-reasoning-xu-2023.pdf](./rewoo-decoupling-reasoning-xu-2023.pdf)
- **Citation Count:** ~300+

### 요약
ReAct처럼 "생각 → 도구 호출 → 관찰 → 다시 생각"을 번갈아 하는 방식은 매 단계마다 이전 프롬프트 전체를 다시 넣어야 해서 토큰 낭비가 큽니다. ReWOO(Reasoning WithOut Observation)는 **Planner**가 도구 결과를 보지 않고 전체 계획을 변수(#E1, #E2…)와 함께 한 번에 작성하고, **Worker**가 도구를 실행해 변수를 채우며, **Solver**가 모든 증거를 모아 최종 답을 내는 3단 모듈 구조입니다. HotpotQA에서 토큰 효율 5배, 정확도 4% 향상을 보였고, 도구 실패 상황에서도 더 견고했습니다.

### 핵심 기여
- **Planner–Worker–Solver 분리**: 추론(파라메트릭)과 관찰(비파라메트릭 도구 호출)을 분리해 중복 프롬프트를 제거한 최초의 체계적 연구
- **토큰 효율 정량화**: 6개 공개 벤치마크와 자체 데이터셋에서 ReAct 대비 토큰 사용량을 크게 줄이면서 정확도는 유지·개선
- **소형 모델로의 오프로딩**: 계획 능력만 따로 떼어 instruction tuning할 수 있어, GPT-3.5(175B)의 계획 능력을 LLaMA 7B로 증류하는 데 성공

### 이 논문이 중요한 이유
ReAct는 에이전트의 기본형이지만, 실서비스에서는 "느리고 비싸다"는 문제가 바로 드러납니다. ReWOO는 오늘날 LangGraph의 **Plan-and-Execute**, LLMCompiler, 병렬 함수 호출(parallel tool calls) 같은 패턴의 원형입니다. 언제 ReAct(관찰에 따라 유연하게 경로 수정)를 쓰고, 언제 ReWOO(미리 계획 → 일괄 실행)를 쓸지 판단하는 것이 에이전트 설계자의 핵심 역량입니다.

### 사전 지식
- ReAct의 Thought–Action–Observation 루프 (이전 사이클 Day 19)
- Function calling / Tool use의 기본 구조
- 프롬프트 토큰 비용과 컨텍스트 누적 문제

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [An LLM Compiler for Parallel Function Calling (Kim et al., 2023)](https://arxiv.org/abs/2312.04511)
- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)

### 실무 적용
- **비용 최적화**: 검색 3~5회가 필요한 리서치형 질의에서 ReAct 대신 "계획 1회 + 병렬 검색 + 종합 1회"로 바꾸면 LLM 호출 수와 지연 시간이 크게 줄어듭니다.
- **하이브리드 설계**: 기본은 ReWOO로 빠르게 처리하고, Solver가 "증거 부족"을 감지하면 ReAct 루프로 폴백(fallback)하는 구조가 실무에서 자주 쓰입니다.
- **모델 라우팅**: Planner만 큰 모델, Worker·Solver는 작은 모델로 나누면 품질을 유지하면서 단가를 낮출 수 있습니다.

---

## Paper 3 (Recent): Large Language Monkeys: Scaling Inference Compute with Repeated Sampling
- **Authors:** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.21787
- **PDF:** [./large-language-monkeys-brown-2024.pdf](./large-language-monkeys-brown-2024.pdf)
- **Citation Count:** ~400+

### 요약
"한 문제에 한 번만 답하게 한다"는 관행을 깨고, 같은 문제에 대해 답을 **수백~수만 번 반복 샘플링**하면 무슨 일이 일어나는지 체계적으로 측정한 연구입니다. 생성된 샘플 중 하나라도 정답인 비율(coverage)이 샘플 수에 따라 4자릿수(10⁴) 범위에서 꾸준히 증가하며, 그 관계가 로그-선형(지수화된 거듭제곱 법칙)으로 모델링된다는 **추론 시점 스케일링 법칙**을 제시했습니다. SWE-bench Lite에서 DeepSeek-Coder-V2-Instruct의 해결률이 1회 샘플 15.9% → 250회 샘플 56%로 올라 당시 단일 샘플 SOTA(43%)를 넘었습니다.

### 핵심 기여
- **Coverage 스케일링 법칙**: 반복 샘플링 횟수와 정답 포함률 사이의 정량적 관계를 여러 모델·과제(GSM8K, MATH, MiniF2F, CodeContests, SWE-bench)에서 확인
- **"약한 모델 × 많은 샘플 > 강한 모델 × 1회"**: 같은 비용으로 저렴한 모델을 여러 번 돌리는 것이 더 나은 경우가 있음을 비용(FLOPs·달러) 기준으로 비교
- **검증기(verifier)의 병목 발견**: 자동 검증이 불가능한 영역에서는 다수결(Self-Consistency)과 보상 모델이 수백 샘플 이후 정체되어, coverage 증가분을 성능으로 전환하지 못함을 밝힘

### 이 논문이 중요한 이유
Self-Consistency가 "여러 번 샘플링해서 다수결"이었다면, 이 논문은 그 아이디어를 **스케일링 법칙의 언어로 일반화**하고, 진짜 병목은 "생성"이 아니라 "선택(검증)"이라는 점을 명확히 했습니다. 이는 2025년 이후 test-time compute, best-of-N, 검증기·보상 모델 연구의 출발점이 되었습니다. AI 엔지니어 입장에서는 "샘플 수를 늘릴 것인가, 검증기를 개선할 것인가"라는 예산 배분 질문에 데이터 기반으로 답하는 프레임을 줍니다.

### 사전 지식
- Self-Consistency와 pass@k 지표의 차이
- 학습 시점 스케일링 법칙(Kaplan et al., 2020) 개념 (Day 14)
- 단위 테스트·증명 검사기 같은 자동 검증기와 보상 모델(RM)의 개념

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

### 실무 적용
- **코드 생성 에이전트**: 테스트 코드가 있는 환경이라면 N개 패치를 병렬 생성하고 테스트 통과 여부로 고르는 best-of-N 전략이 단일 시도보다 훨씬 높은 해결률을 냅니다(SWE-bench 에이전트들의 표준 기법).
- **검증기 우선 투자**: 정답을 자동으로 판정할 수 없는 영역(요약, 상담 답변)에서는 샘플 수를 늘리기보다 LLM-as-a-Judge·규칙 기반 체커 같은 검증 레이어를 먼저 개선하는 것이 ROI가 높습니다.
- **비용 실험 설계**: "작은 모델 k회 vs 큰 모델 1회"를 같은 예산에서 A/B로 비교하는 실험 템플릿으로 활용할 수 있습니다.

---

## 추천 읽기 순서
1. **ReWOO** — ReAct를 이미 알고 있다면 가장 빠르게 읽히고, "에이전트 비용"이라는 실무 감각을 먼저 잡아 줍니다.
2. **RAP** — ToT를 강화학습·계획의 관점(세계 모델, MCTS, 보상)으로 다시 보는 법을 익힙니다. 3~4장의 MCTS 설계 부분을 집중해서 읽으세요.
3. **Large Language Monkeys** — 앞의 두 논문이 "어떻게 더 잘 탐색할까"였다면, 이 논문은 "탐색량을 늘리면 얼마나 좋아지고, 어디서 막히는가"를 정량화합니다. Figure 1~3과 검증기 정체 구간 분석을 꼭 보세요.

## 핵심 테이크어웨이
- **ToT → RAP**: 추론 트리 탐색의 품질은 "다음 상태를 얼마나 잘 예측하는가(세계 모델)"와 "어떤 보상으로 가지를 고르는가"에 달려 있습니다.
- **ReAct → ReWOO**: 관찰을 매번 기다리는 루프는 유연하지만 비쌉니다. 미리 계획하고 병렬 실행하는 구조와의 트레이드오프를 의식적으로 선택해야 합니다.
- **Self-Consistency → Repeated Sampling**: 샘플을 늘리면 정답이 "어딘가에는" 있을 확률은 계속 오르지만, 그것을 골라낼 검증기가 없으면 성능으로 이어지지 않습니다.
- 세 논문의 공통 질문은 **"추론 시점에 계산을 어디에, 얼마나 쓸 것인가"** 이며, 이는 곧 AI 제품의 단가·지연 시간·품질을 결정하는 설계 질문입니다.

## 다음 토픽과의 연결
다음 토픽은 **Automatic Prompt Optimization**입니다. 오늘 본 논문들이 "주어진 프롬프트로 추론을 어떻게 더 잘 탐색할까"였다면, 다음은 **프롬프트 자체를 탐색·최적화의 대상**으로 삼습니다. RAP의 탐색-보상 구조, Monkeys의 "검증기가 병목"이라는 교훈은 APE·OPRO·DSPy·TextGrad에서 프롬프트 후보를 생성하고 평가 지표로 고르는 과정과 그대로 연결됩니다.
