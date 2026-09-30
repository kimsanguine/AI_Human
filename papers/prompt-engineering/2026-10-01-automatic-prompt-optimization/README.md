# Daily AI Paper Recommendations

> **Date:** 2026-10-01
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): GPT Understands, Too
- **Authors:** Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2103.10385
- **PDF:** [./p-tuning-gpt-understands-too-liu-2021.pdf](./p-tuning-gpt-understands-too-liu-2021.pdf)
- **Citation Count:** ~1,500+

### 요약
수작업으로 만든 이산(discrete) 프롬프트는 단어 하나만 바꿔도 성능이 크게 흔들린다는 문제에서 출발합니다. P-Tuning은 학습 가능한 연속(continuous) 프롬프트 임베딩을 이산 프롬프트와 결합하고, 작은 프롬프트 인코더(LSTM/MLP)로 이를 최적화합니다. 모델을 고정한 상태에서 LAMA 기준 수작업 프롬프트 대비 20포인트 이상 향상되었고, few-shot SuperGLUE에서도 SOTA를 달성했습니다.

### 핵심 기여
- 이산 프롬프트의 불안정성(단어 하나에 따른 성능 급변)을 실증적으로 보여줌
- 연속 프롬프트 임베딩 + 프롬프트 인코더(재매개변수화)로 학습 안정성 확보
- "GPT는 NLU를 못한다"는 통념을 뒤집어, 적절한 프롬프트 튜닝으로 GPT류도 BERT에 필적함을 보임

### 이 논문이 중요한 이유
"프롬프트를 사람이 쓰지 않고 학습한다"는 흐름(Prefix-Tuning, Prompt Tuning, P-Tuning v2)의 핵심 축입니다. 자동 프롬프트 최적화를 '연속 공간 최적화'와 '이산 텍스트 탐색' 두 갈래로 나눠 이해할 때 연속 공간 쪽의 기준점이 됩니다.

### 사전 지식
- BERT/GPT 구조와 임베딩 레이어의 역할
- 프롬프트 기반 few-shot 학습 (PET, LM-BFF)
- 역전파와 파라미터 고정(frozen) 학습 개념

### 관련 논문
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks (Liu et al., 2021)](https://arxiv.org/abs/2110.07602)
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)

### 실무 적용
오픈소스 모델을 자체 서빙하는 환경에서 태스크별로 수십~수백 개의 가상 토큰만 학습해 저장하면, 하나의 베이스 모델로 여러 도메인(분류, 추출, 라우팅)을 저비용으로 운영할 수 있습니다. Hugging Face PEFT 라이브러리의 `PromptEncoder`가 바로 이 방식입니다. 단, API형 폐쇄 모델에는 적용할 수 없다는 한계가 있습니다.

---

## Paper 2 (Classic): Instruction Induction: From Few Examples to Natural Language Task Descriptions
- **Authors:** Or Honovich, Uri Shaham, Samuel R. Bowman, Omer Levy
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2205.10782
- **PDF:** [./instruction-induction-honovich-2022.pdf](./instruction-induction-honovich-2022.pdf)
- **Citation Count:** ~200+

### 요약
몇 개의 입력-출력 예시만 보여주고 LLM에게 "이 예시들을 설명하는 지시문(instruction)"을 직접 생성하게 하는 Instruction Induction 과제를 제안합니다. 24개 태스크 데이터셋과, 생성된 지시문만으로 zero-shot 수행 성능을 재는 "실행 기반(execution accuracy)" 평가 지표를 정의했습니다. InstructGPT는 사람이 쓴 지시문 성능의 약 65% 수준까지 도달했습니다.

### 핵심 기여
- "예시 → 자연어 지시문" 역추론을 독립 과제로 정식화
- 지시문의 품질을 문자열 유사도가 아닌 '실행 결과'로 평가하는 지표 제안
- 지시 튜닝된 모델(InstructGPT)에서만 이 능력이 뚜렷하게 나타남을 발견

### 이 논문이 중요한 이유
APE(Zhou et al., 2022)와 OPRO 등 "LLM이 프롬프트를 제안하고, 점수로 고른다"는 이산 프롬프트 최적화의 직접적인 출발점입니다. APE의 후보 생성 단계가 사실상 이 논문의 Instruction Induction이고, 실행 기반 평가 역시 이후 연구들의 표준이 되었습니다.

### 사전 지식
- In-context learning과 few-shot 프롬프팅
- Instruction tuning (FLAN, InstructGPT)
- 평가 지표 설계 (정확도, BERTScore 등의 한계)

### 관련 논문
- [Large Language Models Are Human-Level Prompt Engineers / APE (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [Training language models to follow instructions / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

### 실무 적용
좋은 응답 예시(골든셋) 10~20개만 있으면 LLM에게 시스템 프롬프트 초안을 역으로 뽑게 하고, 그 초안을 골든셋에 실행해 점수가 높은 것을 채택하는 "프롬프트 역설계" 파이프라인을 만들 수 있습니다. PM이 프롬프트를 직접 쓰기보다 "좋은 결과 예시"를 정의하는 데 집중하는 워크플로우의 이론적 근거가 됩니다.

---

## Paper 3 (Recent): Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs
- **Authors:** Ching-An Cheng, Allen Nie, Adith Swaminathan
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2406.16218
- **PDF:** [./trace-next-autodiff-cheng-2024.pdf](./trace-next-autodiff-cheng-2024.pdf)
- **Code:** https://github.com/microsoft/Trace
- **Citation Count:** ~100+

### 요약
프롬프트, 코드, 하이퍼파라미터 등 미분 불가능한 이질적 파라미터로 구성된 AI 워크플로우 전체를 end-to-end로 최적화하는 프레임워크 Trace를 제안합니다. 그래디언트 대신 "실행 트레이스(중간 결과와 그 사용 관계를 기록한 계산 그래프)"를 역전파하고, LLM 기반 옵티마이저 OptoPrime이 이 트레이스와 풍부한 피드백(에러 메시지, 자연어 평가 등)을 해석해 파라미터를 갱신합니다. 프롬프트 최적화, 하이퍼파라미터 튜닝, 로봇 제어 코드 설계 등에서 효과를 보였습니다.

### 핵심 기여
- OPTO(Optimization with Trace Oracle)라는 생성형 최적화 문제 정식화
- PyTorch처럼 쓰는 Python 라이브러리 (`@bundle`, `node`) — 기존 코드에 데코레이터만 붙여 최적화 대상으로 지정
- 단일 프롬프트가 아닌 멀티스텝 에이전트 워크플로우 전체를 동시에 최적화 (TextGrad 대비 효율적)

### 이 논문이 중요한 이유
자동 프롬프트 최적화가 "프롬프트 한 줄 고치기"에서 "에이전트 시스템 전체 컴파일"로 확장되는 흐름(DSPy → TextGrad → Trace → GEPA)을 보여줍니다. Agentic AI를 만드는 엔지니어에게 '프롬프트 + 코드 + 툴 설정'을 하나의 학습 가능한 시스템으로 보는 관점을 제공합니다.

### 사전 지식
- 자동미분(AutoDiff)과 계산 그래프 개념
- TextGrad, DSPy의 "텍스트 그래디언트" / "프롬프트 컴파일" 아이디어
- LLM 에이전트 워크플로우 (ReAct, 툴 호출)

### 관련 논문
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (Agrawal et al., 2025)](https://arxiv.org/abs/2507.19457)

### 실무 적용
고객지원 에이전트처럼 "분류 프롬프트 → 검색 쿼리 생성 → 답변 생성 → 툴 호출 코드"가 얽힌 파이프라인에서, 최종 결과에 대한 피드백(CSAT, 에러 로그, 평가자 코멘트)을 한 번에 역전파해 어느 단계를 고쳐야 할지 LLM이 스스로 판단하고 수정하게 할 수 있습니다. 평가셋과 피드백 함수만 잘 정의하면 프롬프트 튜닝의 수작업 반복을 크게 줄일 수 있습니다.

---

## 추천 읽기 순서
1. **Instruction Induction (2022)** — "LLM이 예시로부터 지시문을 만들 수 있다"는 가장 직관적인 아이디어로 시작
2. **GPT Understands, Too / P-Tuning (2021)** — 대조적으로 "텍스트가 아닌 임베딩 공간에서 프롬프트를 학습"하는 연속 최적화 관점 이해
3. **Trace (2024)** — 두 관점을 넘어, 프롬프트를 포함한 워크플로우 전체를 LLM이 최적화하는 최신 패러다임으로 확장

## 핵심 테이크어웨이
- 자동 프롬프트 최적화는 크게 **연속 공간 튜닝**(P-Tuning 계열, 모델 접근 필요)과 **이산 텍스트 탐색**(Instruction Induction → APE → OPRO, API만으로 가능) 두 갈래로 발전했습니다.
- 어떤 방식이든 핵심은 **"실행 기반 평가"** — 프롬프트의 좋고 나쁨은 실제로 돌려본 결과로만 판단할 수 있습니다. 좋은 평가셋이 곧 좋은 최적화입니다.
- 최신 흐름은 단일 프롬프트가 아닌 **에이전트 시스템 전체를 최적화 대상**으로 봅니다. 풍부한 피드백(에러, 트레이스, 자연어 비평)이 스칼라 점수보다 훨씬 효율적인 학습 신호가 됩니다.

## 다음 토픽과의 연결
다음 모듈 **"LLM Application Frameworks and Orchestration"**에서는 LangChain, LangGraph, 컴파운드 AI 시스템처럼 여러 LLM 호출과 툴을 엮는 방법을 다룹니다. 오늘 본 Trace/DSPy는 바로 그런 오케스트레이션 파이프라인을 "최적화 가능한 프로그램"으로 다루는 관점이므로, 프레임워크를 설계할 때 각 단계의 프롬프트와 파라미터를 나중에 자동 튜닝할 수 있도록 모듈화하는 것이 중요하다는 연결고리를 기억해 두세요.
