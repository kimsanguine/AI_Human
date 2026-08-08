# Daily AI Paper Recommendations

> **Date:** 2026-08-08
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models
- **Authors:** Archiki Prasad, Peter Hase, Xiang Zhou, Mohit Bansal
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2203.07281
- **PDF:** [./grips-gradient-free-instruction-search-prasad-2022.pdf](./grips-gradient-free-instruction-search-prasad-2022.pdf)
- **Citation Count:** ~400+

### 요약
GrIPS는 그래디언트(gradient) 접근 없이 자연어 명령어(instruction)를 자동으로 편집·탐색하여 개선하는 방법이다. 사람이 작성한 명령어를 입력으로 받아 구(phrase) 단위로 삭제/추가/교환/패러프레이즈 연산을 반복 적용하며 더 나은 프롬프트를 탐색한다. 모델 내부 파라미터에 접근하지 않으므로 API만 제공되는 블랙박스 LLM에도 그대로 적용할 수 있다.

### 핵심 기여
- 역전파·그래디언트 없이 명령어 텍스트 편집만으로 성능을 개선하여 블랙박스(API) LLM에 적용 가능함을 입증
- 구 단위 4가지 편집 연산(delete, add, swap, paraphrase)과 이를 조합한 local search 탐색 알고리즘 제안
- Natural-Instructions 8개 분류 태스크에서 InstructGPT 기준 평균 최대 4.30%p 개선, 수작업 재작성 및 예시 기반 프롬프트를 모두 능가

### 이 논문이 중요한 이유
프롬프트 최적화가 반드시 모델 내부 접근(soft prompt tuning, 파인튜닝)을 요구하지 않으며, 오직 텍스트 편집만으로도 자동화가 가능함을 보여준 초기 대표작이다. 상용 폐쇄형 LLM API가 주류가 된 환경에서 "가중치를 못 건드려도 프롬프트는 최적화할 수 있다"는 실용적 방향을 제시했다.

### 사전 지식
- In-context learning과 instruction(지시문)의 개념
- Discrete prompt vs. continuous(soft) prompt의 차이
- Local search 등 기초 탐색 알고리즘

### 관련 논문
- [Large Language Models Are Human-Level Prompt Engineers / APE (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts (Shin et al., 2020)](https://arxiv.org/abs/2010.15980)

### 실무 적용
파인튜닝이 불가능한 상용 API 환경에서 시스템 프롬프트/지시문을 자동으로 개선한다. 평가셋을 기준으로 명령어를 편집·검증하는 프롬프트 A/B 자동화 파이프라인의 원형으로 활용할 수 있다.

---

## Paper 2 (Classic): Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
- **Authors:** Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, Tim Rocktäschel
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2309.16797
- **PDF:** [./promptbreeder-self-referential-self-improvement-fernando-2023.pdf](./promptbreeder-self-referential-self-improvement-fernando-2023.pdf)
- **Citation Count:** ~500+

### 요약
Promptbreeder(DeepMind)는 진화 알고리즘을 이용해 프롬프트를 자기참조적(self-referential)으로 개선하는 범용 자기개선 메커니즘이다. task-prompt 집단을 변이(mutation)시켜 훈련셋에 대한 적합도(fitness)로 평가하고, 이때 변이를 만들어내는 mutation-prompt 자체도 함께 진화시키는 것이 핵심이다. 즉 "프롬프트를 개선하는 프롬프트"까지 동시에 최적화한다.

### 핵심 기여
- LLM을 변이 연산자로 활용하는 진화적(Evolutionary) 프롬프트 최적화 프레임워크 제안
- task-prompt뿐 아니라 그것을 개선하는 mutation-prompt까지 진화시키는 self-referential 구조
- 산술·상식추론 벤치마크에서 Chain-of-Thought, Plan-and-Solve 등 SOTA 프롬프트 전략을 상회

### 이 논문이 중요한 이유
"프롬프트를 개선하는 프롬프트"라는 메타 최적화 개념을 구체적인 알고리즘으로 구현했다. 인간이 설계한 고정된 최적화 규칙에 의존하지 않고 최적화 과정 자체를 스스로 개선한다는 점에서, 자기개선(self-improvement) 에이전트 설계의 사상적 토대가 된다.

### 사전 지식
- 진화 알고리즘(유전 알고리즘: 개체군, mutation, selection, fitness)
- Chain-of-Thought 등 기본 프롬프트 전략
- Self-improvement / meta-learning의 개념

### 관련 논문
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [Large Language Models Are Human-Level Prompt Engineers / APE (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)

### 실무 적용
도메인 특화 프롬프트를 소규모 평가셋만으로 자동 진화시켜 프롬프트 풀(pool)을 지속적으로 유지·개선한다. 반복적으로 프롬프트를 재생성해야 하는 에이전트 파이프라인에서 자기개선 루프의 설계 참고 사례가 된다.

---

## Paper 3 (Recent): PromptWizard: Task-Aware Prompt Optimization Framework
- **Authors:** Eshaan Agarwal, Joykirat Singh, Vivek Dani, Raghav Magazine, Tanuja Ganu, Akshay Nambi
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2405.18369
- **PDF:** [./promptwizard-task-aware-prompt-optimization-agarwal-2024.pdf](./promptwizard-task-aware-prompt-optimization-agarwal-2024.pdf)
- **Citation Count:** ~150+

### 요약
PromptWizard(Microsoft)는 LLM이 스스로 프롬프트를 생성(generate)·비평(critique)·정제(refine)하는 self-evolving 프레임워크다. 명령어(instruction)와 in-context 예시를 함께 반복 최적화하여 사람이 읽을 수 있는 태스크 특화 프롬프트를 만든다. 45개 태스크에서 우수한 성능을 보이면서도 API 호출·토큰 사용량·전체 비용을 크게 절감한다.

### 핵심 기여
- feedback 기반 self-critique와 synthesis로 명령어와 in-context 예시를 동시에 최적화
- exploration(탐색)과 exploitation(활용)의 균형을 맞춘 self-adaptive 메커니즘, 소량 데이터·작은 LLM에서도 강건
- 기존 프롬프트 최적화 대비 API 호출/토큰/비용을 대폭 절감하여 실무 확장성 확보

### 이 논문이 중요한 이유
프롬프트 최적화를 "성능"뿐 아니라 "비용 효율" 관점에서 실무화한 최신 대표작이다. critique-refine 루프라는 에이전트형 접근을 통해, 작은 모델에 최적화된 프롬프트를 붙여도 큰 모델에 준하는 성능을 훨씬 낮은 비용으로 달성할 수 있음을 보여준다.

### 사전 지식
- self-critique / feedback loop 개념
- in-context example 선택과 그 영향
- 앞선 APE·OPRO 계열의 discrete prompt 최적화 흐름

### 관련 논문
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)

### 실무 적용
프로덕션 프롬프트를 저비용으로 자동 튜닝하는 데 직접 적용할 수 있다. 작은 모델 + 최적화된 프롬프트 조합으로 서빙 비용을 낮추거나, 태스크별 프롬프트를 자동 생성하는 내부 툴을 구축하는 데 활용된다.

---

## 추천 읽기 순서
1. **GrIPS (2022)** — 그래디언트 없이 텍스트 편집만으로 프롬프트를 최적화하는 가장 기본적인 아이디어부터 이해한다.
2. **Promptbreeder (2023)** — 편집을 넘어 "최적화 과정 자체를 진화시키는" 메타 최적화 개념으로 확장한다.
3. **PromptWizard (2024)** — 앞의 아이디어들을 critique-refine 에이전트 루프로 정리하고, 비용 효율까지 고려한 최신 실무형 프레임워크로 마무리한다.

## 핵심 테이크어웨이
- 자동 프롬프트 최적화의 핵심 질문은 "어떻게 탐색할 것인가"이다: 편집 기반 탐색(GrIPS) → 진화 기반 탐색(Promptbreeder) → self-critique 기반 정제(PromptWizard)로 발전해 왔다.
- 세 논문 모두 모델 가중치에 접근하지 않는 블랙박스/discrete 최적화 계열로, 상용 API 시대에 특히 실용적이다.
- 최신 흐름은 단순 성능 향상을 넘어 **API 비용·토큰 효율**과 **명령어+예시의 동시 최적화**로 이동하고 있다.
- 명령어를 개선하는 주체가 LLM 자신이라는 점에서, 프롬프트 최적화는 자기개선 에이전트 설계와 직접 맞닿아 있다.

## 다음 토픽과의 연결
프롬프트 최적화는 개별 호출을 넘어 여러 LLM 호출을 조합·오케스트레이션하는 단계로 이어진다. 다음 모듈(LangChain and LLM Orchestration)에서는 이렇게 최적화된 프롬프트/모듈을 도구 사용(tool use)과 파이프라인으로 엮어 복합 AI 시스템을 구성하는 방법을 다룬다. 특히 DSPy 계열의 "프롬프트 컴파일러" 사고가 오케스트레이션과 자연스럽게 연결된다.
