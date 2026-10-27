# Daily AI Paper Recommendations

> **Date:** 2026-10-28
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

> 이번 사이클에서는 이전 사이클에서 다룬 APE, Prompt Tuning(Lester), AutoPrompt, Prefix-Tuning, OPRO, RLPrompt, APO, DSPy, GrIPS, Promptbreeder, EvoPrompt, Instruction Induction, P-Tuning(v1), LM-BFF 등과 겹치지 않도록 새로운 논문 3편을 골랐습니다.
> 관점: **"모델 내부(gradient)에 접근할 수 있는가?"** 와 **"최적화 대상이 프롬프트 한 줄인가, 워크플로우 전체인가?"**

---

## Paper 1 (Classic): P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
- **Authors:** Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang
- **Year:** 2021 (ACL 2022)
- **arXiv:** https://arxiv.org/abs/2110.07602
- **PDF:** [./p-tuning-v2-liu-2021.pdf](./p-tuning-v2-liu-2021.pdf)
- **Citation Count:** ~2,000+ (approximate)

### 요약
기존 Prompt Tuning(Lester et al.)은 10B 이상 대형 모델에서만 Fine-tuning에 근접했고, 작은 모델이나 시퀀스 라벨링(NER 등) 같은 어려운 태스크에서는 성능이 크게 떨어졌습니다. P-Tuning v2는 입력층뿐 아니라 **모든 Transformer 레이어에 학습 가능한 연속 프롬프트(deep prompt)** 를 넣어, 300M~10B 전 규모와 다양한 NLU 태스크에서 전체 파라미터의 0.1~3%만 학습하고도 Fine-tuning과 비슷한 성능을 냅니다.

### 핵심 기여
- **Deep Prompt Tuning:** 입력 임베딩 한 곳이 아닌 모든 레이어에 prefix 토큰을 추가 → 학습 가능 파라미터와 표현력을 동시에 확보
- **"규모·태스크 보편성" 입증:** 작은 모델(BERT/RoBERTa급)과 어려운 시퀀스 태스크(NER, extractive QA, SRL)에서도 Fine-tuning과 대등
- **실전 레시피 정리:** reparameterization(MLP) 유무, 프롬프트 길이, verbalizer 대신 CLS/토큰 분류 헤드 사용, 멀티태스크 학습 등 구현 선택지를 실험으로 정리

### 이 논문이 중요한 이유
"프롬프트 최적화"에는 **이산(텍스트) 프롬프트** 와 **연속(soft) 프롬프트** 두 갈래가 있습니다. P-Tuning v2는 연속 프롬프트 계열을 "대형 모델에서만 되는 트릭"에서 "범용 PEFT 기법"으로 끌어올린 논문이며, 이후 LoRA 등과 함께 PEFT 라이브러리의 표준 옵션이 되었습니다. 모델 가중치에 접근할 수 있을 때 "프롬프트를 학습한다"는 것이 무엇을 의미하는지 이해하는 기준점입니다.

### 사전 지식
- Transformer의 Self-Attention 구조와 Key/Value 개념
- Prompt Tuning(Lester et al., 2021), Prefix-Tuning(Li & Liang, 2021)의 차이
- Fine-tuning vs Parameter-Efficient Fine-Tuning(PEFT)의 트레이드오프

### 관련 논문
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)
- [GPT Understands, Too / P-Tuning v1 (Liu et al., 2021)](https://arxiv.org/abs/2103.10385)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

### 실무 적용
- **멀티 테넌트 SaaS:** 하나의 베이스 모델 위에 고객사/태스크별로 수 MB 수준의 프롬프트 파라미터만 저장해 교체 → 서빙 비용 절감
- **온프레미스 소형 모델:** 사내 분류·NER·의도 인식처럼 GPU 예산이 작은 환경에서 Full Fine-tuning 대안
- **HuggingFace PEFT** 의 `PrefixTuningConfig` 등으로 바로 실험 가능 — LoRA와 A/B로 비교해보는 것을 권장

---

## Paper 2 (Classic): Black-Box Tuning for Language-Model-as-a-Service
- **Authors:** Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, Xipeng Qiu
- **Year:** 2022 (ICML 2022)
- **arXiv:** https://arxiv.org/abs/2201.03514
- **PDF:** [./black-box-tuning-lmaas-sun-2022.pdf](./black-box-tuning-lmaas-sun-2022.pdf)
- **Citation Count:** ~400+ (approximate)

### 요약
LLM이 API(LMaaS)로만 제공되면 gradient를 쓸 수 없어 Prompt Tuning이 불가능합니다. BBT는 연속 프롬프트를 **무작위로 생성한 저차원 부분공간(subspace)** 에 투영하고, 그 안에서 **CMA-ES 같은 미분 없는(derivative-free) 최적화** 로 API의 출력(loss/logit)만 보며 프롬프트를 튜닝합니다. 소수 샘플 환경에서 수동 프롬프트와 GPT-3 in-context learning을 능가하고, 경사 기반 Prompt Tuning과 Full Fine-tuning에도 견줄 만한 결과를 보였습니다.

### 핵심 기여
- **LMaaS 문제 정의:** "모델은 API 뒤에 있고, 사용자는 입력·출력만 볼 수 있다"는 현실적 시나리오를 최적화 문제로 공식화
- **Intrinsic Dimensionality 활용:** 대형 사전학습 모델은 낮은 내재 차원을 가진다는 관찰에 기반해 수만 차원 프롬프트를 수백 차원 공간에서 탐색
- **Gradient-free 최적화:** CMA-ES로 forward pass만으로 학습 → 서버 측 비용·보안 부담 없이 사용자 측 튜닝 가능

### 이 논문이 중요한 이유
오늘날 대부분의 AI 엔지니어는 GPT·Claude·Gemini를 **API로만** 사용합니다. "가중치도, gradient도 없을 때 어떻게 최적화할 것인가?"라는 질문은 이후 OPRO, TextGrad, DSPy, GEPA 같은 **블랙박스 프롬프트 최적화** 흐름 전체의 출발 문제입니다. BBT는 이를 "연속 공간 + 진화 전략"으로 푼 대표 고전으로, 텍스트 기반 최적화와 비교하며 읽으면 설계 공간이 선명해집니다.

### 사전 지식
- Prompt Tuning(연속 프롬프트)의 기본 개념
- 진화 전략(Evolution Strategies), 특히 CMA-ES의 직관
- Intrinsic Dimension(Aghajanyan et al., 2020) 개념

### 관련 논문
- [BBTv2: Towards a Gradient-Free Future with Large Language Models (Sun et al., 2022)](https://arxiv.org/abs/2205.11200)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan et al., 2020)](https://arxiv.org/abs/2012.13255)
- [RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning (Deng et al., 2022)](https://arxiv.org/abs/2205.12548)
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)

### 실무 적용
- **API 기반 제품의 튜닝 전략 설계:** 가중치 접근이 없을 때 "평가셋 + 블랙박스 탐색" 루프가 기본 패턴임을 이해
- **비용 관점:** forward-only 최적화는 쿼리 수 = 비용. 평가셋 크기·탐색 예산을 KPI처럼 관리해야 함
- 단, 현재 상용 API는 logit 접근이 제한적이므로, 실무에서는 BBT 아이디어를 **텍스트 프롬프트 탐색(OPRO/GEPA류)** 으로 옮겨 적용하는 경우가 많음

---

## Paper 3 (Recent): AFlow: Automating Agentic Workflow Generation
- **Authors:** Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, et al. (MetaGPT team)
- **Year:** 2024 (ICLR 2025 Oral)
- **arXiv:** https://arxiv.org/abs/2410.10762
- **PDF:** [./aflow-agentic-workflow-generation-zhang-2024.pdf](./aflow-agentic-workflow-generation-zhang-2024.pdf)
- **Citation Count:** ~200+ (approximate)

### 요약
AFlow는 최적화 대상을 "프롬프트 한 개"에서 **LLM 호출 노드와 연결(edge)로 이루어진 코드 형태의 워크플로우 전체** 로 확장합니다. 워크플로우 설계를 탐색 문제로 보고, **Monte Carlo Tree Search(MCTS)** 로 코드 수정 → 실행 피드백 → 트리 구조 경험 축적을 반복하며 워크플로우를 개선합니다. 6개 벤치마크(HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP)에서 기존 SOTA 대비 평균 5.7% 향상, 작은 모델이 GPT-4o 대비 4.55% 비용으로 특정 태스크에서 더 나은 성능을 보였습니다.

### 핵심 기여
- **Workflow-as-Code 탐색 공간:** 노드(프롬프트·모델·파라미터)와 오퍼레이터(Ensemble, Review & Revise, Programmer 등)를 코드로 표현해 프롬프트와 구조를 함께 최적화
- **MCTS 기반 최적화:** LLM이 워크플로우 코드를 수정(expansion)하고, 실행 점수로 평가(backprop)하며, 과거 수정 경험을 트리로 재사용
- **비용-성능 프런티어:** 자동 생성된 워크플로우가 작은/저렴한 모델의 성능을 끌어올려 비용 대비 효율을 개선

### 이 논문이 중요한 이유
DSPy·TextGrad가 "프롬프트를 컴파일/미분한다"는 관점이었다면, AFlow는 **"에이전트 시스템의 구조 자체를 자동 설계한다"** 는 다음 단계를 보여줍니다. Agentic AI 제품에서 실제 병목은 단일 프롬프트보다 "어떤 단계를 몇 번, 어떤 검증과 함께 호출하느냐"인 경우가 많습니다. 프롬프트 최적화 → 워크플로우 최적화로 넘어가는 브리지 논문이자, 다음 모듈(LLM Orchestration)과 직접 연결됩니다.

### 사전 지식
- Monte Carlo Tree Search의 Selection / Expansion / Evaluation / Backpropagation 단계
- DSPy의 모듈·컴파일 개념, 또는 OPRO/TextGrad의 "LLM as optimizer" 개념
- Self-Consistency, Self-Refine 같은 기본 추론 패턴(오퍼레이터로 사용됨)

### 관련 논문
- [Automated Design of Agentic Systems / ADAS (Hu et al., 2024)](https://arxiv.org/abs/2408.08435)
- [GPTSwarm: Language Agents as Optimizable Graphs (Zhuge et al., 2024)](https://arxiv.org/abs/2402.16823)
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (Hong et al., 2023)](https://arxiv.org/abs/2308.00352)

### 실무 적용
- **평가셋이 먼저:** AFlow류 자동화는 "점수를 낼 수 있는 태스크"에서만 작동 → 제품팀은 골든셋과 자동 채점기부터 확보해야 함
- **모델 다운그레이드 전략:** 상위 모델 1회 호출 대신 하위 모델 + 앙상블/리뷰 구조로 비용을 낮추는 실험 설계에 활용
- **오픈소스:** MetaGPT 저장소의 AFlow 구현으로 사내 태스크(코드 생성, QA 등)에 바로 실험 가능

---

## 추천 읽기 순서
1. **P-Tuning v2** — 가중치에 접근할 수 있을 때의 "학습 가능한 프롬프트" 기준선을 먼저 잡습니다. (6페이지로 짧음)
2. **Black-Box Tuning** — 같은 연속 프롬프트를 gradient 없이 최적화하면 무엇이 달라지는지 비교합니다.
3. **AFlow** — 최적화 대상을 프롬프트에서 워크플로우 전체로 넓혀, 블랙박스 최적화가 에이전트 설계로 확장되는 모습을 봅니다.

## 핵심 테이크어웨이
- **Q. 프롬프트 최적화의 첫 번째 분기점은?** → "gradient 접근 가능 여부". 가능하면 P-Tuning v2/LoRA 같은 학습형, 불가능하면 BBT/OPRO/AFlow 같은 탐색형.
- **Q. 블랙박스 최적화의 비용 구조는?** → 탐색 = API 호출. 탐색 공간을 줄이는 것(BBT의 저차원 subspace, AFlow의 오퍼레이터)이 핵심 설계 포인트.
- **Q. 최적화 단위는 어디까지 커질 수 있나?** → 토큰 임베딩 → 텍스트 지시문 → 프롬프트 파이프라인(DSPy) → 워크플로우 구조(AFlow). 단위가 커질수록 평가 함수의 품질이 성패를 가릅니다.
- **가설 검증용 질문:** "우리 제품의 품질 병목은 프롬프트 문구인가, 호출 구조인가?" — 이를 구분하는 실험을 먼저 설계해보세요.

## 다음 토픽과의 연결
다음 토픽은 **Module 8: LLM Application Frameworks and Orchestration** 입니다. AFlow가 자동으로 찾아낸 "노드 + 오퍼레이터 + 연결"은 곧 LangChain/LangGraph에서 사람이 손으로 설계하는 체인·그래프와 같은 개념입니다. Toolformer와 MRKL을 읽을 때, "LLM이 외부 도구·모듈을 언제, 어떻게 호출하도록 구성할 것인가"를 오늘의 워크플로우 최적화 관점과 연결해보세요.
