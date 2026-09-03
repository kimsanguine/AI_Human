# Daily AI Paper Recommendations

> **Date:** 2026-09-04
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): Making Pre-trained Language Models Better Few-shot Learners (LM-BFF)
- **Authors:** Tianyu Gao, Adam Fisch, Danqi Chen
- **Year:** 2021 (ACL 2021, arXiv 2020-12)
- **arXiv:** https://arxiv.org/abs/2012.15723
- **PDF:** [./lm-bff-few-shot-learners-gao-2021.pdf](./lm-bff-few-shot-learners-gao-2021.pdf)
- **Citation Count:** ~2,700 (approximate)

### 요약
GPT-3가 거대한 파라미터 규모로만 달성했던 few-shot 성능을, 작은 언어모델(RoBERTa-large)에 프롬프트 기반 파인튜닝을 적용해 재현할 수 있음을 보인 논문이다. 핵심은 사람이 템플릿을 손으로 짜는 대신, T5를 이용해 템플릿(template)과 레이블 워드(label word)를 **자동 생성**하고 검증셋으로 후보를 선별하는 파이프라인이다. 여기에 입력 문맥에 demonstration을 동적으로 선택해 삽입하는 전략을 결합해, 표준 파인튜닝 대비 최대 30%까지 성능을 끌어올렸다.

### 핵심 기여
- 프롬프트를 "탐색 가능한 검색 공간"으로 정의하고, 생성 모델(T5)로 템플릿 후보군을 자동 생성한 뒤 dev set 점수로 선택하는 automatic prompt generation 파이프라인 제시
- 레이블 워드(verbalizer)까지 자동 탐색 대상으로 확장 — 프롬프트 최적화가 템플릿만의 문제가 아님을 실증
- SBERT 임베딩 유사도 기반으로 입력마다 다른 demonstration을 골라 넣는 dynamic demonstration selection 제안
- few-shot 평가에서 시드/데이터 분할에 따른 분산이 매우 크다는 점을 체계적으로 보고 — 이후 프롬프트 연구의 평가 프로토콜 기준이 됨

### 이 논문이 중요한 이유
"프롬프트는 사람이 감으로 쓰는 것"이라는 통념을 깨고, **프롬프트 = 최적화 대상**이라는 프레임을 처음으로 실용적인 파이프라인 형태로 정착시킨 논문이다. 오늘날 DSPy, MIPRO, GEPA 같은 automatic prompt optimizer가 공유하는 3단 구조 — (1) 후보 생성 → (2) 데이터셋으로 평가 → (3) 최고 후보 선택 — 가 여기서 온다. AI 엔지니어 입장에서는 "프롬프트도 CI로 회귀 테스트하고 스코어로 관리해야 한다"는 실무 원칙의 출발점이며, few-shot 성능 분산 보고는 지금도 프롬프트 A/B 테스트 설계 시 반드시 참고해야 할 내용이다.

### 사전 지식
- MLM(Masked Language Model)과 cloze-style 태스크 변환 개념
- 파인튜닝 vs 프롬프트 기반 파인튜닝의 차이 (head를 새로 붙이는가, LM head를 재사용하는가)
- verbalizer / label word mapping 개념
- Sentence-BERT 임베딩과 유사도 검색 기초
- GPT-3의 in-context learning (arXiv:2005.14165) 사전 독해 권장

### 관련 논문
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts (Shin et al., 2020)](https://arxiv.org/abs/2010.15980)
- [Exploiting Cloze Questions for Few Shot Text Classification / PET (Schick & Schütze, 2020)](https://arxiv.org/abs/2001.07676)
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)

### 실무 적용
소규모 모델을 파인튜닝해 분류 파이프라인(문의 유형 분류, 콘텐츠 모더레이션, 인텐트 라우팅)을 만들 때 여전히 가장 비용 효율적인 레시피다. 라벨 100~200건 수준의 초기 제품에서 "라벨을 더 모을까, 프롬프트를 더 다듬을까"를 결정할 때 LM-BFF의 분산 보고가 실질적인 판단 근거가 된다. 또한 자동 템플릿 생성 → dev set 스코어링 구조는 그대로 사내 프롬프트 관리 시스템(프롬프트 레지스트리 + 골든셋 평가)의 설계 원형으로 쓸 수 있다.

---

## Paper 2 (Classic): EvoPrompt — Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers
- **Authors:** Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2309.08532
- **PDF:** [./evoprompt-guo-2023.pdf](./evoprompt-guo-2023.pdf)
- **Citation Count:** ~600 (approximate)

### 요약
이산(discrete) 자연어 프롬프트를 진화 알고리즘(GA, DE)으로 최적화하는 프레임워크다. 핵심 아이디어는 "변이(mutation)와 교차(crossover) 연산자를 LLM에게 시키자"는 것 — LLM이 언어적으로 자연스러운 새 프롬프트를 생성하고, 진화 알고리즘이 dev set 점수를 기준으로 개체군(population)을 선택·유지한다. GPT-3.5와 Alpaca에서 31개 데이터셋(언어 이해/생성 + BIG-Bench Hard)을 대상으로 실험해, 사람이 만든 프롬프트 대비 최대 25%, 기존 자동 프롬프트 생성 방법 대비 최대 14% 향상을 보였다.

### 핵심 기여
- gradient도 파라미터 접근도 필요 없는 완전 블랙박스 프롬프트 최적화 — API-only 모델(GPT-3.5 등)에 그대로 적용 가능
- 진화 연산자(GA의 crossover/mutation, DE의 차분 벡터)를 자연어 지시문으로 번역해 LLM이 수행하도록 한 설계
- 개체군 기반 탐색으로 지역 최적해(local optimum)를 회피 — 단일 프롬프트를 그리디하게 개선하는 방식의 한계를 극복
- 31개 데이터셋 규모의 광범위한 벤치마크로 discrete prompt optimization의 일반성을 실증

### 이 논문이 중요한 이유
실무에서 우리가 쓰는 모델 대부분은 gradient에 접근할 수 없는 API 모델이다. EvoPrompt는 그 제약 안에서 "LLM을 옵티마이저로 쓴다"는 패러다임을 가장 명확하게 보여준 논문 중 하나이며, OPRO·Promptbreeder·GEPA로 이어지는 계보의 중심에 있다. 특히 population 유지 + Pareto/스코어 기반 선택이라는 구조는 오늘날 거의 모든 modern prompt optimizer의 공통 골격이다. AI 엔지니어는 이 논문을 통해 "프롬프트 튜닝을 사람의 직관이 아니라 탐색 예산(rollout budget)의 문제로 재정의"하는 사고를 얻게 된다.

### 사전 지식
- 유전 알고리즘(GA)과 차분 진화(DE)의 기본 연산: selection, crossover, mutation
- dev set 기반 fitness 평가와 오버피팅 위험
- discrete prompt vs continuous(soft) prompt의 차이
- BIG-Bench Hard 등 추론 벤치마크의 성격
- LLM 추론 비용 구조 (rollout 수 = 비용)

### 관련 논문
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution (Fernando et al., 2023)](https://arxiv.org/abs/2309.16797)
- [Large Language Models Are Human-Level Prompt Engineers / APE (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)
- [GrIPS: Gradient-free, Edit-based Instruction Search (Prasad et al., 2022)](https://arxiv.org/abs/2203.07281)

### 실무 적용
프롬프트를 손으로 만지는 대신, 골든셋(50~200건)과 자동 평가 지표만 준비하면 야간 배치로 프롬프트를 진화시킬 수 있다. 실제로 콘텐츠 분류·요약 품질·툴 호출 정확도 같은 지표가 명확한 태스크에서, 사람이 한 주 걸려 튜닝하던 성능을 수 시간의 탐색으로 넘어서는 경우가 많다. 단, 개체군 크기 × 세대 수 × 평가셋 크기만큼 API 호출이 곱해지므로 비용 상한과 조기 종료 조건을 반드시 설계해야 한다. 골든셋에 대한 과적합을 막기 위해 별도 홀드아웃 검증은 필수다.

---

## Paper 3 (Recent): GEPA — Reflective Prompt Evolution Can Outperform Reinforcement Learning
- **Authors:** Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2507.19457
- **PDF:** [./gepa-reflective-prompt-evolution-agrawal-2025.pdf](./gepa-reflective-prompt-evolution-agrawal-2025.pdf)
- **Citation Count:** ~200 (approximate, 2025년 논문 기준 매우 빠른 인용 증가)

### 요약
GEPA(Genetic-Pareto)는 자연어 **리플렉션**을 프롬프트 최적화의 1급 시민으로 끌어올린 옵티마이저다. 시스템의 실행 궤적(추론 과정, 툴 호출, 툴 출력, 에러 로그)을 샘플링한 뒤, LLM이 그것을 자연어로 되짚어 "무엇이 왜 실패했는가"를 진단하고 프롬프트 수정안을 제안·검증한다. 여기에 자기 시도들의 Pareto frontier에서 서로 보완적인 교훈을 결합하는 전략을 더했다. 6개 태스크에서 강화학습 기반 GRPO를 평균 6%, 최대 20% 상회하면서 rollout은 최대 35배 적게 썼고, 최강 프롬프트 옵티마이저인 MIPROv2도 10% 이상(AIME-2025에서 +12%) 앞섰다.

### 핵심 기여
- 스칼라 보상(scalar reward) 대신 **자연어 피드백**을 학습 신호로 사용 — 한 번의 롤아웃에서 훨씬 많은 정보를 추출해 샘플 효율을 극적으로 개선
- Pareto frontier 기반 후보 선택: 단일 총점 1등이 아니라 "각 인스턴스에서 최고인 후보들"의 집합을 유지해 다양성 붕괴와 지역 최적해를 동시에 회피
- 멀티 모듈 복합 AI 시스템(compound AI system)의 여러 프롬프트를 동시에 최적화하는 절차 제시
- 프롬프트 최적화가 RL 기반 가중치 업데이트(GRPO)의 실질적 대안이 될 수 있음을 정량적으로 입증 — 35배 적은 rollout으로 더 나은 성능
- 추론 시점(inference-time) 코드 최적화 탐색 전략으로도 유망함을 보임

### 이 논문이 중요한 이유
2025년 현재 "모델을 더 학습시킬 것인가, 프롬프트/컨텍스트를 더 잘 설계할 것인가"는 모든 AI 제품 조직의 핵심 자원 배분 질문이다. GEPA는 이 질문에 대해 "많은 경우 프롬프트 진화가 RL보다 싸고 빠르고 좋다"는 강한 실증적 답을 내놓았다. 특히 **리플렉션이 스칼라 보상보다 정보 밀도가 높다**는 통찰은 에이전트 설계 전반(자기 수정 루프, 평가 피드백 설계, 에러 핸들링)에 그대로 전이된다. Agentic AI를 다루는 엔지니어라면 GEPA는 프롬프트 최적화 논문이 아니라 **에이전트 학습 신호 설계 논문**으로 읽어야 한다.

### 사전 지식
- DSPy의 모듈/시그니처/컴파일 개념과 MIPROv2 옵티마이저 (arXiv:2406.11695)
- GRPO 등 LLM용 정책 최적화 계열 RL 기법의 기본 구조와 rollout 비용
- Pareto optimality / multi-objective optimization 기초
- 복합 AI 시스템(compound AI system)에서의 credit assignment 문제
- 자동 평가 지표(metric) 설계와 자연어 피드백 함수 작성 경험

### 관련 논문
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs / MIPRO (Opsahl-Ong et al., 2024)](https://arxiv.org/abs/2406.11695)
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO 원 논문, Shao et al., 2024)](https://arxiv.org/abs/2402.03300)

### 실무 적용
에이전트 제품에서 "실패 케이스가 쌓이는데 파인튜닝은 부담스럽다"는 전형적 상황의 정답에 가깝다. 실패 로그 + 자연어 피드백 함수를 만들어 두면, GEPA는 그 로그를 읽고 시스템 프롬프트를 스스로 개선한다. rollout이 35배 적다는 것은 곧 "주 단위 파인튜닝 사이클을 시간 단위 프롬프트 사이클로 대체할 수 있다"는 뜻이고, 이는 그로스 관점에서 실험 속도를 한 자릿수 이상 끌어올린다. 실무 체크리스트: (1) 태스크별 자연어 피드백을 반환하는 metric 함수를 먼저 설계할 것, (2) Pareto 후보를 저장해 롤백 가능하게 할 것, (3) 홀드아웃셋으로 골든셋 과적합을 상시 감시할 것.

---

## 추천 읽기 순서

1. **LM-BFF (2021)** — 먼저 "프롬프트를 자동 생성하고 dev set으로 고른다"는 가장 단순한 형태의 최적화 루프를 이해한다. 이후 모든 논문이 이 루프의 변주다.
2. **EvoPrompt (2023)** — 그 루프에 개체군(population)과 진화 연산자를 넣으면 어떻게 탐색력이 커지는지 본다. 블랙박스 API 환경이라는 실무 제약도 함께 체득한다.
3. **GEPA (2025)** — 마지막으로 "점수" 대신 "자연어 리플렉션"을 신호로 쓰면 왜 샘플 효율이 수십 배 좋아지는지 확인한다. RL과의 비교 섹션은 반드시 정독할 것.

시간이 부족하다면 GEPA → LM-BFF 순으로 읽고, EvoPrompt는 Section 3(방법론)만 훑어도 된다.

## 핵심 테이크어웨이

- **프롬프트는 코드가 아니라 파라미터다.** 손으로 고치는 대상이 아니라, 평가셋과 탐색 예산으로 최적화하는 대상이다. 세 논문 모두 이 전제를 공유한다.
- **최적화 루프의 3요소는 항상 같다:** 후보 생성기(generator) → 평가 함수(metric) → 선택 전략(selector). 세 논문의 차이는 오직 이 3요소의 구현이다. LM-BFF는 T5/dev-score/top-k, EvoPrompt는 LLM 진화 연산자/dev-score/population, GEPA는 리플렉션/자연어 피드백/Pareto frontier.
- **학습 신호의 정보 밀도가 샘플 효율을 결정한다.** 스칼라 보상 1개보다 "왜 틀렸는지" 문단 하나가 훨씬 많은 정보를 담는다. GEPA가 GRPO를 35배 적은 rollout으로 이긴 근본 원인이다.
- **다양성 유지가 성능을 만든다.** 단일 최고 후보만 남기면 지역 최적해에 빠진다. EvoPrompt의 population, GEPA의 Pareto frontier는 같은 문제에 대한 서로 다른 처방이다.
- **평가 함수 설계가 곧 제품 전략이다.** 옵티마이저는 metric이 정의한 방향으로만 간다. metric이 얕으면 프롬프트도 얕게 최적화된다 — AI 엔지니어의 진짜 레버리지는 옵티마이저 선택이 아니라 metric 설계에 있다.
- **골든셋 과적합은 상수 리스크다.** 세 논문 모두 dev set을 쓰며, 홀드아웃 검증 없이는 실제 트래픽에서 성능이 무너질 수 있다.

## 다음 토픽과의 연결

다음은 **Module 8: LangChain and LLM Orchestration — LLM Application Frameworks and Orchestration**이다. 오늘 다룬 automatic prompt optimization은 단일 프롬프트를 다듬는 기술처럼 보이지만, GEPA가 명시적으로 다루듯 실제 대상은 여러 모듈이 연결된 **복합 AI 시스템 전체**다. 즉 오늘의 "최적화 대상"이 내일의 "오케스트레이션 구조"와 같은 객체다.

구체적 연결점 세 가지:

- **컴파일 개념의 연속성** — DSPy가 파이프라인을 "컴파일"한다고 말할 때, 그 컴파일러가 하는 일이 바로 오늘 본 프롬프트 최적화다. 내일 프레임워크를 볼 때 "이 프레임워크는 최적화 가능한 형태로 파이프라인을 표현하고 있는가"를 기준으로 평가하면 좋다.
- **credit assignment** — 오케스트레이션이 복잡해질수록 "어느 모듈의 프롬프트가 문제였는가"를 특정하기 어려워진다. GEPA의 궤적 기반 리플렉션은 이 문제에 대한 한 가지 답이며, 내일 볼 Toolformer/MRKL 계열의 모듈 분해 설계와 짝을 이룬다.
- **관측 가능성(observability)** — 프롬프트를 자동 최적화하려면 실행 궤적을 남겨야 한다. 이는 곧 오케스트레이션 레이어가 트레이싱을 1급으로 지원해야 한다는 요구로 이어진다. 내일 프레임워크를 비교할 때 트레이싱/로깅 설계를 반드시 함께 볼 것.
