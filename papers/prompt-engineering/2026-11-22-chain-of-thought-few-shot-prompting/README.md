# Daily AI Paper Recommendations

> **Date:** 2026-11-22
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)
- **Authors:** Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis
- **Year:** 2022 (Findings of EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2210.03350
- **PDF:** [./self-ask-compositionality-gap-press-2022.pdf](./self-ask-compositionality-gap-press-2022.pdf)
- **Citation Count:** ~900+

### 요약
LLM이 개별 사실(A, B)은 맞히지만 둘을 조합해야 하는 멀티홉 질문에서는 자주 틀리는 현상을 "구성성 격차(compositionality gap)"로 정의하고 측정했습니다. 모델 크기를 키워도 단일 사실 정확도는 오르지만 이 격차는 줄지 않는다는 점을 보였습니다. 해결책으로, 모델이 스스로 후속 질문(Follow up)을 던지고 답하는 few-shot 프롬프트 "Self-Ask"를 제안하고, 후속 질문을 검색엔진에 연결해 정확도를 더 끌어올렸습니다.

### 핵심 기여
- 멀티홉 추론 능력을 정량화하는 지표(compositionality gap)와 Compositional Celebrities 데이터셋 제안
- 스케일링만으로는 "아는 것을 조합하는 능력"이 해결되지 않음을 실증
- 명시적 하위 질문 분해 프롬프트(Self-Ask) + 검색 도구 연결(Self-Ask + Search Engine)로 CoT 대비 성능 개선

### 이 논문이 중요한 이유
CoT의 "생각을 쓰게 한다"를 "질문을 쪼개게 한다"로 구조화한 대표 논문입니다. 하위 질문이 명시적 텍스트로 나오기 때문에 그 지점에 도구(검색)를 꽂을 수 있다는 아이디어가 이후 ReAct, 에이전트형 RAG로 직결됩니다. 프롬프트 설계가 곧 "도구 호출 인터페이스 설계"가 되는 출발점입니다.

### 사전 지식
Chain-of-Thought 프롬프팅, few-shot in-context learning, 멀티홉 QA(HotpotQA, 2WikiMultiHopQA) 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models (Zhou et al., 2022)](https://arxiv.org/abs/2205.10625)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

### 실무 적용
리서치 에이전트, 사내 지식 QA 봇에서 "질문 분해 → 하위 질문별 검색 → 종합" 파이프라인의 원형입니다. 하위 질문 로그가 남아 디버깅과 평가(어느 홉에서 틀렸는지)가 쉬워지는 것도 운영상 큰 장점입니다.

---

## Paper 2 (Classic): PAL: Program-aided Language Models
- **Authors:** Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig
- **Year:** 2022 (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2211.10435
- **PDF:** [./pal-program-aided-lm-gao-2022.pdf](./pal-program-aided-lm-gao-2022.pdf)
- **Citation Count:** ~1,500+

### 요약
CoT에서 LLM은 문제를 잘 분해하지만 계산·논리 단계에서 자주 실수한다는 관찰에서 출발합니다. PAL은 추론 단계를 자연어 대신 Python 코드로 작성하게 하고, 실제 풀이(실행)는 인터프리터에 맡깁니다. 13개 수학·기호·알고리즘 과제에서 효과를 보였고, GSM8K에서 CoT를 쓴 PaLM-540B를 top-1 기준 약 15%p 앞섰습니다.

### 핵심 기여
- "분해는 LLM, 실행은 런타임"이라는 역할 분리 패러다임 제시
- few-shot 예시를 코드 형태의 추론으로 바꾸는 것만으로 산술 오류를 크게 감소
- 숫자가 큰/변형된 문제(GSM-Hard)에서 CoT 대비 견고함을 입증

### 이 논문이 중요한 이유
"LLM이 못하는 건 도구에 넘긴다"는 설계 원칙을 프롬프트 수준에서 증명한 논문입니다. 오늘날 Code Interpreter, 데이터 분석 에이전트, 툴 콜링의 사고방식이 여기서 출발합니다. 어떤 단계를 모델에, 어떤 단계를 결정적 시스템에 맡길지 판단하는 감각을 기를 수 있습니다.

### 사전 지식
Chain-of-Thought, few-shot 프롬프팅, 코드 생성 모델(Codex) 기본 개념, 간단한 Python

### 관련 논문
- [Program of Thoughts Prompting (Chen et al., 2022)](https://arxiv.org/abs/2211.12588)
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [Training Verifiers to Solve Math Word Problems / GSM8K (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)

### 실무 적용
가격·견적 계산, 재무 리포트 수치 집계, 스프레드시트/SQL 분석 등 "숫자가 틀리면 안 되는" 기능은 모델이 답을 직접 쓰게 하지 말고 코드를 생성·실행하게 설계해야 합니다. 샌드박스 실행 환경과 실행 결과 검증 루프가 함께 필요합니다.

---

## Paper 3 (Recent): s1: Simple test-time scaling
- **Authors:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
- **Year:** 2025 (EMNLP 2025)
- **arXiv:** https://arxiv.org/abs/2501.19393
- **PDF:** [./s1-simple-test-time-scaling-muennighoff-2025.pdf](./s1-simple-test-time-scaling-muennighoff-2025.pdf)
- **Citation Count:** ~700+ (2025년 발표 이후 급증)

### 요약
테스트 타임 스케일링(추론 시 더 오래 생각하게 해 성능을 올리는 것)의 가장 단순한 레시피를 찾는 연구입니다. 난이도·다양성·품질로 엄선한 1,000개의 추론 트레이스(s1K)로 Qwen2.5-32B-Instruct를 SFT하고, 생각을 강제로 끊거나 "Wait"를 붙여 더 생각하게 만드는 "budget forcing"을 도입했습니다. 결과 모델 s1-32B는 MATH·AIME24에서 o1-preview를 최대 27% 앞섰고, AIME24 성능이 50%→57%로 연산량에 비례해 올랐습니다.

### 핵심 기여
- 단 1,000개 샘플의 SFT만으로 긴 CoT 추론 능력을 끌어낼 수 있음을 입증 (데이터 선별의 중요성)
- 추론 길이(사고 토큰 예산)를 제어하는 간단한 디코딩 기법 budget forcing 제안
- 모델·데이터·코드를 모두 공개해 "추론 모델" 재현의 기준선 제공

### 이 논문이 중요한 이유
2022년의 CoT가 "프롬프트로 생각을 쓰게 하기"였다면, 2025년의 CoT는 "얼마나 오래 생각하게 할지 제어하기"로 진화했습니다. 이 논문은 그 전환을 가장 짧고 명료하게 보여줍니다. 프롬프트 엔지니어링이 학습 데이터 큐레이션·디코딩 제어와 맞닿는 지점을 이해하는 데 필수입니다.

### 사전 지식
Chain-of-Thought, SFT(지도 미세조정), 추론 모델(o1, DeepSeek-R1) 개념, 테스트 타임 컴퓨트

### 관련 논문
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)
- [LIMO: Less is More for Reasoning (Ye et al., 2025)](https://arxiv.org/abs/2502.03387)

### 실무 적용
추론 모델의 "thinking budget"을 기능별로 다르게 설정하는 설계 근거가 됩니다(간단 분류는 짧게, 복잡한 계획·코드 리뷰는 길게). 또한 사내 도메인 추론 모델을 만들 때 대량 데이터보다 소량의 고품질 추론 트레이스 큐레이션이 비용 대비 효과가 크다는 전략적 시사점을 줍니다.

---

## 추천 읽기 순서
1. **PAL** — 가장 직관적. "생각은 모델, 계산은 도구"라는 역할 분리를 먼저 체감합니다.
2. **Self-Ask** — 분해된 단계를 명시적 하위 질문으로 만들고 검색 도구를 연결하는 패턴으로 확장합니다.
3. **s1** — 프롬프트 기반 CoT가 학습·디코딩 수준의 "사고 예산 제어"로 어떻게 진화했는지 확인합니다.

## 핵심 테이크어웨이
- CoT의 가치는 "중간 단계를 텍스트로 드러내는 것"이고, 드러난 단계는 도구(인터프리터, 검색)에 연결할 수 있는 인터페이스가 됩니다.
- 모델을 키운다고 조합 추론과 정확한 계산 문제가 저절로 해결되지 않습니다 → 구조화된 프롬프트와 외부 실행이 필요합니다.
- 2025년 이후에는 "어떻게 생각을 쓰게 할까"보다 "얼마나, 어떤 데이터로 생각하게 할까"가 핵심 질문입니다.
- 질문으로 정리하면: *우리 제품에서 모델이 직접 답하면 안 되는 단계는 어디인가? 그 단계에 어떤 도구를 꽂을 것인가? 각 기능의 적정 사고 예산은 얼마인가?*

## 다음 토픽과의 연결
다음 토픽 **Advanced Prompting (ToT, ReAct, Self-Consistency)**에서는 오늘의 단일 경로 CoT를 여러 경로 탐색(Tree of Thoughts), 다수결(Self-Consistency), 추론-행동 교차(ReAct)로 확장합니다. Self-Ask의 "하위 질문 + 검색"은 ReAct의 직접적인 전신이고, s1의 budget forcing은 탐색 기반 방법과 같은 "테스트 타임 컴퓨트 배분" 문제를 다른 각도에서 다룹니다.
