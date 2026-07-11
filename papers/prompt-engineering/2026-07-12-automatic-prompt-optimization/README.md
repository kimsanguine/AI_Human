# Daily AI Paper Recommendations

> **Date:** 2026-07-12
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): Automatic Prompt Optimization with "Gradient Descent" and Beam Search
- **Authors:** Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.03495
- **PDF:** [./apo-gradient-descent-beam-search-pryzant-2023.pdf](./apo-gradient-descent-beam-search-pryzant-2023.pdf)
- **Citation Count:** ~700+

### 요약
경사하강법의 개념을 텍스트 공간으로 옮긴 ProTeGi(APO) 논문입니다. LLM이 현재 프롬프트의 실패 사례를 자연어로 비평하게 하고("텍스트 그래디언트"), 그 비평을 반영해 프롬프트를 수정한 뒤 빔 서치로 최적의 후보를 탐색합니다. 사람이 프롬프트를 손으로 다듬는 과정을 완전히 자동화한 초기 대표작입니다.

### 핵심 기여
- 미분 불가능한 텍스트 프롬프트에 "경사하강법" 메타포를 적용 — 실패 사례에 대한 자연어 비평을 그래디언트로 사용
- 빔 서치 + 밴딧 선택(UCB)으로 프롬프트 후보 공간을 효율적으로 탐색
- 사람 개입 없이 프롬프트 성능을 최대 31%까지 개선함을 실증

### 이 논문이 중요한 이유
프롬프트 튜닝을 "감"이 아니라 데이터 기반 최적화 루프로 바꾼 전환점입니다. TextGrad, DSPy 등 이후 등장한 거의 모든 자동 프롬프트 최적화 프레임워크가 이 논문의 "자연어 피드백 = 그래디언트" 아이디어를 계승합니다.

### 사전 지식
경사하강법과 빔 서치의 기본 개념, few-shot 프롬프팅, LLM 평가 지표(정확도 기반 스코어링)에 대한 이해가 필요합니다.

### 관련 논문
- [Large Language Models Are Human-Level Prompt Engineers (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)

### 실무 적용
고객 지원 분류, 콘텐츠 모더레이션처럼 라벨된 평가셋이 있는 태스크에서 프롬프트를 자동으로 개선하는 파이프라인에 그대로 적용됩니다. 평가셋 + 최적화 루프만 있으면 프롬프트 유지보수 비용을 크게 줄일 수 있습니다.

---

## Paper 2 (Classic): DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **Authors:** Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2310.03714
- **PDF:** [./dspy-compiling-lm-pipelines-khattab-2023.pdf](./dspy-compiling-lm-pipelines-khattab-2023.pdf)
- **Citation Count:** ~1,000+

### 요약
프롬프트를 손으로 쓰는 대신, LM 파이프라인을 선언적 모듈(Signature)로 정의하고 컴파일러(Teleprompter)가 프롬프트와 few-shot 예시를 자동 생성·최적화하게 하는 프레임워크입니다. "프롬프트 엔지니어링"을 "프로그래밍 + 컴파일" 패러다임으로 대체하자는 제안입니다.

### 핵심 기여
- Signature(입력→출력 선언), Module(Predict, ChainOfThought 등), Teleprompter(옵티마이저)로 구성된 프로그래밍 모델 제시
- 손으로 쓴 프롬프트 없이 부트스트래핑으로 few-shot 데모를 자동 생성하는 컴파일 개념 도입
- 작은 모델(T5, Llama2-13b)도 컴파일을 통해 GPT-3.5 수준 파이프라인 성능에 도달함을 실증

### 이 논문이 중요한 이유
프롬프트 문자열에 의존하는 시스템은 모델이 바뀔 때마다 깨집니다. DSPy는 태스크 정의와 프롬프트 구현을 분리해 모델 교체·업그레이드에 강건한 LLM 시스템 설계를 가능하게 했고, 현재 가장 널리 쓰이는 프롬프트 최적화 프레임워크로 성장했습니다. MIPRO(3주기 추천 논문)가 바로 DSPy의 옵티마이저입니다.

### 사전 지식
Chain-of-Thought, few-shot 프롬프팅, RAG 파이프라인 구조, 그리고 PyTorch 스타일의 모듈형 프로그래밍 개념을 알면 이해가 빠릅니다.

### 관련 논문
- [Optimizing Instructions and Demonstrations for Multi-Stage LM Programs / MIPRO (Opsahl-Ong et al., 2024)](https://arxiv.org/abs/2406.11695)
- [Demonstrate-Search-Predict (Khattab et al., 2022)](https://arxiv.org/abs/2212.14024)

### 실무 적용
멀티스텝 에이전트나 RAG 제품에서 프롬프트를 코드처럼 버전 관리하고, 모델 교체 시 재컴파일만으로 성능을 복원하는 워크플로우가 가능합니다. 평가셋 기반으로 프롬프트 회귀 테스트를 자동화하는 데도 활용됩니다.

---

## Paper 3 (Recent): A Systematic Survey of Automatic Prompt Optimization Techniques
- **Authors:** Kiran Ramnath, Kang Zhou, Sheng Guan, et al. (Amazon)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.16923
- **PDF:** [./systematic-survey-apo-ramnath-2025.pdf](./systematic-survey-apo-ramnath-2025.pdf)
- **Citation Count:** ~50+ (2025년 2월 공개)

### 요약
자동 프롬프트 최적화(APO)를 공식적으로 정의하고, 5개 구성요소(초기 프롬프트, 평가 함수, 후보 생성, 필터링, 반복 종료)로 이루어진 통합 프레임워크를 제시한 서베이입니다. APE, APO, OPRO, DSPy 등 기존 연구 전체를 이 프레임워크로 분류·비교합니다.

### 핵심 기여
- APO의 공식 정의와 5-파트 통합 프레임워크 제시 — 산재된 연구들을 하나의 지도로 정리
- 후보 생성 전략(휴리스틱 편집, 진화 알고리즘, 텍스트 그래디언트, 메타 프롬프팅 등)의 체계적 분류
- 미해결 과제(평가 비용, 멀티태스크 일반화, 멀티모달 확장) 정리

### 이 논문이 중요한 이유
APO 분야는 2~3년 사이 논문이 폭증해 전체 그림을 잡기 어렵습니다. 이 서베이 하나로 1·2·3주기에서 읽은 APE, Prefix-Tuning, OPRO, RLPrompt, TextGrad, MIPRO가 프레임워크의 어느 위치에 있는지 한눈에 정리됩니다. 커리큘럼 마무리용 지도 역할을 합니다.

### 사전 지식
이번 주기 Paper 1·2(APO, DSPy)와 이전 주기의 APE, OPRO 정도를 먼저 읽으면 서베이의 분류 체계가 훨씬 잘 읽힙니다.

### 관련 논문
- [Automatic Prompt Optimization with "Gradient Descent" and Beam Search (Pryzant et al., 2023)](https://arxiv.org/abs/2305.03495)
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [PromptWizard: Task-Aware Prompt Optimization Framework (Agarwal et al., 2024)](https://arxiv.org/abs/2405.18369)

### 실무 적용
새 프로젝트에서 어떤 APO 기법을 채택할지 결정할 때 의사결정 프레임워크로 사용합니다. 평가셋 크기, API 비용, 태스크 유형에 따라 적합한 기법 계열(진화형 vs 그래디언트형 vs 컴파일형)을 선택하는 기준을 제공합니다.

---

## 추천 읽기 순서
1. **Paper 1 (APO/ProTeGi)** — "텍스트 그래디언트" 개념으로 자동 최적화의 직관을 잡습니다.
2. **Paper 2 (DSPy)** — 개별 프롬프트 최적화를 넘어 파이프라인 전체를 컴파일하는 패러다임을 이해합니다.
3. **Paper 3 (APO Survey)** — 4주기에 걸쳐 읽은 모든 기법을 하나의 프레임워크로 정리합니다.

## 핵심 테이크어웨이
- 프롬프트는 손으로 쓰는 아티팩트가 아니라 평가셋 기반으로 최적화되는 파라미터로 취급하는 것이 현대적 접근입니다.
- 자연어 비평을 그래디언트처럼 쓰는 아이디어(ProTeGi → TextGrad)와 파이프라인을 선언적으로 컴파일하는 아이디어(DSPy)가 APO의 양대 축입니다.
- 좋은 평가 함수(metric) 설계가 최적화 기법 선택보다 먼저입니다 — 평가가 없으면 자동 최적화도 없습니다.

## 다음 토픽과의 연결
다음 토픽인 "LLM Application Frameworks and Orchestration"(Module 8)에서는 최적화된 프롬프트를 실제 도구 사용·오케스트레이션 파이프라인(Toolformer, MRKL, LangGraph)에 통합하는 방법을 다룹니다. DSPy의 모듈 개념이 오케스트레이션 프레임워크와 자연스럽게 이어집니다.
