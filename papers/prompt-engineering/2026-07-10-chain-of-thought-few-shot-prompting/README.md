# Daily AI Paper Recommendations

> **Date:** 2026-07-10
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Automatic Chain of Thought Prompting in Large Language Models
- **Authors:** Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2210.03493](https://arxiv.org/abs/2210.03493)
- **PDF:** [./automatic-chain-of-thought-zhang-2022.pdf](./automatic-chain-of-thought-zhang-2022.pdf)
- **Citation Count:** approx. 1,500+

### 요약
CoT 프롬프팅은 사람이 직접 만든 추론 예시(demonstration)에 크게 의존하는데, 이 수작업은 비용이 크고 최적이 아닐 수 있다. Auto-CoT는 "Let's think step by step"이라는 제로샷 프롬프트로 LLM 스스로 추론 체인을 생성하게 하고, 질문들을 k-means로 군집화해 다양한 예시를 자동 선택함으로써 사람 개입 없이 수작업 수준의 성능을 달성한다.

### 핵심 기여
- 사람이 만든 예시 없이 LLM이 스스로 추론 체인을 만들어 few-shot CoT 예시를 자동 구성하는 파이프라인 제안
- 질문 다양성 확보를 위한 클러스터링 기반 샘플링으로, 유사 질문에 몰려 발생하는 오류 전파(misleading demonstration) 완화
- GSM8K 등 10개 추론 벤치마크에서 수작업 CoT와 대등하거나 그 이상의 성능 입증

### 이 논문이 중요한 이유
프롬프트 엔지니어링을 "장인의 수작업"에서 "자동화 가능한 파이프라인"으로 옮긴 초기 대표작이다. AI 엔지니어가 프롬프트를 사람이 매번 튜닝하는 대신, 데이터 기반으로 예시를 선택·생성하는 시스템을 설계하는 사고방식의 출발점이 된다.

### 사전 지식
- Chain-of-Thought(Wei et al., 2022)와 Zero-shot CoT(Kojima et al., 2022)의 기본 개념
- 문장 임베딩과 k-means 클러스터링
- few-shot in-context learning의 작동 원리와 예시 민감성

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)

### 실무 적용
RAG나 에이전트 파이프라인에서 few-shot 예시를 하드코딩하지 않고, 사용자 질문을 임베딩·군집화해 대표 예시를 동적으로 선택하는 "예시 라이브러리" 설계에 직접 응용된다. 프롬프트 유지보수 비용을 줄이면서 도메인 확장성을 확보하는 패턴이다.

---

## Paper 2 (Classic): What Makes Good In-Context Examples for GPT-3?
- **Authors:** Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, Weizhu Chen
- **Year:** 2021
- **arXiv:** [https://arxiv.org/abs/2101.06804](https://arxiv.org/abs/2101.06804)
- **PDF:** [./good-in-context-examples-gpt3-liu-2021.pdf](./good-in-context-examples-gpt3-liu-2021.pdf)
- **Citation Count:** approx. 1,900+

### 요약
GPT-3의 few-shot 성능은 어떤 in-context 예시를 넣느냐에 매우 민감하다. 이 논문은 테스트 샘플과 임베딩 공간에서 가까운 예시가 일관되게 더 좋은 성능을 낸다는 것을 관찰하고, 검색(nearest neighbor) 기반으로 예시를 선택하는 KATE 방법을 제안한다.

### 핵심 기여
- few-shot 예시 선택이 성능을 좌우하는 핵심 변수임을 실증적으로 규명
- 테스트 입력과 의미적으로 가까운 예시를 검색해 프롬프트를 구성하는 retrieval 기반 예시 선택(KATE) 제안
- Table-to-text(ToTTo +41.9%), 오픈도메인 QA(NQ +45.5%) 등에서 랜덤 선택 대비 큰 성능 향상 입증

### 이 논문이 중요한 이유
"프롬프트에 무엇을 넣을지"를 검색 문제로 재정의한 논문으로, 오늘날 RAG와 dynamic few-shot의 이론적 뿌리다. AI 엔지니어에게 in-context learning이 왜 예시 선택 전략에 좌우되는지를 이해하는 필독 근거를 제공한다.

### 사전 지식
- GPT-3와 in-context(few-shot) learning 개념
- 문장 임베딩과 코사인 유사도 기반 nearest-neighbor 검색
- 벡터 검색의 기본 원리

### 관련 논문
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Fantastically Ordered Prompts and Where to Find Them (Lu et al., 2021)](https://arxiv.org/abs/2104.08786)

### 실무 적용
프로덕션 LLM 애플리케이션에서 예시를 고정하지 않고, 사용자 입력을 임베딩해 벡터DB에서 가장 유사한 예시를 검색·주입하는 dynamic few-shot 프롬프트 설계의 표준 패턴으로 쓰인다. RAG와 결합해 정확도와 도메인 적응력을 동시에 높인다.

---

## Paper 3 (Recent): Chain-of-Thought Reasoning Without Prompting
- **Authors:** Xuezhi Wang, Denny Zhou
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** [https://arxiv.org/abs/2402.10200](https://arxiv.org/abs/2402.10200)
- **PDF:** [./cot-reasoning-without-prompting-wang-2024.pdf](./cot-reasoning-without-prompting-wang-2024.pdf)
- **Citation Count:** approx. 250+

### 요약
CoT를 이끌어내려면 반드시 특별한 프롬프트가 필요하다는 통념에 도전한다. 저자들은 greedy 디코딩 대신 top-k 대안 토큰을 탐색하는 방식(CoT-decoding)만으로도 사전학습 모델 안에 잠재된 추론 경로가 드러난다는 것을 보인다. 또한 디코딩 경로에 CoT가 존재할 때 모델의 정답 신뢰도가 더 높다는 상관관계를 발견한다.

### 핵심 기여
- 프롬프트 없이 디코딩 절차만 바꿔 CoT 추론 경로를 이끌어내는 CoT-decoding 제안
- CoT 경로 존재 여부와 모델의 답변 신뢰도(logit 차이) 사이의 강한 상관관계 규명
- 다양한 추론 벤치마크에서 greedy 디코딩에 가려져 있던 내재적 추론 능력을 정량적으로 입증

### 이 논문이 중요한 이유
"CoT는 프롬프트 기법"이라는 프레임을 넘어, 추론이 모델 내부에 이미 존재하며 디코딩으로 끌어낼 수 있음을 보여준다. 프롬프트 튜닝만이 아니라 디코딩·샘플링 전략도 추론 성능을 좌우하는 제어 지점임을 알려주는 최신 필독작이다.

### 사전 지식
- 언어모델의 디코딩 방식(greedy, top-k, beam search)과 토큰 확률
- Chain-of-Thought / Self-Consistency의 개념
- logit과 모델 신뢰도(confidence) 해석

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

### 실무 적용
프롬프트를 바꾸기 어려운 상황(고정 시스템 프롬프트, 함수 호출 등)에서도 디코딩 파라미터와 후보 경로 선택으로 추론 품질을 개선할 수 있음을 시사한다. CoT-decoding의 신뢰도 신호는 답변 자동 검증·재시도(self-verification) 로직 설계에 활용 가능하다.

---

## 추천 읽기 순서
1. **Paper 2 (What Makes Good In-Context Examples, 2021)** — few-shot 성능이 예시 선택에 왜 민감한지 먼저 이해한다.
2. **Paper 1 (Auto-CoT, 2022)** — 그 예시 선택·생성을 자동화하는 파이프라인으로 확장한다.
3. **Paper 3 (CoT Reasoning Without Prompting, 2024)** — 프롬프트를 넘어 디코딩 차원에서 추론을 제어하는 최신 관점으로 마무리한다.

## 핵심 테이크어웨이
- few-shot/CoT 성능은 "무엇을 예시로 넣는가"에 결정적으로 좌우된다 → 예시 선택은 검색·클러스터링 문제로 다룰 수 있다.
- 사람 수작업 프롬프트는 자동화 가능하다: Auto-CoT처럼 모델이 스스로 예시를 만들고 다양성을 확보하게 설계하라.
- 추론은 프롬프트뿐 아니라 디코딩 전략에도 내재한다 → 프롬프트, 예시 검색, 디코딩을 함께 최적화하는 것이 AI 네이티브 설계의 방향이다.

## 다음 토픽과의 연결
오늘 다룬 예시 선택·자동화·디코딩 제어는 Day 19의 **Advanced Prompting (Tree-of-Thoughts, ReAct, Self-Consistency)** 로 이어진다. 단일 CoT 경로를 넘어 다중 경로 탐색과 도구 사용(action)을 결합하는 고급 추론 프레임워크로 확장되는 흐름을 미리 염두에 두면 좋다.
