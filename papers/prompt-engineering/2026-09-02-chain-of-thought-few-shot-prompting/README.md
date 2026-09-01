# Daily AI Paper Recommendations

> **Date:** 2026-09-02
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Complexity-Based Prompting for Multi-Step Reasoning
- **Authors:** Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, Tushar Khot
- **Year:** 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2210.00720
- **PDF:** [./complexity-based-prompting-fu-2022.pdf](./complexity-based-prompting-fu-2022.pdf)
- **Citation Count:** 약 900+

### 요약
Chain-of-Thought 프롬프트에서 "어떤 예시를 고를 것인가"라는 질문에 대해, 추론 단계 수가 많은(복잡한) 예시일수록 성능이 좋다는 놀랍도록 단순한 규칙을 발견한 논문이다. 저자들은 이 복잡도 기준을 프롬프트 선택(입력)뿐 아니라 디코딩(출력)까지 확장해, 여러 추론 체인을 샘플링한 뒤 그중 가장 복잡한 상위 K개 체인의 다수결로 답을 정하는 Complexity-based Consistency를 제안한다. 3개 수학 벤치마크와 2개 BigBench-Hard 태스크에서 당시 SOTA를 달성했다.

### 핵심 기여
- CoT 예시의 "추론 단계 수(reasoning complexity)"가 성능을 예측하는 강력한 단일 지표임을 실증
- 복잡도 기준을 디코딩 단계로 확장한 Complexity-based Consistency 제안 (Self-Consistency의 복잡도 가중 버전)
- 예시의 정답 여부가 틀려도 복잡도가 높으면 성능이 유지된다는 것을 보여, "형식(format)이 내용보다 중요할 수 있다"는 후속 연구 흐름에 근거 제공
- 프롬프트 예시 선택을 휴리스틱이 아닌 측정 가능한 기준으로 전환

### 이 논문이 중요한 이유
대부분의 실무자는 Few-shot 예시를 "적당히 대표적인 것"으로 손으로 고른다. 이 논문은 그 선택이 성능을 수십 퍼센트 단위로 흔들 수 있으며, 심지어 자동화할 수 있는 명시적 기준이 존재한다는 것을 보여준다. AI 엔지니어에게 이는 프롬프트를 "예술"이 아니라 "최적화 가능한 설계 변수"로 다루는 첫 사고 전환점이 된다. DSPy·MIPRO 같은 자동 프롬프트 최적화 도구들이 데모 선택을 탐색 공간으로 다루는 이유의 원류이기도 하다.

### 사전 지식
- Chain-of-Thought Prompting (Wei et al., 2022)의 기본 개념
- Self-Consistency 디코딩 (다중 샘플링 + 다수결)
- GSM8K, MultiArith, BigBench-Hard 등 추론 벤치마크의 구조
- 온도(temperature) 샘플링과 greedy decoding의 차이

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Least-to-Most Prompting Enables Complex Reasoning (Zhou et al., 2022)](https://arxiv.org/abs/2205.10625)
- [Automatic Chain of Thought Prompting (Zhang et al., 2022)](https://arxiv.org/abs/2210.03493)

### 실무 적용
프롬프트 라이브러리를 관리할 때 예시 풀(pool)에 "추론 단계 수" 메타데이터를 붙여두면, 태스크 난이도에 맞춰 예시를 동적으로 선택하는 라우팅 로직을 만들 수 있다. 실제로 수학·코드·다단계 워크플로우 에이전트에서는 짧고 깔끔한 예시보다 장황하더라도 단계가 명시된 예시를 넣는 편이 성능이 좋은 경우가 많다. 다만 토큰 비용이 늘어나므로, 복잡도 상향과 비용 사이의 트레이드오프를 A/B로 측정해야 한다.

---

## Paper 2 (Classic): Calibrate Before Use: Improving Few-Shot Performance of Language Models
- **Authors:** Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh
- **Year:** 2021 (ICML 2021)
- **arXiv:** https://arxiv.org/abs/2102.09690
- **PDF:** [./calibrate-before-use-zhao-2021.pdf](./calibrate-before-use-zhao-2021.pdf)
- **Citation Count:** 약 2,000+

### 요약
Few-shot 프롬프팅이 실제로는 매우 불안정하다는 사실을 체계적으로 폭로한 논문이다. 프롬프트 포맷, 예시의 선택, 심지어 예시의 "순서"만 바꿔도 정확도가 무작위 수준에서 SOTA 수준까지 요동친다. 저자들은 이 불안정성의 원인을 모델의 세 가지 편향(majority label bias, recency bias, common token bias)으로 규명하고, "N/A" 같은 내용 없는(content-free) 입력에 대한 예측이 균등해지도록 출력 확률을 보정하는 Contextual Calibration을 제안한다.

### 핵심 기여
- Few-shot 성능의 분산이 얼마나 큰지를 대규모로 정량화 (동일 예시라도 순서만 바꾸면 큰 폭으로 변동)
- 불안정성의 원인을 majority label / recency / common token 세 가지 편향으로 분해
- 학습 없이 아핀 변환만으로 출력 분포를 교정하는 Contextual Calibration 제안 (평균 최대 30% 절대 성능 향상)
- Few-shot 결과를 보고할 때 단일 프롬프트 점수를 신뢰하면 안 된다는 평가 방법론적 경고

### 이 논문이 중요한 이유
"프롬프트를 바꿨더니 성능이 올랐다"는 주장의 상당수가 실은 노이즈일 수 있다는 것을 보여준다. AI 엔지니어에게 이 논문은 프롬프트 실험 설계의 기준선을 제공한다 — 여러 시드/순서에 대해 평균과 분산을 함께 보고하지 않으면 그 실험은 의미가 없다. 또한 LLM 출력 확률을 분류 신호로 쓰는 모든 파이프라인(라우팅, 가드레일, LLM-as-judge)에서 캘리브레이션이 필수라는 점을 일깨운다.

### 사전 지식
- In-Context Learning과 Few-shot 프롬프팅의 기본 구조
- 로짓(logit)과 소프트맥스 출력 확률, 확률 캘리브레이션(temperature scaling 등)의 개념
- 분류 태스크에서의 verbalizer(레이블 단어) 설계
- GPT-3 계열 모델의 API 확률(logprob) 출력 방식

### 관련 논문
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Fantastically Ordered Prompts and Where to Find Them (Lu et al., 2021)](https://arxiv.org/abs/2104.08786)
- [Rethinking the Role of Demonstrations (Min et al., 2022)](https://arxiv.org/abs/2202.12837)
- [Surface Form Competition (Holtzman et al., 2021)](https://arxiv.org/abs/2104.08315)

### 실무 적용
LLM을 분류기로 쓰는 제품(문의 인텐트 분류, 콘텐츠 모더레이션, 스팸 판정)에서 클래스 편향이 프롬프트 예시 순서 때문에 생기는 경우가 흔하다. 운영 팁: (1) 예시 순서를 셔플한 여러 프롬프트의 앙상블을 쓰거나, (2) content-free 입력("N/A", 빈 문자열)에 대한 모델의 사전 분포를 측정해 임계값을 조정한다. 프롬프트 회귀 테스트를 만들 때도 단일 프롬프트가 아니라 순서 순열 집합에 대한 성능 분포를 지표로 삼아야 한다.

---

## Paper 3 (Recent): To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning
- **Authors:** Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, Greg Durrett
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2409.12183
- **PDF:** [./to-cot-or-not-to-cot-sprague-2024.pdf](./to-cot-or-not-to-cot-sprague-2024.pdf)
- **Citation Count:** 약 500+

### 요약
CoT를 언제 써야 하는지에 대한 지금까지 가장 큰 규모의 실증 연구다. 100편이 넘는 논문의 결과를 메타 분석하고, 14개 모델 × 20개 데이터셋에 대해 직접 실험한 결과, CoT의 이득이 거의 전적으로 수학·논리·기호 추론 태스크에 집중되어 있음을 보였다. MMLU에서 CoT 성능 향상의 약 95%가 질문이나 생성 결과에 "=" 기호가 포함된 문항에서 나왔다. 상식 추론·지식 QA 등에서는 이득이 미미하거나 오히려 손해였다.

### 핵심 기여
- CoT 효과를 태스크 유형별로 분해한 대규모 메타 분석 (100+ 논문, 20 데이터셋, 14 모델)
- CoT 이득의 원천이 "계획(planning)"보다 "기호 실행(symbolic execution)" 단계에 있음을 분리 실험으로 규명
- CoT를 무조건 켜두는 것이 지연시간·비용 측면에서 손해라는 실용적 결론 제시
- 기호 실행이 필요한 태스크는 CoT보다 도구 호출(코드 실행, 심볼릭 솔버)이 더 낫다는 방향 제시

### 이 논문이 중요한 이유
"항상 step by step으로 생각하게 하라"는 2022~2023년의 통념을 데이터로 반박한다. 추론 모델(o-series, R1 계열)이 보편화되고 토큰당 비용과 지연시간이 제품 지표가 된 지금, CoT를 켤지 끌지는 아키텍처 결정이다. AI 엔지니어는 이 논문을 통해 "어떤 태스크에 추론 예산을 배분할 것인가"라는 라우팅 문제로 CoT를 재정의하게 된다. 오늘의 고전 두 편이 "프롬프트 내부를 어떻게 다듬을까"였다면, 이 논문은 "그 프롬프트를 애초에 쓸 것인가"를 묻는다.

### 사전 지식
- Chain-of-Thought, Zero-shot CoT의 기본 개념
- MMLU, GSM8K, MATH, BigBench-Hard 등 벤치마크의 태스크 구성
- 도구 호출(tool use) 및 Program-aided LM(PAL) 패러다임
- 메타 분석의 기본 개념(효과 크기, 이질성)

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [PAL: Program-aided Language Models (Gao et al., 2022)](https://arxiv.org/abs/2211.10435)
- [Chain-of-Thought Reasoning Without Prompting (Wang & Zhou, 2024)](https://arxiv.org/abs/2402.10200)
- [Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot (Wang et al., 2025)](https://arxiv.org/abs/2506.14641)

### 실무 적용
프로덕션 LLM 파이프라인에서 CoT를 전역 기본값으로 켜두는 대신 태스크 라우터를 두어, (1) 수식·계산·논리 제약이 있는 요청 → CoT 또는 코드 실행 도구, (2) 검색·요약·분류·톤 조정 요청 → 직답(direct answer)으로 분기시키면 지연시간과 비용을 크게 줄이면서 품질을 유지할 수 있다. 실제로 챗봇 응답의 상당 비율은 CoT가 필요 없는데, CoT를 켜두면 응답 시간이 수 배로 늘고 이탈이 발생한다. 또한 기호 실행이 필요한 구간은 CoT로 "말로 계산"하게 두지 말고 계산기·SQL·코드 인터프리터를 붙이는 편이 정확도와 비용 모두에서 유리하다.

---

## 추천 읽기 순서
1. **Calibrate Before Use (Zhao et al., 2021)** — 먼저 Few-shot 프롬프팅이 얼마나 불안정한 도구인지 체감한다. 프롬프트 실험을 신뢰하는 기준선이 여기서 잡힌다.
2. **Complexity-Based Prompting (Fu et al., 2022)** — 그 불안정성 위에서 "그래도 어떤 예시가 좋은가"에 대한 측정 가능한 답을 얻는다.
3. **To CoT or not to CoT? (Sprague et al., 2024)** — 마지막으로 한 단계 올라가, CoT 자체를 언제 쓸지 결정하는 관점을 얻는다.

## 핵심 테이크어웨이
- **프롬프트는 예술이 아니라 최적화 문제다.** 예시의 선택·순서·복잡도는 모두 측정 가능한 설계 변수이며, 성능 변동 폭이 모델 교체에 맞먹을 만큼 크다.
- **단일 프롬프트 점수는 증거가 아니다.** 순서 순열과 시드에 대한 분포(평균 ± 분산)를 보고하지 않으면 그 프롬프트 개선은 검증되지 않은 주장이다.
- **CoT는 공짜가 아니다.** 이득은 수학·기호 추론에 몰려 있고, 나머지 태스크에서는 비용과 지연만 늘린다. 기본값이 아니라 라우팅 결정으로 다뤄야 한다.
- **기호 실행은 도구에게 맡겨라.** CoT로 "말로 계산"시키는 대신 코드 실행·솔버를 붙이는 것이 정확도·비용 모두에서 우위다.
- **세 논문을 관통하는 질문:** 모델의 출력이 프롬프트의 *내용* 때문인가, 아니면 *형식과 편향* 때문인가. 이 구분을 못 하면 프롬프트 튜닝은 노이즈 쫓기가 된다.

## 다음 토픽과의 연결
오늘은 "단일 프롬프트 안에서 추론을 어떻게 유도하고, 언제 유도하지 말아야 하는가"를 다뤘다. 다음 토픽인 **Advanced Prompting (ToT, ReAct, Self-Consistency)** 은 여기서 한 걸음 나아가, 추론을 하나의 선형 체인이 아니라 **탐색·분기·외부 행동이 있는 구조**로 확장한다. 오늘 본 Complexity-based Consistency가 다중 샘플링의 초기 형태였다면 Tree of Thoughts는 그 샘플링을 명시적 탐색 트리로, ReAct는 추론과 도구 호출의 교대 루프로 일반화한다. 그리고 "To CoT or not to CoT?"가 던진 "기호 실행은 도구에게"라는 결론이 바로 ReAct와 에이전트 아키텍처로 이어지는 다리가 된다.
