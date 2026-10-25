# Daily AI Paper Recommendations

> **Date:** 2026-10-26
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity
- **Authors:** Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp
- **Year:** 2021 (ACL 2022 Outstanding Paper)
- **arXiv:** https://arxiv.org/abs/2104.08786
- **PDF:** [./fantastically-ordered-prompts-lu-2021.pdf](./fantastically-ordered-prompts-lu-2021.pdf)
- **Citation Count:** ~1,000+

### 요약
같은 few-shot 예시라도 **순서만 바꿔도** 성능이 SOTA 수준에서 거의 랜덤 수준까지 흔들린다는 것을 보인 논문입니다. 이 현상은 모델 크기와 무관하게 나타나며, 한 모델에서 좋은 순서가 다른 모델로 옮겨지지도 않습니다. 저자들은 라벨된 검증 셋 없이 모델이 스스로 만든 "probing set"과 엔트로피 지표로 좋은 순서를 고르는 방법을 제안해, GPT 계열에서 11개 분류 과제 평균 13%의 상대적 개선을 얻었습니다.

### 핵심 기여
- Few-shot 프롬프트의 **순서 민감도(order sensitivity)**를 체계적으로 측정하고, 모델 크기를 키워도 사라지지 않는다는 것을 증명
- 좋은 순서는 모델 간 **전이되지 않는다**는 발견 — 모델을 바꾸면 프롬프트도 다시 검증해야 함
- 라벨 없이 생성한 probing set + Global/Local Entropy로 순서를 고르는 **비지도 순서 선택 기법** 제시

### 이 논문이 중요한 이유
"예시 몇 개 넣으면 된다"는 few-shot의 직관이 얼마나 불안정한지 수치로 보여준 첫 번째 대표 연구입니다. 프롬프트를 '코드'처럼 다뤄야 하는 이유 — 버전 관리, 회귀 테스트, 모델 교체 시 재평가 — 의 실증적 근거가 됩니다.

### 사전 지식
- GPT-3의 in-context learning(ICL) 개념
- 텍스트 분류 과제와 라벨 분포, 엔트로피의 기본 개념

### 관련 논문
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models (Zhao et al., 2021)](https://arxiv.org/abs/2102.09690)
- [What Makes Good In-Context Examples for GPT-3? (Liu et al., 2021)](https://arxiv.org/abs/2101.06804)
- [Rethinking the Role of Demonstrations (Min et al., 2022)](https://arxiv.org/abs/2202.12837)

### 실무 적용
- Few-shot 예시 순서를 여러 개로 섞어(permutation) 평가셋에서 분산을 측정하고, 분산이 크면 예시 수·형식을 재설계
- 모델 업그레이드(예: GPT-4o → 새 모델) 시 프롬프트 회귀 테스트를 필수 단계로 포함
- 분류형 기능(의도 분류, 라우팅)에서는 예시를 최근성 편향이 덜한 순서로 배치하거나 앙상블

---

## Paper 2 (Classic): Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting
- **Authors:** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman
- **Year:** 2023 (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2305.04388
- **PDF:** [./unfaithful-cot-explanations-turpin-2023.pdf](./unfaithful-cot-explanations-turpin-2023.pdf)
- **Citation Count:** ~600+

### 요약
CoT가 내놓는 "추론 과정"이 실제로 모델이 답을 결정한 이유와 **다를 수 있음(unfaithful)**을 보인 논문입니다. few-shot 예시의 정답을 항상 (A)로 재배열하는 등 편향 요인을 넣으면 모델의 답이 바뀌지만, CoT 설명에는 그 영향이 전혀 언급되지 않고 오히려 틀린 답을 그럴듯하게 합리화합니다. BIG-Bench Hard 13개 과제에서 GPT-3.5, Claude 1.0의 정확도가 최대 36%까지 떨어졌습니다.

### 핵심 기여
- "Answer is Always A", "Suggested Answer" 등 **편향 주입 실험 프레임워크** 제시
- CoT 설명이 편향을 숨긴 채 **사후 합리화(rationalization)**를 한다는 정량적 증거
- 사회적 편향 과제(BBQ)에서 고정관념에 맞는 답을 편향 언급 없이 정당화하는 사례 발견

### 이 논문이 중요한 이유
CoT는 정확도를 올리는 기법이면서 동시에 "설명 가능성" 도구로 쓰이는데, 이 논문은 후자의 신뢰성에 근본적 의문을 던집니다. 추론 모델(o1, R1 등)의 reasoning trace를 모니터링·감사에 쓰려는 현재 흐름에서 반드시 알아야 할 한계입니다. Paper 1과 연결하면, few-shot 예시의 순서·구성 같은 사소한 요인이 답을 바꾸면서도 설명에는 드러나지 않는다는 것을 알 수 있습니다.

### 사전 지식
- Chain-of-Thought Prompting (Wei et al., 2022)
- Few-shot 프롬프팅과 BIG-Bench Hard 벤치마크
- 설명 가능성(XAI)에서 faithfulness vs plausibility 구분

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al., 2023)](https://arxiv.org/abs/2307.13702)
- [Faithful Chain-of-Thought Reasoning (Lyu et al., 2023)](https://arxiv.org/abs/2301.13379)

### 실무 적용
- CoT 출력을 사용자에게 "근거"로 보여주는 UX라면, 설명이 실제 결정 요인이 아닐 수 있음을 전제로 설계 (근거는 RAG 출처·도구 실행 결과 등 검증 가능한 것으로)
- LLM-as-a-Judge 평가에서 예시 정답 위치·순서를 무작위화하여 위치 편향 제거
- 에이전트 로그 감사 시 reasoning trace만 믿지 말고, 입력 변형(counterfactual) 테스트를 병행

---

## Paper 3 (Recent): Chain of Draft: Thinking Faster by Writing Less
- **Authors:** Silei Xu, Wenhao Xie, Lingxiao Zhao, Pengcheng He (Zoom Communications)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.18600
- **PDF:** [./chain-of-draft-xu-2025.pdf](./chain-of-draft-xu-2025.pdf)
- **Code:** https://github.com/sileix/chain-of-draft
- **Citation Count:** ~100+ (빠르게 증가 중)

### 요약
사람이 문제를 풀 때 장황한 문장 대신 핵심만 메모하듯, LLM에게도 **각 추론 단계를 짧은 초안(draft, 단계당 약 5단어 이내)**으로 쓰게 하는 프롬프팅 기법입니다. 산술·상식·기호 추론 과제에서 CoT와 비슷하거나 더 나은 정확도를 내면서 토큰을 최소 7.6% 수준까지 줄여, 비용과 지연 시간을 크게 낮췄습니다. 단, few-shot 예시 없이(zero-shot) 쓰거나 작은 모델에서는 효과가 떨어진다는 한계도 함께 보고합니다.

### 핵심 기여
- "Think step by step, but only keep a minimum draft for each thinking step" 형태의 **간결한 CoT 프롬프트** 제안
- GSM8K 등에서 CoT 대비 **토큰 사용량·지연 시간 대폭 감소**와 정확도 유지를 실증
- Zero-shot·소형 모델에서의 성능 저하를 분석하며, CoD 스타일 few-shot 예시의 중요성을 확인

### 이 논문이 중요한 이유
CoT의 정확도 이득은 곧 **토큰 비용과 지연 시간**이라는 대가를 동반합니다. 실서비스 PM/엔지니어 관점에서 "추론 품질 vs 비용·속도" 트레이드오프를 프롬프트 수준에서 조절할 수 있는 실용적인 레버입니다. 또한 few-shot 예시의 **형식**이 출력 스타일을 결정한다는 점에서 Paper 1과도 연결됩니다.

### 사전 지식
- Chain-of-Thought, Zero-shot CoT ("Let's think step by step")
- 토큰 기반 API 비용 구조와 레이턴시(TTFT, 생성 시간) 개념

### 관련 논문
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)
- [Token-Budget-Aware LLM Reasoning (Han et al., 2024)](https://arxiv.org/abs/2412.18547)
- [To CoT or not to CoT? (Sprague et al., 2024)](https://arxiv.org/abs/2409.12183)

### 실무 적용
- 실시간 응답이 중요한 챗봇·음성 에이전트에서 CoT 대신 CoD로 레이턴시 절감
- 쿼리 난이도에 따라 CoD / CoT / 추론 모델을 고르는 **라우터** 설계 (저자들도 권장)
- 대량 배치 처리(분류, 추출)에서 토큰 비용 절감 실험의 1순위 후보

---

## 추천 읽기 순서
1. **Fantastically Ordered Prompts** — few-shot 프롬프트가 얼마나 불안정한지 먼저 체감
2. **Unfaithful Explanations in CoT** — CoT의 "설명"을 어디까지 믿을 수 있는지 한계 이해
3. **Chain of Draft** — 위 한계를 알고 난 뒤, CoT를 실무 비용 관점에서 최적화하는 방법 학습

## 핵심 테이크어웨이
- Few-shot 예시의 **순서·위치·형식**은 사소한 디테일이 아니라 성능을 좌우하는 변수다 → 프롬프트도 테스트 대상
- CoT는 정확도를 올리지만, 그 추론 텍스트가 **실제 결정 과정을 반영한다는 보장은 없다**
- CoT의 길이는 줄일 수 있다 — 핵심은 "얼마나 많이 쓰느냐"보다 "필요한 중간 상태를 남기느냐"
- 실무에서는 정확도·신뢰성·비용 세 축을 함께 보고 프롬프팅 전략을 고른다

## 다음 토픽과의 연결
다음 토픽 **Advanced Prompting (ToT, ReAct, Self-Consistency)**에서는 단일 추론 체인을 넘어, 여러 경로를 탐색(ToT)하거나 다수결로 합치고(Self-Consistency) 외부 도구와 상호작용(ReAct)하는 방법을 다룹니다. 오늘 본 CoT의 불안정성과 불충실성 문제를 "여러 번 샘플링하고 검증한다"는 방향으로 보완하는 흐름으로 이어집니다.
