# Daily AI Paper Recommendations

> **Date:** 2026-08-30
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Scaling Instruction-Finetuned Language Models
- **Authors:** Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2210.11416
- **PDF:** [./scaling-instruction-finetuned-lm-chung-2022.pdf](./scaling-instruction-finetuned-lm-chung-2022.pdf)
- **Citation Count:** ~3,800

### 요약
Instruction tuning을 (1) 태스크 수, (2) 모델 크기, (3) Chain-of-Thought 데이터라는 세 축으로 스케일링했을 때 무슨 일이 일어나는지를 대규모로 실험한 논문이다. 1,836개 태스크로 파인튜닝한 Flan-PaLM 540B는 5-shot MMLU 75.2%를 달성했고, 80M짜리 Flan-T5조차 훨씬 큰 미세조정되지 않은 모델을 few-shot에서 앞질렀다. 즉, instruction tuning은 모델 크기와 무관하게 거의 항상 이득이며 사전학습 대비 연산 비용은 극히 작다는 것을 실증했다.

### 핵심 기여
- 태스크 수를 62 → 282 → 1,836개로 늘리며 "태스크 다양성이 성능을 끌어올린다"는 스케일링 축을 정량화했다. 다만 수익 체감(diminishing return)도 함께 보여줬다.
- CoT 데이터를 instruction tuning 믹스에 9개 데이터셋만 섞어도 zero-shot 추론 능력이 크게 살아나며, 반대로 CoT를 빼면 추론 성능이 오히려 퇴화한다는 것을 밝혔다.
- T5 / PaLM / U-PaLM 등 아키텍처와 80M~540B 규모를 가로질러 일반화됨을 보였고, Flan-T5 체크포인트를 공개해 오픈소스 instruction tuning의 표준 출발점을 만들었다.

### 이 논문이 중요한 이유
AI 엔지니어 입장에서 이 논문은 "파인튜닝 예산을 어디에 쓸 것인가"에 대한 가장 명확한 실증 근거다. 사전학습 FLOPs의 0.2% 수준의 instruction tuning만으로 두 자릿수 성능 향상이 나온다는 결과는, 자체 모델을 만들 수 없는 팀도 데이터 큐레이션만으로 경쟁력을 확보할 수 있다는 뜻이다. 또한 "태스크 다양성 > 태스크당 샘플 수"라는 교훈은 사내 파인튜닝 데이터셋 설계 시 가장 먼저 적용해야 할 원칙이다.

### 사전 지식
Transformer encoder-decoder(T5)와 decoder-only(PaLM)의 차이, few-shot / zero-shot 프롬프팅, Chain-of-Thought 프롬프팅, MMLU·BBH 같은 벤치마크의 의미. FLAN(Wei et al., 2021)을 먼저 읽으면 이 논문이 "무엇을 더 밀어붙였는지"가 선명해진다.

### 관련 논문
- [Finetuned Language Models Are Zero-Shot Learners / FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (Longpre et al., 2023)](https://arxiv.org/abs/2301.13688)
- [Super-NaturalInstructions (Wang et al., 2022)](https://arxiv.org/abs/2204.07705)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

### 실무 적용
사내 도메인 특화 LLM을 만들 때, 단일 태스크 데이터를 대량으로 모으는 대신 "우리 제품에서 유저가 요구하는 행동"을 태스크 단위로 쪼개 수십~수백 개 유형으로 다양화하는 전략의 근거가 된다. 또한 Flan-T5는 지금도 분류·추출·요약처럼 지연시간과 비용이 중요한 파이프라인 구간에서 대형 LLM을 대체하는 저비용 워크호스로 쓰인다. 추론이 필요한 태스크라면 파인튜닝 데이터에 CoT 샘플을 반드시 일부 섞어야 한다는 실무 규칙도 여기서 나온다.

---

## Paper 2 (Classic): LIMA: Less Is More for Alignment
- **Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.11206
- **PDF:** [./lima-less-is-more-for-alignment-zhou-2023.pdf](./lima-less-is-more-for-alignment-zhou-2023.pdf)
- **Citation Count:** ~2,000

### 요약
LLaMA 65B를 엄선한 1,000개의 프롬프트-응답 쌍만으로, RLHF나 선호 모델링 없이 표준 supervised loss로만 파인튜닝한 LIMA를 제시한다. 그럼에도 LIMA는 GPT-4 대비 43%, Bard 대비 58%의 응답에서 동등하거나 더 선호되는 결과를 얻었다. 저자들은 이를 근거로 "모델의 지식과 능력은 거의 전부 사전학습에서 획득되며, 정렬(alignment)은 그 능력을 어떤 형식으로 꺼낼지를 가르치는 얕은 과정"이라는 Superficial Alignment Hypothesis를 제안한다.

### 핵심 기여
- 1,000개 샘플이라는 극단적으로 작은 데이터로도 고품질 대화 모델이 나온다는 것을 보여, "정렬 = 대규모 데이터"라는 통념을 깼다.
- 데이터 품질·다양성이 데이터 양보다 지배적임을 통제 실험으로 입증했다. 필터링되지 않은 대량 데이터(Stack Exchange 2,000 vs 필터링 200 등) 비교가 특히 설득력 있다.
- 학습 데이터에 없던 대화 형식도 30개 멀티턴 예시만 추가하면 크게 개선된다는 것을 보여, 형식 학습의 샘플 효율성을 드러냈다.

### 이 논문이 중요한 이유
RLHF 파이프라인 전체를 구축할 여력이 없는 대부분의 팀에게 이 논문은 현실적인 대안 경로를 제시한다. "데이터 1,000건을 잘 만드는 것이 10만 건을 대충 모으는 것보다 낫다"는 결론은 데이터 라벨링 예산 배분, PM의 스코프 설계, 그리고 LLM 제품의 초기 MVP 전략을 직접적으로 바꾼다. 동시에 왜 사전학습 모델 선택이 그토록 중요한지도 설명해준다.

### 사전 지식
Supervised Fine-Tuning(SFT)과 RLHF의 차이, LLaMA 계열 모델 구조, 사람 선호 기반 pairwise 평가 방법론, 그리고 사전학습에서 획득되는 지식과 파인튜닝에서 획득되는 행동의 구분.

### 관련 논문
- [Training Language Models to Follow Instructions with Human Feedback / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2022)](https://arxiv.org/abs/2212.10560)
- [AlpaGasus: Training a Better Alpaca with Fewer Data (Chen et al., 2023)](https://arxiv.org/abs/2307.08701)
- [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning / URIAL (Lin et al., 2023)](https://arxiv.org/abs/2312.01552)

### 실무 적용
스타트업이 도메인 특화 어시스턴트를 만들 때 가장 먼저 시도할 전략이다. 도메인 전문가가 직접 작성한 수백~수천 건의 고품질 예시로 SFT를 돌리면, 대규모 RLHF 없이도 제품 수준의 톤·형식·정확도를 확보할 수 있다. 실무에서는 이 원칙이 "데이터 큐레이션 담당자 1명 채용 > GPU 클러스터 증설"이라는 우선순위로 나타난다. 다만 LIMA는 안전성·견고성에서는 RLHF 모델에 뒤지므로, 프로덕션에서는 가드레일과 선호 최적화 단계를 별도로 얹어야 한다.

---

## Paper 3 (Recent): Tulu 3: Pushing Frontiers in Open Language Model Post-Training
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi
- **Year:** 2024 (v5: 2025)
- **arXiv:** https://arxiv.org/abs/2411.15124
- **PDF:** [./tulu-3-open-post-training-lambert-2024.pdf](./tulu-3-open-post-training-lambert-2024.pdf)
- **Citation Count:** ~500+

### 요약
Allen AI가 post-training 전 과정 — 데이터, 코드, 학습 레시피, 평가 프레임워크 — 을 완전 공개한 모델 패밀리다. Llama 3.1 베이스 위에 SFT → DPO → RLVR(Reinforcement Learning with Verifiable Rewards)의 3단계 파이프라인을 적용해, Llama 3.1-Instruct와 Qwen 2.5-Instruct는 물론 GPT-4o-mini, Claude 3.5-Haiku 같은 클로즈드 모델까지 여러 벤치마크에서 앞섰다. 2026년 현재 "오픈 post-training 레시피"의 사실상 표준 참조 문서다.

### 핵심 기여
- RLVR 도입: 수학·코드처럼 정답을 프로그램적으로 검증할 수 있는 태스크에서, 학습된 reward model 대신 검증 가능한 이진 보상만으로 RL을 돌리는 방식을 정식화했다. reward hacking을 구조적으로 줄이는 접근이다.
- 스킬 단위 데이터 큐레이션 방법론: 목표 스킬(추론, 수학, 코딩, 지시 따르기, 안전성 등)을 먼저 정의하고 각 스킬별로 데이터를 합성·정제하며, 평가셋과의 decontamination까지 파이프라인화했다.
- 개발용(dev)과 검증용(unseen) 평가를 분리한 재현 가능한 평가 프레임워크를 공개해, post-training 연구의 과적합 문제를 정면으로 다뤘다.

### 이 논문이 중요한 이유
InstructGPT가 "RLHF가 된다"를 보였다면, Tulu 3는 "당신 팀이 실제로 그것을 어떻게 재현하는가"를 처음부터 끝까지 문서화했다. 데이터 믹스 비율, 하이퍼파라미터, 실패한 시도까지 공개되어 있어 자체 post-training을 시도하는 엔지니어에게는 교과서에 가깝다. 특히 RLVR은 이후 DeepSeek-R1 계열의 추론 모델 학습으로 이어지는 흐름의 중요한 길목이다.

### 사전 지식
SFT, DPO(Direct Preference Optimization), PPO 기반 RLHF의 기본 구조. reward model과 reward hacking 개념. 그리고 벤치마크 오염(contamination)과 평가 과적합이 왜 post-training에서 핵심 문제인지에 대한 감각.

### 관련 논문
- [Direct Preference Optimization / DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning / GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2 (Ivison et al., 2023)](https://arxiv.org/abs/2311.10702)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)

### 실무 적용
자체 SaaS에 특화 모델을 붙이려는 팀이라면 Tulu 3의 레시피를 그대로 출발점으로 삼을 수 있다. 특히 RLVR은 "정답 검증이 가능한 도메인"(SQL 생성, API 호출 스키마 준수, 계산, 구조화된 출력)에서 즉시 응용 가능하다 — Agentic AI 제품에서 tool call 형식 준수율을 올리는 데 reward model 없이 적용할 수 있다는 뜻이다. 또한 dev/unseen 평가 분리 방식은 사내 LLM 평가 체계를 설계할 때 그대로 차용할 만하다.

---

## 추천 읽기 순서

1. **Scaling Instruction-Finetuned Language Models (Chung et al., 2022)** — instruction tuning이 "왜, 얼마나" 통하는지의 정량적 지형도를 먼저 얻는다. 스케일링 축(태스크 수 / 모델 크기 / CoT)을 머리에 넣고 시작해야 이후 논문들의 선택이 이해된다.
2. **LIMA (Zhou et al., 2023)** — 위의 "많을수록 좋다"에 대한 강력한 반례이자 보완이다. 1번을 읽은 직후 읽어야 "양 vs 질" 논쟁의 양쪽 극단을 대비시켜 볼 수 있다.
3. **Tulu 3 (Lambert et al., 2024)** — 앞의 두 관점이 실제 프로덕션 파이프라인에서 어떻게 합성되는지를 본다. SFT 데이터 큐레이션(1·2번의 교훈) 위에 DPO와 RLVR이 얹히는 전체 그림이 완성된다.

## 핵심 테이크어웨이

- **정렬은 얕고, 지식은 사전학습에 있다.** LIMA의 Superficial Alignment Hypothesis는 post-training 예산을 어디에 쓸지 결정하는 첫 번째 질문("우리 베이스 모델이 이미 이 능력을 갖고 있는가?")을 만들어준다. 능력이 없으면 SFT 데이터를 아무리 늘려도 생기지 않는다.
- **데이터의 다양성 > 데이터의 양, 데이터의 품질 > 데이터의 다양성.** Chung은 태스크 다양성의 가치를, LIMA는 품질의 지배력을 보였다. 실무 우선순위는 (1) 고품질 소량 확보 → (2) 태스크 유형 확장 → (3) 총량 증대 순서다.
- **검증 가능한 보상은 학습된 보상보다 안전하다.** Tulu 3의 RLVR은 reward model이 만드는 reward hacking을 우회한다. 정답을 코드로 검증할 수 있는 태스크라면 선호 데이터 수집 없이도 RL을 돌릴 수 있다는 뜻이며, 이는 Agentic 제품에서 특히 강력하다.
- **CoT 데이터를 빼면 추론이 퇴화한다.** 파인튜닝 데이터 믹스에서 추론 샘플은 옵션이 아니라 필수 성분이다.
- **평가 설계가 곧 post-training 설계다.** Tulu 3가 dev/unseen을 분리한 이유는, 평가셋에 맞춰 튜닝하는 순간 실제 유저 경험과의 상관이 끊기기 때문이다. 벤치마크 점수보다 프로덕션 지표와 연결된 자체 평가셋을 먼저 만들어야 한다.

## 다음 토픽과의 연결

다음 토픽인 **LLM Evaluation and Benchmarks**로 자연스럽게 이어진다. 오늘 세 논문은 모두 "무엇을 학습시킬 것인가"를 다뤘지만, 세 논문 모두 결국 "그래서 좋아진 것을 어떻게 증명할 것인가"라는 질문에 부딪힌다 — Chung은 MMLU·BBH에, LIMA는 사람 pairwise 선호 평가에, Tulu 3는 자체 평가 프레임워크와 decontamination에 의존했다. 특히 Tulu 3가 dev/unseen 평가 분리를 강조한 지점은 다음 토픽에서 다룰 벤치마크 오염, 평가 과적합, LLM-as-a-Judge의 신뢰성 문제로 직결된다. 정렬 기법의 발전 속도가 평가 방법론의 발전 속도에 발목 잡히고 있다는 문제의식을 갖고 다음 논문들을 읽으면 좋다.
