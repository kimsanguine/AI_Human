# Daily AI Paper Recommendations

> **Date:** 2026-04-20
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2201.11903
- **PDF:** [./chain-of-thought-prompting-wei-2022.pdf](./chain-of-thought-prompting-wei-2022.pdf)
- **Citation Count:** 9,000+ (NeurIPS 2022)

### 요약
이 논문은 대규모 언어 모델(LLM)에 "단계별로 생각해보자"와 같이 중간 추론 과정을 예시로 제시하는 Chain-of-Thought(CoT) 프롬프팅 기법을 제안했다. 단순한 입력-출력 예시가 아닌 추론 체인을 포함한 few-shot 예시를 통해, 540B 파라미터 규모의 PaLM 모델이 GSM8K 수학 벤치마크에서 기존 SOTA를 능가하는 58%의 정확도를 달성했다. 이 기법은 모델 파라미터 업데이트 없이 프롬프트만으로 수학, 상식, 상징적 추론 태스크에서 극적인 성능 향상을 보였다.

### 핵심 기여
- 중간 추론 단계(rationale)를 few-shot 예시에 포함하는 단순하면서도 강력한 프롬프팅 기법 제시
- CoT는 모델 규모가 충분히 클 때만 효과가 나타나는 "emergent ability"임을 실증 (약 100B 파라미터 이상)
- 수학적 추론(GSM8K, SVAMP, MAWPS), 상식 추론(CSQA, StrategyQA), 상징적 추론(Letter Concat, Coin Flip) 전반에서 일관된 성능 향상 증명

### 이 논문이 중요한 이유
CoT는 모든 현대 프롬프트 엔지니어링의 출발점이자 가장 근본적인 테크닉이다. "Let's think step by step"으로 대표되는 이 기법은 이후 Self-Consistency, Tree-of-Thoughts, ReAct, OpenAI o1/o3의 추론 모델까지 파생된 모든 reasoning 기반 LLM 시스템의 이론적 토대를 제공한다. AI 엔지니어라면 CoT가 왜 작동하는지, 어떤 조건에서 효과가 있는지 반드시 이해해야 한다.

### 사전 지식
- Few-shot / in-context learning 개념 (GPT-3 논문)
- Transformer 기반 LLM의 기본 구조
- 벤치마크 태스크 이해: GSM8K(수학), MMLU, CommonsenseQA
- Scaling law와 emergent abilities에 대한 기초 지식

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682)

### 실무 적용
- **RAG 파이프라인의 답변 생성 단계**: 검색된 문서를 바탕으로 CoT 프롬프트를 구성해 근거 기반 추론을 유도
- **Agent 시스템**: ReAct 패턴과 결합하여 "Thought → Action → Observation" 루프 구현
- **복잡한 고객 문의 자동화**: 다단계 의사결정이 필요한 B2B SaaS 지원 워크플로우에서 CoT로 오류율 감소
- **에이전틱 AI 제품**: 태스크 분해(task decomposition)의 핵심 메커니즘으로 CoT를 활용해 멀티스텝 작업 신뢰성 향상
- **프로덕션 팁**: 비용이 중요하면 CoT 출력을 짧게 제약하거나 "draft → final" 두 단계로 분리하여 토큰 비용 최적화

---

## Paper 2 (Classic): Language Models are Few-Shot Learners (GPT-3)
- **Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. (OpenAI)
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2005.14165
- **PDF:** [./language-models-are-few-shot-learners-brown-2020.pdf](./language-models-are-few-shot-learners-brown-2020.pdf)
- **Citation Count:** 40,000+ (NeurIPS 2020 Best Paper)

### 요약
GPT-3 논문은 175B 파라미터 규모의 autoregressive 언어 모델이 파라미터 업데이트 없이도 단지 몇 개의 예시(few-shot)만으로 다양한 NLP 태스크를 수행할 수 있음을 보였다. 저자들은 zero-shot, one-shot, few-shot 세 가지 설정으로 모델 크기를 125M부터 175B까지 확장하며 실험했고, 모델이 커질수록 in-context learning 능력이 비약적으로 향상됨을 증명했다. 이는 "학습된 가중치는 그대로 두고 프롬프트만 바꿔서 태스크를 푼다"는 현대 LLM 활용 패러다임의 출발점이다.

### 핵심 기여
- In-context learning(ICL) 개념을 대규모로 실증: 파라미터 fine-tuning 없이 프롬프트만으로 태스크 수행
- 모델 스케일이 커질수록 few-shot 성능이 비선형적으로 향상된다는 경험적 법칙 확립
- 번역, QA, cloze, reasoning, 코드 생성 등 수십 개 벤치마크에서 범용 LLM의 잠재력 공개
- 프롬프트 설계(데이터 형식, 예시 수, 예시 순서)가 모델 성능에 미치는 영향을 체계적으로 분석

### 이 논문이 중요한 이유
GPT-3 논문은 "LLM을 API로 호출해 제품을 만드는 시대"를 열었다. Fine-tuning 없이 프롬프트 엔지니어링만으로 프로토타입을 빠르게 검증할 수 있게 되면서 AI 제품 개발의 속도와 비용 구조가 근본적으로 바뀌었다. CPO 관점에서 보면 이 논문 이후 "데이터를 모으고 모델을 학습시키는 회사"에서 "프롬프트와 파이프라인을 설계하는 회사"로 AI 제품 기업의 정의가 변했다. 모든 ChatGPT, Claude, Gemini 기반 제품의 사상적 기원이다.

### 사전 지식
- Transformer decoder 구조 (GPT-2 논문)
- Language modeling의 기본 개념(next-token prediction)
- Fine-tuning vs. in-context learning의 차이
- Scaling law의 초기 형태에 대한 이해

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners / GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)

### 실무 적용
- **MVP 프로토타입**: Fine-tuning 파이프라인을 구축하기 전에 few-shot 프롬프트로 가설 빠르게 검증
- **분류/추출 태스크**: 3~5개 예시만으로 도메인 특화 분류기를 학습 없이 구현
- **프롬프트 템플릿 시스템**: 제품 내에서 예시를 동적으로 선택·주입하는 retrieval 기반 ICL 구현
- **비용 최적화**: 태스크 난이도에 따라 zero-shot → few-shot → CoT로 단계적으로 확장하며 토큰 예산 관리
- **Evaluation 설계**: in-context 예시 수·순서·포맷을 체계적으로 A/B 테스트하여 production 프롬프트 고도화

---

## Paper 3 (Recent): Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot
- **Authors:** (2025 arXiv preprint — refer to arXiv metadata for the full author list)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2506.14641
- **PDF:** [./revisiting-cot-prompting-zero-shot-2025.pdf](./revisiting-cot-prompting-zero-shot-2025.pdf)
- **Citation Count:** 신규 논문 (2025년 6월 공개) — 빠르게 인용 축적 중

### 요약
이 논문은 Qwen2.5 등 최신 강력한 LLM에서 전통적인 Chain-of-Thought few-shot 예시를 추가하는 것이 Zero-Shot CoT 대비 추론 성능을 향상시키지 못한다는 놀라운 발견을 제시한다. 저자들은 few-shot 예시의 주된 역할이 실제로 추론 능력을 가르치는 것이 아니라 출력 형식을 사람의 기대에 맞춰 정렬시키는 것임을 실증적으로 보였다. 이는 Wei et al. (2022)의 원래 CoT 논문의 결과가 2026년 현재의 강력한 모델에서는 재현되지 않을 수 있음을 시사한다.

### 핵심 기여
- 최신 instruction-tuned LLM에서 few-shot CoT가 zero-shot CoT 대비 이점이 없거나 오히려 해가 됨을 다수의 벤치마크에서 증명
- Few-shot 예시의 진짜 기능은 "reasoning 학습"이 아니라 "output format alignment"임을 분리 실험으로 규명
- 현대 LLM 시대에 맞는 새로운 프롬프트 설계 가이드라인 제시: 간결한 zero-shot + 명확한 출력 포맷 지시

### 이 논문이 중요한 이유
Wei 2022의 CoT 논문이 여전히 바이블처럼 인용되는 현실에서, 이 논문은 "이 기법이 아직도 유효한가?"를 2025년 관점에서 재평가한다. Instruction tuning과 RLHF가 보편화된 지금 prompt engineering의 best practice는 크게 바뀌었고, AI 엔지니어는 과거의 통념을 그대로 답습하기보다 현재 모델의 특성에 맞는 새로운 패턴을 채택해야 한다. 프로덕션 환경에서 few-shot 예시 제거는 곧 토큰 비용 절감과 latency 감소로 직결되므로 실무적 함의도 크다.

### 사전 지식
- Chain-of-Thought prompting (Paper 1)
- Zero-shot CoT / "Let's think step by step" (Kojima et al., 2022)
- Instruction tuning과 RLHF 개념
- 최신 오픈소스 모델 생태계 (Qwen2.5, Llama 3, DeepSeek 등)

### 관련 논문
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)
- [Qwen2.5 Technical Report (Qwen Team, 2024)](https://arxiv.org/abs/2412.15115)
- [Meta Prompting for AI Systems (Zhang et al., 2023)](https://arxiv.org/abs/2311.11482)

### 실무 적용
- **토큰 비용 최적화**: Few-shot 예시를 제거하고 zero-shot + 명확한 포맷 지시로 대체하여 입력 토큰 30~70% 절감
- **Latency 개선**: 예시가 길어질수록 input processing time이 선형 증가 — zero-shot 전환으로 p99 latency 감소
- **프롬프트 리팩토링 가이드**: 기존 제품에서 few-shot 예시를 쓰고 있다면, 최신 모델 기준으로 A/B 테스트를 거쳐 제거 여부 판단
- **Agentic AI 설계**: 에이전트의 각 도구 호출 프롬프트에서 예시 대신 structured output(JSON schema) 강제로 성능·비용 균형
- **CPO 관점**: "벤더 모델이 업그레이드될 때마다 프롬프트를 재평가한다"는 운영 원칙을 팀에 내재화

---

## 추천 읽기 순서
1. **Paper 2 (GPT-3)** → in-context learning이라는 패러다임의 기원부터 이해
2. **Paper 1 (CoT)** → 그 위에 얹어진 가장 중요한 프롬프팅 기법의 원리 학습
3. **Paper 3 (Revisiting CoT, 2025)** → 2026년 현재 관점에서 Paper 1의 결론이 어떻게 바뀌고 있는지 비판적으로 검토

## 핵심 테이크어웨이
- **In-context learning**은 GPT-3 이후 모든 LLM 제품의 작동 원리 — 모델이 크면 "예시로 가르치는" 것이 가능하다
- **Chain-of-Thought**는 중간 추론을 드러내어 복잡한 태스크의 정확도를 끌어올리는 가장 간단한 기법이고, 모든 reasoning 모델(o1, o3, DeepSeek-R1 등)의 사상적 기원이다
- **하지만 2025년 이후**, 강력한 instruction-tuned 모델에서는 few-shot CoT의 상당 부분이 "형식 정렬" 용도로 축소되었다는 실증 연구가 등장 — 프롬프트 설계는 고정된 레시피가 아니라 모델 세대에 맞춰 진화해야 한다
- AI 엔지니어는 **"어떤 기법을 언제 쓸 것인가"** 를 경험적으로 판단할 수 있어야 하며, 이를 위해 항상 자체 평가셋(eval set)으로 A/B 테스트해야 한다
- **비용, 지연시간, 정확도**의 3-way trade-off를 의식하고 프롬프트를 설계하는 것이 프로덕션 AI 제품의 핵심 역량이다

## 다음 토픽과의 연결
내일(Day 19, Module 7)은 **Advanced Prompting — Tree-of-Thoughts, ReAct, Self-Consistency**를 다룬다. 오늘 학습한 Chain-of-Thought는 "단일 선형 추론 체인"이라는 한계가 있는데, Tree-of-Thoughts는 여러 추론 경로를 탐색하고 ReAct는 추론과 외부 도구 사용(action)을 결합한다. 특히 Agentic AI 제품을 리딩하는 CPO 관점에서는 CoT → ReAct로의 진화가 "순수 추론기"에서 "세계와 상호작용하는 에이전트"로의 질적 도약을 의미하므로 이 연결을 주의 깊게 살펴볼 것.
