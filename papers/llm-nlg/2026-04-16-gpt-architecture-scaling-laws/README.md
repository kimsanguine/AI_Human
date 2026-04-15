# Daily AI Paper Recommendations

> **Date:** 2026-04-16
> **Module:** LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

---

## Paper 1 (Classic): Language Models are Few-Shot Learners (GPT-3)
- **Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, et al.
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2005.14165
- **PDF:** [./gpt3-few-shot-learners-brown-2020.pdf](./gpt3-few-shot-learners-brown-2020.pdf)
- **Citation Count:** 30,000+ (AI 분야 최상위권)

### 요약
175B 파라미터 규모의 autoregressive 언어 모델 GPT-3를 제안하고, fine-tuning 없이 **few-shot / one-shot / zero-shot** 만으로 수많은 NLP 태스크에서 SOTA에 근접한 성능을 달성함을 보인 논문이다. 모델의 규모(scale)가 커질수록 meta-learning 능력이 창발적으로 나타난다는 것을 실증한 기념비적 작업이다.

### 핵심 기여
- **In-context learning 개념의 확립**: 가중치 업데이트 없이 프롬프트 내 예시만으로 태스크를 수행하는 능력을 대규모로 입증.
- **규모의 힘 실증**: 125M부터 175B까지 8개 모델 크기 스펙트럼에서 zero/one/few-shot 성능이 예측 가능하게 향상됨을 보임.
- **범용 언어 모델의 가능성 제시**: 번역, QA, 산술, 코드, 에세이 생성 등 광범위한 태스크를 단일 모델로 처리.

### 이 논문이 중요한 이유
GPT-3는 "AI 엔지니어링" 시대의 출발점을 만든 논문이다. 모든 후속 LLM(ChatGPT, Claude, Llama, Gemini)의 기반이 되는 **scale + in-context learning** 패러다임을 확립했으며, Agentic AI와 Prompt Engineering이라는 새로운 분야 자체를 만들어냈다. 현재의 AI 제품 설계자라면 이 논문이 제시한 "모델은 학습하지 않고, 프롬프트로 조종한다"는 관점을 이해해야 한다.

### 사전 지식
- Transformer 구조 (Attention Is All You Need)
- Autoregressive Language Modeling의 기본 학습 목적함수
- Fine-tuning vs. Transfer Learning의 차이
- Perplexity, BLEU, LAMBADA 등의 NLP 평가 지표

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training / GPT-1 (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners / GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### 실무 적용
- **프롬프트 엔지니어링**: Few-shot 예시를 프롬프트에 삽입하는 모든 현업 LLM 워크플로우의 이론적 근거.
- **제품 전략 결정**: "Fine-tuning vs. Prompting" 의사결정에서 Prompting이 우선적으로 고려되는 배경을 제공.
- **AI Native 사고**: 하나의 Foundation Model을 여러 제품 기능으로 재활용하는 멀티 유즈케이스 아키텍처의 출발점.

---

## Paper 2 (Classic): Scaling Laws for Neural Language Models
- **Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2001.08361
- **PDF:** [./scaling-laws-neural-language-models-kaplan-2020.pdf](./scaling-laws-neural-language-models-kaplan-2020.pdf)
- **Citation Count:** 6,000+

### 요약
언어 모델의 loss가 **모델 크기 N, 데이터셋 크기 D, 컴퓨트 예산 C**에 대해 멱법칙(power law)으로 예측 가능하게 감소함을 실증한 논문이다. 주어진 컴퓨트 예산 하에서 어떻게 N과 D를 배분해야 최적의 성능이 나오는지에 대한 정량적 공식을 처음 제시했다.

### 핵심 기여
- **3개 변수의 멱법칙 관계 도출**: L(N) ∝ N^-0.076, L(D) ∝ D^-0.095, L(C) ∝ C^-0.050 형태의 예측 가능한 scaling 관계 발견.
- **Compute-optimal 학습**: 고정된 컴퓨트 예산에서 모델을 끝까지 수렴시키는 것보다 더 큰 모델을 덜 학습시키는 것이 유리함을 제안 (이후 Chinchilla에서 보정됨).
- **Sample efficiency 증명**: 큰 모델이 작은 모델보다 샘플 효율성이 좋다는 것을 수학적으로 정식화.

### 이 논문이 중요한 이유
이 논문이 없었다면 GPT-3, PaLM, Llama 등 초거대 모델에 대한 투자가 정당화되지 못했을 것이다. **"더 크게 만들면 더 좋아진다"**는 AI 산업의 투자 논리를 제공한 핵심 문헌이며, CPO/CTO 레벨에서 모델 훈련 예산을 배분할 때 의사결정 프레임워크가 된다. Chinchilla, Llama 3까지 이어지는 scaling 연구의 원점이다.

### 사전 지식
- Cross-entropy loss와 언어 모델 목적함수
- FLOPs(부동소수점 연산) 계산 방법
- Log-log 플롯에서의 멱법칙 해석
- 모델 파라미터 수 계산 (dense layer, attention head 구성)

### 관련 논문
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Scaling Laws for Autoregressive Generative Modeling (Henighan et al., 2020)](https://arxiv.org/abs/2010.14701)
- [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682)

### 실무 적용
- **모델 예산 기획**: 파인튜닝/프리트레이닝 프로젝트에서 GPU 시간을 얼마나 할당할지 결정.
- **제품 로드맵**: "지금 5B 모델을 학습시킬 것인가, 데이터를 더 모을 것인가"와 같은 전략적 선택의 근거.
- **그로스 관점**: 모델 성능 → 유저 리텐션/Win rate의 역산을 통해 R&D ROI를 평가하는 데이터 드리븐 프로세스에 활용.

---

## Paper 3 (Recent): The Llama 3 Herd of Models
- **Authors:** Llama Team, AI @ Meta (Abhimanyu Dubey et al., 500+ authors)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.21783
- **PDF:** [./llama3-herd-of-models-meta-2024.pdf](./llama3-herd-of-models-meta-2024.pdf)
- **Citation Count:** 3,500+ (2024년 공개 이후 급증)

### 요약
405B 파라미터의 dense Transformer를 포함한 Llama 3 시리즈를 공개하며, 다국어·코딩·추론·도구 사용 능력을 갖춘 오픈소스 foundation model을 선보인 논문이다. GPT-4 수준의 품질을 공개 가중치로 제공하며, 15T+ 토큰 사전학습, 128K 컨텍스트, 이미지/비디오/음성 모달리티 통합까지 다룬 종합 리포트이다.

### 핵심 기여
- **오픈소스 SOTA**: 405B 모델이 GPT-4, Claude 3.5 Sonnet과 비교 가능한 벤치마크 성능을 공개 가중치로 달성.
- **Compute-optimal을 넘어선 over-training**: Chinchilla 가이드보다 훨씬 많은 15.6T 토큰을 학습시켜 추론 비용 효율을 극대화하는 전략 제시.
- **Post-training 레시피 공개**: Rejection sampling + SFT + DPO 반복 루프의 상세 파이프라인을 문서화.
- **멀티모달 compositional 접근**: Vision/Speech encoder를 별도로 학습 후 LLM과 어댑터로 결합하는 실용적 방법론.

### 이 논문이 중요한 이유
현재 대부분의 B2B/B2C AI 제품이 기반으로 삼는 open-weight LLM의 최전선이다. **오픈소스 vs. 클로즈드 API**라는 제품 전략 의사결정에서 Llama 3를 이해하지 못하면 아키텍처 선택이 편향된다. 특히 Agentic AI 제품에서는 fine-tuning, on-prem 배포, 비용 최적화를 위한 대안으로 Llama 3 계열의 채택이 급증하고 있다.

### 사전 지식
- GPT-3 논문(특히 Decoder-only Transformer와 scaling law 개념)
- RLHF / DPO의 기본 원리
- Mixture-of-Experts (MoE) 구조의 장단점 (Llama 3는 dense를 선택한 이유 이해)
- Grouped-Query Attention(GQA), RoPE 등 현대 Transformer 최적화 기법

### 관련 논문
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288)
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Qwen2 Technical Report (Yang et al., 2024)](https://arxiv.org/abs/2407.10671)

### 실무 적용
- **제품 배포 전략**: 고객사 on-prem 요구, 데이터 주권(data sovereignty) 요건이 있는 B2B SaaS에서 Llama 3를 기반으로 한 커스텀 파인튜닝 제품 설계.
- **비용 구조 최적화**: API 기반 GPT-4 대비 자체 호스팅 Llama 3의 단가 분석 모델을 구성하는 기준점.
- **Agentic AI 개발**: Tool-use와 긴 컨텍스트(128K)가 필요한 agentic 워크플로우에서 오픈소스 옵션으로서의 실전 선택지.

---

## 추천 읽기 순서
1. **Scaling Laws (Kaplan et al., 2020)** — 왜 크기가 중요한가라는 "공리"를 먼저 설치한다.
2. **GPT-3 (Brown et al., 2020)** — 그 공리가 실제 모델에서 어떻게 발현되는지 확인한다 (in-context learning의 등장).
3. **Llama 3 (Meta AI, 2024)** — 앞선 두 원칙이 4년 후 어떻게 세련되고 공학적으로 확장됐는지(데이터 큐레이션, post-training, 멀티모달) 이해한다.

## 핵심 테이크어웨이
- **Scale은 전략이다**: 모델 크기·데이터·컴퓨트의 멱법칙 관계를 이해해야 R&D 투자 의사결정이 가능하다.
- **Prompting이 Fine-tuning을 대체할 수 있다**: In-context learning은 제품 빌드 속도를 10배 이상 당긴다. 이는 PM 관점에서 feature 출시 cycle 자체를 바꾼다.
- **오픈소스는 제품 차별화 축이 될 수 있다**: Llama 3는 "클로즈드 API에 락인되지 않는" 제품 전략의 실존적 대안이다.
- **Compute-optimal은 고정 공식이 아니다**: Llama 3는 Chinchilla 가이드보다 과도하게 학습(over-train)함으로써 추론 비용을 낮췄다. 학습 효율과 추론 효율은 다른 최적화 문제임을 기억해야 한다.

## 다음 토픽과의 연결
내일 주제인 **Instruction Tuning and RLHF**는 오늘 읽은 Llama 3 post-training 파이프라인의 출발점인 InstructGPT/DPO로 이어진다. 오늘의 세 논문이 "Pre-training"의 스케일링 문제를 다뤘다면, 다음 토픽은 "사전학습된 거인을 어떻게 사용자 의도에 정렬(align)할 것인가"의 문제로 넘어간다. Agentic AI 제품의 UX 품질은 대부분 이 정렬 단계에서 결정된다.
