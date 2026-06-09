# Daily AI Paper Recommendations

> **Date:** 2026-06-09
> **Module:** Module 6 — LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

---

## Paper 1 (Classic): Language Models are Unsupervised Multitask Learners (GPT-2)
- **Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Year:** 2019
- **arXiv:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf (OpenAI 공식 PDF)
- **PDF:** [./gpt2-language-models-unsupervised-multitask-learners-radford-2019.pdf](./gpt2-language-models-unsupervised-multitask-learners-radford-2019.pdf)
- **Citation Count:** 약 12,000+ 회

### 요약
GPT-2는 1.5B 파라미터의 디코더-only 트랜스포머로, 별도의 태스크별 파인튜닝 없이 대규모 웹 텍스트(WebText)만으로 언어 모델링을 학습했다. 핵심 주장은 충분히 큰 언어 모델이 명시적 지도 없이도 독해, 번역, 요약, QA 같은 다양한 다운스트림 태스크를 "제로샷"으로 수행할 수 있다는 것이다.

### 핵심 기여
- 언어 모델링 목표만으로 멀티태스크 학습이 emergent하게 발생함을 실증 (zero-shot task transfer)
- WebText라는 고품질 대규모 웹 코퍼스 구축 및 데이터 품질의 중요성 강조
- 모델 크기(117M → 1.5B)를 키울수록 제로샷 성능이 로그-선형적으로 향상됨을 관찰 — 이후 스케일링 법칙 연구의 토대

### 이 논문이 중요한 이유
오늘날 모든 GPT 계열 LLM의 직접적 조상이다. "파인튜닝 없이 프롬프트만으로 태스크를 푼다"는 현재 LLM 사용 패러다임(in-context learning, prompt engineering)의 출발점이며, 스케일이 능력을 만든다는 직관을 처음으로 설득력 있게 보여줬다.

### 사전 지식
- 트랜스포머 디코더 구조와 self-attention (Vaswani et al., 2017)
- 언어 모델링(다음 토큰 예측)과 perplexity
- 전이학습/파인튜닝 vs 제로샷의 차이

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training / GPT-1 (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)

### 실무 적용
프롬프트 기반 제로/퓨샷 추론이라는 현재 LLM 활용의 기본 동작 원리를 이해하는 출발점이다. 파인튜닝 비용 없이 프롬프트 설계만으로 신규 태스크를 빠르게 프로토타이핑하는 전략, 그리고 모델 크기 선택이 곧 성능 trade-off라는 의사결정의 근거가 된다.

---

## Paper 2 (Classic): PaLM — Scaling Language Modeling with Pathways
- **Authors:** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. (Google Research)
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2204.02311
- **PDF:** [./palm-scaling-language-modeling-chowdhery-2022.pdf](./palm-scaling-language-modeling-chowdhery-2022.pdf)
- **Citation Count:** 약 7,000+ 회

### 요약
PaLM은 540B 파라미터의 밀집(dense) 디코더-only 트랜스포머로, Google의 Pathways 시스템을 사용해 6,144개 TPU v4 칩에서 학습되었다. 수백 개 벤치마크에서 SOTA 퓨샷 성능을 달성했고, 특히 다단계 추론(chain-of-thought) 과제에서 스케일이 만드는 비약적(emergent) 성능 향상을 보였다.

### 핵심 기여
- 540B 규모 dense 모델의 효율적 학습을 가능케 한 Pathways 분산 학습 인프라 제시
- CoT 프롬프팅과 결합 시 산술/상식 추론에서 인간 평균을 능가하는 emergent ability 입증
- 모델 규모와 추론 능력의 비선형적(불연속적) 관계를 대규모로 관찰

### 이 논문이 중요한 이유
스케일링이 단순히 perplexity를 낮추는 것을 넘어 "추론" 같은 질적으로 새로운 능력을 발현시킨다는 점을 가장 명확하게 보여준 대표작이다. emergent abilities 논쟁과 CoT 연구의 실증적 근거이며, 초대형 모델의 시스템적 학습 방법론도 함께 제시한다.

### 사전 지식
- GPT-3와 퓨샷 in-context learning
- Chain-of-Thought 프롬프팅 개념
- 데이터/모델/연산 병렬화 등 분산 학습 기초

### 관련 논문
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682)

### 실무 적용
모델 규모 선택과 학습 예산 배분의 근거를 제공한다. "어느 규모부터 특정 능력이 켜지는가"를 이해하면, 제품에 필요한 능력(예: 멀티스텝 추론)에 맞는 모델 티어를 고르고, 작은 모델로는 안 되는 과제를 조기에 판별할 수 있다.

---

## Paper 3 (Recent): Qwen2.5 Technical Report
- **Authors:** Qwen Team (Alibaba)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2412.15115
- **PDF:** [./qwen2.5-technical-report-qwen-2024.pdf](./qwen2.5-technical-report-qwen-2024.pdf)
- **Citation Count:** 약 900+ 회 (2024년 12월 공개, 빠르게 증가 중)

### 요약
Qwen2.5는 0.5B부터 72B까지의 오픈웨이트 LLM 시리즈로, 사전학습 데이터를 7T에서 18T 토큰으로 확장하고 사후학습(SFT+RLHF)을 대폭 강화했다. 플래그십 Qwen2.5-72B-Instruct는 약 5배 큰 Llama-3-405B-Instruct에 필적하는 성능을 보이며, 언어 이해·추론·수학·코딩 전반에서 최상위권을 기록했다.

### 핵심 기여
- 18T 토큰 규모의 고품질 데이터 큐레이션과 데이터 스케일링 전략 공개
- 긴 컨텍스트(최대 128K) 지원 및 구조화 출력·도구 호출 능력 강화
- 다양한 크기(0.5B~72B)의 모델 패밀리로 배포 환경별 성능/비용 trade-off 제공

### 이 논문이 중요한 이유
2024년 가장 널리 채택된 오픈웨이트 LLM 중 하나로, GPT-2/PaLM이 제시한 스케일링 원리가 최신 프로덕션급 모델에서 데이터 스케일링과 사후학습으로 어떻게 정교화되었는지 보여준다. 고전 논문의 아이디어가 현재 실무 모델로 이어지는 연결고리다.

### 사전 지식
- 디코더-only 트랜스포머와 GPT 계열 아키텍처
- 사전학습/사후학습(SFT, RLHF/DPO) 파이프라인
- 컨텍스트 길이 확장(RoPE 등) 기초 개념

### 관련 논문
- [The Llama 3 Herd of Models (Meta, 2024)](https://arxiv.org/abs/2407.21783)
- [Qwen2.5-Coder Technical Report (Hui et al., 2024)](https://arxiv.org/abs/2409.12186)
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437)

### 실무 적용
온프레미스/프라이빗 배포에 적합한 오픈웨이트 모델 선택의 실전 레퍼런스다. 작은 모델로 비용을 줄이거나, 긴 컨텍스트·도구 호출이 필요한 에이전트 워크로드에 맞는 티어를 고르는 등, 제품 요구사항에 따른 모델 라인업 구성 의사결정에 직접 활용된다.

---

## 추천 읽기 순서
1. **GPT-2 (2019)** — "스케일이 능력을 만든다"는 직관과 제로샷 패러다임의 출발점
2. **PaLM (2022)** — 그 직관이 초대형 규모에서 emergent reasoning으로 확장되는 지점
3. **Qwen2.5 (2024)** — 고전의 원리가 데이터 스케일링·사후학습으로 정교화된 최신 프로덕션 모델

## 핵심 테이크어웨이
- 언어 모델링이라는 단일 목표만으로도 규모가 커지면 멀티태스크·추론 능력이 emergent하게 발현된다.
- 스케일링은 파라미터뿐 아니라 데이터(품질·양)와 사후학습이 함께 작동할 때 실제 제품 성능으로 전환된다.
- 모델 크기는 곧 능력/비용의 trade-off이며, 제품 요구 능력에 맞는 티어 선택이 핵심 의사결정이다.

## 다음 토픽과의 연결
다음 토픽인 **Instruction Tuning과 RLHF**는 이번에 다룬 "사전학습된 거대 모델"을 사람의 의도에 맞게 정렬(alignment)하는 단계다. Qwen2.5가 강조한 사후학습 파이프라인이 어떻게 InstructGPT/DPO 계열 기법으로 구현되는지 이어서 살펴본다.
