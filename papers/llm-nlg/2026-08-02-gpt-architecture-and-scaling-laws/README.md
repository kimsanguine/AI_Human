# Daily AI Paper Recommendations

> **Date:** 2026-08-02
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

---

## Paper 1 (Classic): Improving Language Understanding by Generative Pre-Training (GPT-1)
- **Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- **Year:** 2018
- **arXiv:** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf (OpenAI Technical Report)
- **PDF:** [./gpt1-improving-language-understanding-radford-2018.pdf](./gpt1-improving-language-understanding-radford-2018.pdf)
- **Citation Count:** ~13,000+

### 요약
GPT 계열의 출발점이 된 논문으로, 대규모 비지도 텍스트로 Transformer 디코더를 생성적(generative) 사전학습한 뒤 각 다운스트림 태스크에 판별적(discriminative) 파인튜닝을 적용하는 2단계 패러다임을 제안했다. 태스크별 특수 아키텍처 없이도 단일 범용 모델이 12개 과제 중 9개에서 SOTA를 갱신하며, "먼저 언어를 사전학습하고 나중에 태스크에 적응시킨다"는 현대 LLM의 근간을 세웠다.

### 핵심 기여
- 대규모 비지도 사전학습 + 태스크별 파인튜닝이라는 전이학습 패러다임을 언어 모델에 확립
- 태스크마다 입력을 구조화된 시퀀스로 변환(traversal-style)하여, 아키텍처 변경 없이 단일 모델로 분류·함의·유사도·QA를 처리
- Transformer 디코더가 LSTM 기반보다 긴 문맥 전이에 유리함을 실증하고, 사전학습 표현의 zero-shot 잠재력을 관찰

### 이 논문이 중요한 이유
오늘날 모든 GPT/LLM이 따르는 "사전학습(pre-train) → 적응(fine-tune/prompt)" 구조의 원형이다. AI 엔지니어가 GPT-2·GPT-3·InstructGPT로 이어지는 스케일업의 계보를 이해하려면 그 뿌리인 GPT-1의 설계 결정을 먼저 읽어야 한다.

### 사전 지식
- Transformer 디코더와 self-attention (Attention Is All You Need)
- 언어 모델링 목적함수(next-token prediction)와 전이학습 개념
- 파인튜닝 시 보조 언어모델링 손실(auxiliary LM objective)의 역할

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners / GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

### 실무 적용
파운데이션 모델을 특정 도메인에 적용할 때의 표준 워크플로(사전학습 체크포인트 로드 → 다운스트림 파인튜닝)가 여기서 시작되었다. 입력을 구조화된 시퀀스로 변환하는 아이디어는 오늘날 프롬프트 템플릿/포맷 설계로 이어진다.

---

## Paper 2 (Classic): Emergent Abilities of Large Language Models
- **Authors:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2206.07682
- **PDF:** [./emergent-abilities-llm-wei-2022.pdf](./emergent-abilities-llm-wei-2022.pdf)
- **Citation Count:** ~4,500+

### 요약
모델 규모를 키울 때, 작은 모델에는 존재하지 않다가 특정 규모를 넘어서면 갑자기 나타나는 능력을 "창발적 능력(emergent abilities)"으로 정의한 논문이다. 산술, 다단계 추론, 지시 따르기 등 다수 과제에서 성능이 규모에 대해 부드럽게 증가하지 않고 임계점에서 급격히 도약하는 현상을 정리했다. 이는 스케일이 단순한 성능 개선을 넘어 질적으로 새로운 능력을 열 수 있음을 시사한다.

### 핵심 기여
- 창발(emergence)을 "규모를 키우기 전에는 예측 불가능한 능력의 출현"으로 명확히 정의하고 다수 벤치마크로 유형화
- few-shot 프롬프팅, chain-of-thought, instruction following 등에서 나타나는 임계점(phase transition)형 성능 향상을 문서화
- 스케일링 법칙(부드러운 예측)과 창발(불연속 도약)의 관계 및 한계를 논의

### 이 논문이 중요한 이유
"모델을 더 키우면 무엇을 얻는가"라는 질문에 대한 핵심 참조점이다. 왜 특정 크기 이하 모델에서는 CoT·지시 따르기가 잘 안 되는지, 언제 소형 모델 대신 대형 모델이 필요한지를 판단하는 근거를 제공한다. 이후 창발이 평가 지표의 산물일 수 있다는 반론(Schaeffer et al.)까지 이어지는 중요한 논쟁의 출발점이다.

### 사전 지식
- 스케일링 법칙(Kaplan 2020, Chinchilla)과 loss-vs-compute 곡선
- few-shot in-context learning과 chain-of-thought 프롬프팅
- 벤치마크 평가 지표(정확도, exact match)의 특성

### 관련 논문
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2023)](https://arxiv.org/abs/2304.15004)

### 실무 적용
모델 규모 선정과 비용/성능 트레이드오프 의사결정에 직접 쓰인다. "이 태스크가 창발 임계점 위에 있는가"를 따져 소형 파인튜닝 모델로 충분한지, 대형 모델의 프롬프팅이 필요한지를 판단하며, 신규 능력 벤치마킹 설계의 기준이 된다.

---

## Paper 3 (Recent): Gemma 2 — Improving Open Language Models at a Practical Size
- **Authors:** Gemma Team, Google DeepMind
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2408.00118
- **PDF:** [./gemma2-improving-open-language-models-google-2024.pdf](./gemma2-improving-open-language-models-google-2024.pdf)
- **Citation Count:** ~900+

### 요약
2B~27B 규모의 오픈 웨이트 모델 제품군으로, Transformer에 여러 검증된 개선(로컬-글로벌 어텐션 교차 배치, group-query attention 등)을 적용했다. 특히 2B·9B 모델을 next-token prediction 대신 대형 교사 모델로부터의 지식 증류(knowledge distillation)로 학습해, 동급 대비 최고 성능과 2~3배 큰 모델에 필적하는 경쟁력을 달성했다. "무작정 키우기"가 아니라 실용적 크기에서 스케일링 효율을 끌어올리는 방향을 보여준다.

### 핵심 기여
- 소형 모델(2B·9B) 학습에 지식 증류를 도입해, 동일 토큰 예산에서 얻는 품질을 크게 향상
- 로컬-글로벌 어텐션 교차와 GQA로 긴 문맥 처리 효율과 추론 비용을 개선한 실용적 아키텍처 제시
- 오픈 웨이트로 공개해 동급 크기에서 SOTA를 세우고, "실용적 크기(practical size)" 스케일링의 레시피를 문서화

### 이 논문이 중요한 이유
GPT-1→GPT-3의 "크게 만들수록 좋다"는 흐름에서, 2024년 현재의 관심사인 "제한된 크기에서 최대 효율"로 이동한 지점을 대표한다. 온디바이스/저비용 배포가 중요한 실무에서 증류·어텐션 설계가 스케일링 못지않게 성능을 좌우함을 보여준다.

### 사전 지식
- Group-Query Attention(GQA)과 로컬/슬라이딩 윈도우 어텐션
- 지식 증류(teacher-student, soft label) 학습
- Chinchilla 최적점과 토큰/파라미터 예산 개념

### 관련 논문
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [The Llama 3 Herd of Models (Dubey et al., 2024)](https://arxiv.org/abs/2407.21783)
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)

### 실무 적용
자원 제약 환경(온디바이스, 사내 서버)에서 9B급 오픈 모델로 대형 모델급 품질을 내야 할 때의 실전 참조다. 증류 기반 소형 모델 학습, GQA를 통한 추론 비용 절감, 오픈 웨이트 파인튜닝 파이프라인 설계에 바로 활용된다.

---

## 추천 읽기 순서
1. **GPT-1 (2018)** — 사전학습→파인튜닝 패러다임의 원형을 먼저 이해한다. GPT 계보의 출발점.
2. **Emergent Abilities (2022)** — 그 패러다임을 "규모"로 밀어붙였을 때 무엇이 질적으로 달라지는지 본다.
3. **Gemma 2 (2024)** — 규모 경쟁을 넘어, 실용적 크기에서 효율(증류·어텐션 설계)로 성능을 얻는 최신 방향을 확인한다.

## 핵심 테이크어웨이
- 현대 LLM의 뼈대는 "대규모 비지도 사전학습 → 태스크 적응"이며, 그 원형은 GPT-1이다.
- 스케일업은 성능을 부드럽게 올릴 뿐 아니라 특정 임계점에서 새로운 능력(창발)을 열 수 있다. 다만 이는 평가 방식에 따라 해석이 갈리는 논쟁적 개념이다.
- 2024년의 전선은 "무작정 크게"가 아니라 "제한된 크기에서 최대 효율"이다. 지식 증류와 어텐션 아키텍처 최적화가 스케일링과 함께 핵심 레버가 된다.

## 다음 토픽과의 연결
다음 토픽인 **Instruction Tuning과 RLHF**는 오늘 다룬 사전학습된 대형 모델을 "사람의 의도에 정렬"시키는 단계다. GPT-1의 파인튜닝, 창발적 지시 따르기 능력, 그리고 Gemma 2 같은 오픈 베이스 모델이 어떻게 InstructGPT·DPO 등의 정렬 기법으로 이어지는지 자연스럽게 연결된다.
