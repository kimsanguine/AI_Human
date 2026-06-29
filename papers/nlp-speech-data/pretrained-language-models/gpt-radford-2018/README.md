# Syllabus Essential (교안 보강): GPT (Generative Pre-Training)

> **Category:** nlp-speech-data / pretrained-language-models
> **Why added:** 교안에 등장하는 핵심 4종(Attention·BERT·GPT·Word2Vec) 중 GPT만 repo에 전용 항목이 없어 보강. BERT(인코더 사전학습)의 짝인 디코더 사전학습 원조
> **Type:** Classic (Must-Read)

## Paper: Improving Language Understanding by Generative Pre-Training
- **Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- **Year:** 2018
- **Source:** OpenAI Technical Report (arXiv 미등재) — https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- **PDF:** [./gpt-improving-language-understanding-radford-2018.pdf](./gpt-improving-language-understanding-radford-2018.pdf)
- **Citation Count:** 약 13,000+ (Google Scholar 기준)

### 요약
대규모 비라벨 텍스트로 Transformer 디코더를 언어모델(다음 단어 예측)로 사전학습한 뒤, 각 다운스트림 과제에 맞게 미세조정(fine-tuning)하는 2단계 패러다임을 제안한다. 과제별 입력을 통일된 토큰 시퀀스로 변환하는 task-aware 입력 변환으로, 모델 구조를 거의 바꾸지 않고 다양한 NLP 과제에 적용한다.

### 핵심 기여
- "비지도 생성적 사전학습 → 지도 미세조정"이라는 전이학습 레시피를 NLP에 정립
- Transformer 디코더 기반 단방향 언어모델로 범용 표현 학습
- 12개 과제 중 9개에서 당시 SOTA를 달성하며 task-agnostic 사전학습의 가능성 입증

### 이 논문이 중요한 이유
오늘날 ChatGPT로 이어지는 GPT 계열(GPT-2/3/4)의 원점이다. BERT가 양방향 인코더 사전학습이라면 GPT는 단방향 디코더 사전학습으로, 둘의 대비는 현대 LLM 아키텍처 선택(인코더 vs 디코더 vs 인코더-디코더)을 이해하는 핵심 축이다. 생성형 LLM의 사상적 출발점이라 교안의 필수 논문이다.

### 사전 지식
- Transformer, 특히 디코더와 causal(마스킹) self-attention
- 언어모델링 목표(다음 토큰 예측)와 우도 최대화
- 전이학습: 사전학습-미세조정 패러다임

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)

### 실무 적용
모든 생성형 LLM 서비스(챗봇, 코드 어시스턴트, 콘텐츠 생성)의 기반 사고다. "범용 사전학습 모델 + 과제별 적응"이라는 패턴은 오늘날 파인튜닝·인스트럭션 튜닝·프롬프팅으로 진화했으며, 인코더(BERT)와 디코더(GPT)의 용도 차이를 이해하는 것은 NLP 제품 설계의 기본 의사결정이다.
