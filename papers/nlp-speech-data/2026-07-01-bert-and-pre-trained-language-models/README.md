# Daily AI Paper Recommendations

> **Date:** 2026-07-01
> **Module:** Module 4: NLP and Speech Data
> **Topic:** BERT and Pre-trained Language Models

---

## Paper 1 (Classic): Deep contextualized word representations (ELMo)
- **Authors:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
- **Year:** 2018
- **arXiv:** [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)
- **PDF:** [./elmo-peters-2018.pdf](./elmo-peters-2018.pdf)
- **Citation Count:** ~16,000+ (Google Scholar 기준)

### 요약
ELMo는 대규모 텍스트로 학습한 양방향 LSTM 언어 모델의 내부 상태를 결합해 "문맥을 반영한 단어 표현(contextualized word representation)"을 만들어낸 논문이다. Word2Vec/GloVe처럼 단어마다 고정된 벡터를 쓰는 대신, 같은 단어라도 문장 속 맥락에 따라 다른 벡터를 부여한다. 기존 모델에 ELMo 임베딩을 얹기만 해도 6개 주요 NLP 과제에서 SOTA를 갱신했다.

### 핵심 기여
- 사전학습된 양방향 언어 모델(biLM)의 여러 층 표현을 태스크별 가중치로 선형 결합하는 방식을 제안 (하위 층=구문, 상위 층=의미).
- 단어의 다의성(polysemy)을 문맥에 따라 구분하는 표현을 학습해 downstream 성능을 크게 향상.
- 기존 아키텍처를 바꾸지 않고 임베딩 층만 교체하는 "feature-based" 전이학습 패러다임을 확립, BERT로 이어지는 사전학습 시대의 문을 열었다.

### 이 논문이 중요한 이유
ELMo는 "고정 임베딩 → 문맥 임베딩"이라는 패러다임 전환의 출발점이다. 왜 BERT/RoBERTa 같은 문맥 인코더가 필요한지, 사전학습-전이학습이 왜 NLP를 지배하게 됐는지 이해하려면 반드시 짚어야 할 다리 역할의 논문이다. AI 엔지니어라면 임베딩이 "단어 사전"이 아니라 "문장을 읽고 만든 표현"으로 진화한 지점을 여기서 확인할 수 있다.

### 사전 지식
- 단어 임베딩(Word2Vec, GloVe)의 개념과 한계(고정 벡터, 다의어 처리 불가)
- RNN/LSTM과 언어 모델링(다음 단어 예측) 기본 개념
- 전이학습에서 feature-based 방식과 fine-tuning 방식의 차이

### 관련 논문
- [Efficient Estimation of Word Representations in Vector Space / Word2Vec (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Universal Language Model Fine-tuning / ULMFiT (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146)

### 실무 적용
검색·분류·개체명 인식 등 태스크에서 문맥 임베딩을 피처로 활용하는 초기 파이프라인의 원형이다. 오늘날에는 Transformer 인코더로 대체되었지만, "사전학습된 표현을 임베딩 층에 주입한다"는 설계 사고는 임베딩 기반 검색(RAG), 리랭킹, 도메인 특화 표현 학습에 그대로 이어진다.

---

## Paper 2 (Classic): Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
- **Authors:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- **Year:** 2020 (arXiv 2019)
- **arXiv:** [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
- **PDF:** [./t5-raffel-2019.pdf](./t5-raffel-2019.pdf)
- **Citation Count:** ~22,000+ (Google Scholar 기준)

### 요약
T5는 번역, 요약, 분류, QA 등 모든 NLP 과제를 "텍스트 입력 → 텍스트 출력"이라는 단일 형식으로 통일한 인코더-디코더 사전학습 모델이다. 저자들은 C4라는 대규모 정제 코퍼스를 만들고, 사전학습 목표·아키텍처·데이터·크기 등 전이학습의 거의 모든 선택지를 체계적으로 비교 실험했다. 그 결과를 하나의 프레임워크로 정리해 사전학습 연구의 "실험 지도"를 제시했다.

### 핵심 기여
- 모든 태스크를 text-to-text 형식으로 환원해, 하나의 모델·손실·디코딩으로 다양한 과제를 처리하는 통일된 프레임워크를 제안.
- 사전학습 목표(마스킹 방식), 모델 구조, 데이터 크기, 학습량, 스케일링 등을 대규모로 ablation 하여 실증적 가이드라인을 제시.
- 정제된 대규모 웹 코퍼스 C4(Colossal Clean Crawled Corpus)를 공개.

### 이 논문이 중요한 이유
"모든 것을 텍스트 생성으로 본다"는 T5의 관점은 이후 GPT 계열의 instruction/prompt 패러다임과 맞닿아 있다. 인코더(BERT)와 인코더-디코더(T5)의 설계 철학 차이, 그리고 사전학습에서 어떤 변수가 성능을 좌우하는지 데이터로 배우기에 최적의 논문이다. LLM 시대의 "task를 프롬프트로 표현한다"는 사고의 뿌리를 이해할 수 있다.

### 사전 지식
- Transformer의 인코더-디코더 구조와 self-attention (Attention Is All You Need)
- BERT의 마스킹 언어 모델링(MLM)과 사전학습-파인튜닝 개념
- 전이학습, 시퀀스-투-시퀀스(seq2seq) 학습의 기본

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., 2019)](https://arxiv.org/abs/1910.13461)

### 실무 적용
T5는 요약, 번역, 데이터-투-텍스트, 쿼리 재작성 등 생성형 태스크의 강력한 베이스라인으로 여전히 활용된다(FLAN-T5 등 instruction 튜닝 버전 포함). "태스크를 입력 프롬프트로 인코딩한다"는 방식은 오늘날 LLM 프롬프트 엔지니어링과 멀티태스크 파인튜닝 설계의 직접적 조상이다.

---

## Paper 3 (Recent): mmBERT: A Modern Multilingual Encoder with Annealed Language Learning
- **Authors:** Marc Marone, Orion Weller, William Fleshman, Eugene Yang, Dawn Lawrie, Benjamin Van Durme
- **Year:** 2025
- **arXiv:** [https://arxiv.org/abs/2509.06888](https://arxiv.org/abs/2509.06888)
- **PDF:** [./mmbert-marone-2025.pdf](./mmbert-marone-2025.pdf)
- **Citation Count:** ~수십 회 (2025년 발표, 빠르게 증가 중)

### 요약
mmBERT는 ModernBERT 아키텍처를 다국어로 확장한 최신 인코더 모델로, 1800개 이상 언어의 약 3조 토큰으로 사전학습되었다. 학습 후반부에 저자원 언어 비중을 점진적으로 높이는 "annealed language learning(어닐링 커리큘럼)"과 inverse masking, 온도 기반 샘플링을 도입해 고자원·저자원 언어 성능을 동시에 끌어올렸다. ModernBERT의 긴 문맥·추론 효율을 유지하면서 XLM-R, EuroBERT 등 기존 다국어 인코더를 능가한다.

### 핵심 기여
- 학습 스케줄 후반에 저자원 언어 데이터를 늘리는 annealed(점진적) 언어 커리큘럼으로, 저자원 언어 표현을 효율적으로 확보.
- inverse masking과 온도 기반 언어 샘플링을 결합해 1800+ 언어, 256K 어휘 규모에서도 균형 잡힌 다국어 표현을 학습.
- ModernBERT 계열의 긴 문맥(long-context)·고효율 추론을 유지하면서 분류·임베딩·검색 태스크에서 SOTA 다국어 성능 달성.

### 이 논문이 중요한 이유
ModernBERT(영어 중심)와 NeoBERT 이후, "현대적 인코더를 다국어로 어떻게 잘 학습시키는가"에 대한 실전적 레시피를 제시한 논문이다. 임베딩·검색·분류처럼 생성이 필요 없는 태스크에서 인코더 모델은 여전히 가장 비용 효율적인 선택이며, 다국어·저자원 언어 서비스를 만들 때 직접적인 백본 후보가 된다.

### 사전 지식
- BERT의 MLM 사전학습과 인코더-only 모델의 용도(분류, 임베딩, 검색)
- ModernBERT의 개선점(RoPE, 긴 문맥, 효율적 어텐션)
- 다국어 사전학습의 어려움(언어 불균형, 토크나이저·어휘 크기 문제)

### 관련 논문
- [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder / ModernBERT (Warner et al., 2024)](https://arxiv.org/abs/2412.13663)
- [Unsupervised Cross-lingual Representation Learning at Scale / XLM-R (Conneau et al., 2019)](https://arxiv.org/abs/1911.02116)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)

### 실무 적용
다국어 검색(RAG)의 임베딩·리랭커, 다국어 문서 분류, 저자원 언어 NLU 서비스의 백본으로 바로 활용 가능하다. 한국어를 포함한 비영어권 제품에서 영어 전용 ModernBERT보다 유리하며, 생성형 LLM 대비 낮은 지연·비용으로 임베딩/분류 파이프라인을 구성할 때 강력한 선택지가 된다.

---

## 추천 읽기 순서
1. **ELMo (2018)** — 고정 임베딩에서 문맥 임베딩으로의 전환점. 왜 사전학습이 필요한지 감을 잡는다.
2. **T5 (2020)** — 사전학습의 설계 변수들을 대규모 실험으로 정리. 인코더 vs 인코더-디코더 관점을 넓힌다.
3. **mmBERT (2025)** — 위 흐름이 다국어·현대 인코더로 어떻게 발전했는지 최신 레시피로 확인한다.

## 핵심 테이크어웨이
- **표현의 진화:** 단어별 고정 벡터(Word2Vec) → 문맥 반영 표현(ELMo) → Transformer 인코더(BERT/T5) → 현대 다국어 인코더(mmBERT)로 이어지는 큰 흐름을 이해하는 것이 핵심이다.
- **사전학습은 레시피 싸움:** T5가 보여주듯 목표·데이터·스케일 같은 선택이 성능을 좌우한다. 아키텍처 하나만이 아니라 학습 전략이 중요하다.
- **인코더는 여전히 유효:** 생성이 필요 없는 검색·분류·임베딩에서는 인코더-only 모델이 비용·지연 측면에서 LLM보다 실용적이며, mmBERT처럼 계속 발전하고 있다.

## 다음 토픽과의 연결
다음 토픽인 **Speech Recognition Fundamentals(Day 10)** 로 넘어가면, 텍스트에서 확립된 사전학습·표현학습 사고가 음성 신호로 확장된다. wav2vec 2.0의 자기지도 사전학습처럼, 여기서 배운 "대규모 비지도 사전학습 → 다운스트림 전이"라는 패러다임이 음성 도메인에서 어떻게 재현되는지 비교하며 읽으면 좋다.
