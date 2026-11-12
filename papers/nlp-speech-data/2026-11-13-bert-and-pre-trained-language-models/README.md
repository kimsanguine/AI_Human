# Daily AI Paper Recommendations

> **Date:** 2026-11-13
> **Module:** Module 4: NLP and Speech Data
> **Topic:** BERT and Pre-trained Language Models

---

## Paper 1 (Classic): Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)
- **Authors:** Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov
- **Year:** 2019 (ACL 2020)
- **arXiv:** https://arxiv.org/abs/1911.02116
- **PDF:** [./xlm-r-cross-lingual-conneau-2019.pdf](./xlm-r-cross-lingual-conneau-2019.pdf)
- **Citation Count:** ~8,000+

### 요약
CommonCrawl에서 정제한 100개 언어, 2.5TB 텍스트로 RoBERTa 방식의 Masked Language Model을 대규모 학습한 다국어 인코더 XLM-R을 제안합니다. 병렬 코퍼스 없이 단일 언어 데이터만으로 학습했는데도 mBERT를 XNLI 기준 평균 +14.6%p, MLQA F1 +13%p 앞질렀고, 저자원 언어(스와힐리어, 우르두어 등)에서 개선 폭이 특히 컸습니다.

### 핵심 기여
- **데이터 규모가 다국어 성능의 핵심 변수**임을 입증: Wikipedia 대신 CC-100(2.5TB)을 사용해 저자원 언어 데이터를 수십~수백 배 확보
- **"Curse of Multilinguality" 정량화**: 고정 용량에서 언어 수를 늘리면 처음엔 전이 효과로 좋아지다가 어느 시점부터 언어당 성능이 떨어지는 트레이드오프를 실험으로 제시하고, 모델 용량 확장으로 완화 가능함을 보임
- **언어 샘플링 온도(α=0.3)와 대형 어휘(250K SentencePiece)** 등 다국어 사전학습의 실전 레시피 정리
- 다국어 모델이 GLUE에서 단일 언어 RoBERTa와 경쟁할 수 있음을 보여 "다국어 = 성능 손해"라는 통념을 반박

### 이 논문이 중요한 이유
한국어를 포함한 비영어권 서비스를 만드는 AI 엔지니어에게 XLM-R은 지금도 다국어 분류·NER·임베딩의 기본 베이스라인입니다. 또한 "Curse of Multilinguality"는 이후 mT5, BLOOM, mmBERT, 그리고 다국어 LLM의 데이터 믹스 설계에서 반복적으로 인용되는 개념이라, 다국어 모델의 용량·데이터 배분을 고민할 때 출발점이 됩니다.

### 사전 지식
- BERT의 MLM 사전학습과 RoBERTa의 학습 개선점(동적 마스킹, NSP 제거, 대규모 배치)
- SentencePiece/BPE 서브워드 토크나이저
- Zero-shot cross-lingual transfer 개념 (영어로 파인튜닝 → 다른 언어로 평가)

### 관련 논문
- [Cross-lingual Language Model Pretraining / XLM (Lample & Conneau, 2019)](https://arxiv.org/abs/1901.07291)
- [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer (Xue et al., 2020)](https://arxiv.org/abs/2010.11934)
- [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning (Marone et al., 2025)](https://arxiv.org/abs/2509.06888)

### 실무 적용
- **다국어 고객지원·콘텐츠 분류**: 영어 라벨 데이터만으로 파인튜닝한 뒤 한국어·일본어·스페인어 문의를 분류하는 zero-shot 전이
- **AI 더빙/로컬라이제이션 파이프라인**: 원문·번역문 품질 검수(QE), 언어 식별, 다국어 NER의 백본
- **다국어 임베딩**: multilingual-e5, BGE-M3 등 다국어 임베딩 모델이 XLM-R을 백본으로 사용하므로 RAG 시스템에서도 간접적으로 계속 쓰이고 있음

---

## Paper 2 (Classic): Don't Stop Pretraining: Adapt Language Models to Domains and Tasks
- **Authors:** Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith
- **Year:** 2020 (ACL 2020, Honorable Mention)
- **arXiv:** https://arxiv.org/abs/2004.10964
- **PDF:** [./dont-stop-pretraining-gururangan-2020.pdf](./dont-stop-pretraining-gururangan-2020.pdf)
- **Citation Count:** ~3,000+

### 요약
범용 사전학습 모델(RoBERTa)을 바로 파인튜닝하지 말고, 타깃 도메인 텍스트로 **한 번 더 사전학습(DAPT)**하거나 타깃 태스크의 비라벨 텍스트로 **추가 사전학습(TAPT)**하면 성능이 일관되게 오른다는 것을 4개 도메인(바이오, CS 논문, 뉴스, 리뷰)·8개 태스크에서 체계적으로 보였습니다. DAPT 후 TAPT를 이어 하는 조합이 가장 좋았습니다.

### 핵심 기여
- **DAPT(Domain-Adaptive Pretraining)**: 도메인 코퍼스로 MLM 추가 학습 → 고자원·저자원 설정 모두에서 성능 향상
- **TAPT(Task-Adaptive Pretraining)**: 훨씬 작은 태스크 비라벨 데이터만으로도 DAPT에 근접하는 효율적 향상
- **도메인 유사도 분석**: 어휘 겹침으로 도메인 간 거리를 측정하고, 엉뚱한 도메인으로 적응하면 오히려 성능이 떨어짐을 확인
- kNN 기반 데이터 선택으로 태스크와 유사한 비라벨 데이터를 골라 TAPT를 보강하는 방법 제시

### 이 논문이 중요한 이유
"범용 모델 + 파인튜닝"과 "도메인 특화 모델" 사이의 질문에 데이터로 답한 논문입니다. 오늘날 LLM 시대의 **continued pretraining(지속 사전학습)**—법률·의료·금융 LLM, 사내 코드 모델 등—의 이론적·실험적 근거가 되며, "라벨링 전에 비라벨 데이터부터 활용하라"는 실무 원칙을 정립했습니다.

### 사전 지식
- 사전학습(pretraining)과 파인튜닝(fine-tuning)의 구분
- RoBERTa의 MLM 학습 방식
- 도메인 시프트(domain shift)와 분포 차이의 개념

### 관련 논문
- [SciBERT: A Pretrained Language Model for Scientific Text (Beltagy et al., 2019)](https://arxiv.org/abs/1903.10676)
- [BioBERT: a pre-trained biomedical language representation model (Lee et al., 2019)](https://arxiv.org/abs/1901.08746)
- [Universal Language Model Fine-tuning for Text Classification / ULMFiT (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146)

### 실무 적용
- **버티컬 SaaS**: 고객사 문서·티켓·로그(비라벨)로 인코더를 TAPT한 뒤 소량 라벨로 분류기를 학습 → 라벨링 비용 절감
- **도메인 LLM**: 법률/의료/금융 코퍼스로 오픈 LLM을 continued pretraining 후 instruction tuning하는 표준 2단계 파이프라인의 원형
- **의사결정 프레임**: "도메인 데이터 양 × 범용 모델과의 어휘 거리"로 DAPT 투자 여부를 판단하는 체크리스트로 활용

---

## Paper 3 (Recent): Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation (T5Gemma)
- **Authors:** Biao Zhang, Fedor Moiseev, Joshua Ainslie, Paul Suganthan, Min Ma, Surya Bhupatiraju, Fede Lebron, Orhan Firat, Armand Joulin, Zhe Dong
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2504.06225
- **PDF:** [./encoder-decoder-gemma-zhang-2025.pdf](./encoder-decoder-gemma-zhang-2025.pdf)
- **Citation Count:** 신규 논문 (2025년 4월 공개, T5Gemma 모델로 공개 및 후속작 T5Gemma 2 발표)

### 요약
처음부터 인코더-디코더를 학습하는 대신, **이미 사전학습된 decoder-only LLM(Gemma 2)의 가중치로 인코더와 디코더를 초기화**한 뒤 UL2 또는 PrefixLM 목표로 추가 학습하는 "적응(adaptation)" 방법을 제안합니다. 같은 추론 비용에서 decoder-only 원본과 비슷하거나 더 나은 품질을 보였고, 큰 인코더 + 작은 디코더(예: 9B-2B) 같은 비대칭 구성으로 품질-효율 트레이드오프를 유연하게 조절할 수 있습니다.

### 핵심 기여
- **Decoder-only → Encoder-Decoder 적응 레시피**: 디코더 가중치를 인코더(양방향 어텐션)와 디코더(cross-attention 추가)에 재사용
- **비대칭 구성(unbalanced)**: 입력 이해는 큰 인코더, 출력 생성은 작은 디코더에 맡겨 요약·번역 등 입력이 긴 태스크에서 추론 효율 개선
- **인코더 표현 품질 향상**: 적응된 인코더가 SuperGLUE 등 이해 태스크에서 decoder-only 표현보다 강한 결과 → BERT 계열 인코더의 현대적 대안 제시
- 사전학습·instruction tuning 이후 모두에서 decoder-only 대비 품질-효율 프런티어를 개선

### 이 논문이 중요한 이유
BERT(인코더) vs GPT(디코더) vs T5(인코더-디코더)로 갈라졌던 사전학습 아키텍처 논쟁이 LLM 시대에 어떻게 다시 합쳐지는지를 보여줍니다. "인코더를 따로 처음부터 학습하지 않고, 강력한 LLM을 재활용해 인코더를 만든다"는 흐름은 LLM2Vec, Ettin 등과 함께 2024-2025년 인코더 연구의 핵심 트렌드이며, 오늘 클래식 두 편(XLM-R·DAPT)이 다룬 "사전학습된 모델을 어떻게 재활용·적응시키는가"라는 질문의 최신 답변이기도 합니다.

### 사전 지식
- T5와 UL2의 사전학습 목표(span corruption, PrefixLM, mixture-of-denoisers)
- Self-attention과 cross-attention, causal mask vs bidirectional mask 차이
- Decoder-only LLM(Gemma/Llama)의 기본 구조

### 관련 논문
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer / T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)
- [UL2: Unifying Language Learning Paradigms (Tay et al., 2022)](https://arxiv.org/abs/2205.05131)
- [T5Gemma 2: Seeing, Reading, and Understanding Longer (2025)](https://arxiv.org/abs/2512.14856)

### 실무 적용
- **요약·번역·전사 후처리**: 입력이 길고 출력이 짧은 워크로드(회의록 요약, 자막 번역, STT 교정)에서 큰 인코더 + 작은 디코더로 지연시간·비용 절감
- **도메인 인코더 구축**: 사내 LLM 체크포인트를 인코더로 적응시켜 분류·검색·리랭킹용 백본으로 재활용
- **모델 선택 기준**: "입력 대비 출력 길이 비율"이 낮은 제품 기능이라면 decoder-only만 고집하지 말고 인코더-디코더 구성을 A/B 테스트할 근거

---

## 추천 읽기 순서
1. **Don't Stop Pretraining (2020)** — "사전학습 모델을 내 도메인에 맞추는 법"이라는 가장 실무적인 질문부터. 실험 설계가 명확해 읽기 쉽습니다.
2. **XLM-R (2019)** — 같은 RoBERTa 레시피를 "언어" 축으로 확장했을 때 데이터 규모와 모델 용량이 어떻게 상호작용하는지 이해합니다.
3. **Encoder-Decoder Gemma (2025)** — 앞의 두 편이 "데이터로 적응"이었다면, 이 논문은 "아키텍처를 적응"시키는 최신 방법입니다.

## 핵심 테이크어웨이
- **사전학습은 일회성이 아니다**: 도메인(DAPT)·태스크(TAPT)·언어(XLM-R)·아키텍처(T5Gemma) 어느 축으로든 "추가 적응"이 성능을 끌어올린다.
- **데이터 규모와 모델 용량은 함께 움직인다**: 언어·도메인을 늘릴수록 용량이 모자라면 오히려 손해(Curse of Multilinguality).
- **처음부터 학습하는 대신 재활용하라**: 2025년의 정답은 강력한 기존 체크포인트를 출발점으로 삼아 목적에 맞게 변형하는 것.
- **PM 관점의 질문**: "우리 제품의 차별점이 되는 비라벨 데이터는 무엇이고, 그것으로 모델을 한 번 더 적응시킬 가치가 있는가?"

## 다음 토픽과의 연결
다음 토픽은 **Speech Recognition Fundamentals**입니다. 오늘 본 "대규모 비라벨 데이터로 사전학습 → 다운스트림 적응" 패러다임은 음성에서도 그대로 반복됩니다. XLM-R의 다국어 확장 아이디어는 음성의 XLSR(wav2vec 2.0 다국어판)로 이어지고, 인코더-디코더 구조는 CTC·Listen-Attend-Spell을 거쳐 Whisper로 발전합니다. 텍스트 사전학습에서 익힌 개념을 음성 신호에 대입하며 읽어보세요.
