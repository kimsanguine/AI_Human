# Daily AI Paper Recommendations

> **Date:** 2026-10-17
> **Module:** Module 4: NLP and Speech Data
> **Topic:** BERT and Pre-trained Language Models

---

## Paper 1 (Classic): Unified Language Model Pre-training for Natural Language Understanding and Generation (UniLM)
- **Authors:** Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon
- **Year:** 2019 (NeurIPS 2019)
- **arXiv:** https://arxiv.org/abs/1905.03197
- **PDF:** [./unilm-dong-2019.pdf](./unilm-dong-2019.pdf)
- **Citation Count:** ~1,800+

### 요약
UniLM은 Transformer 하나로 단방향(left-to-right / right-to-left), 양방향, sequence-to-sequence 세 가지 언어 모델링 목표를 함께 학습합니다. 목표마다 따로 모델을 두지 않고 **self-attention mask**만 바꿔서 각 토큰이 볼 수 있는 문맥을 조절합니다. 그 결과 BERT처럼 이해(NLU) 태스크를 잘하면서도 요약·질문 생성 같은 생성(NLG) 태스크까지 한 모델로 fine-tuning할 수 있습니다.

### 핵심 기여
- **Attention mask 하나로 여러 LM 목표 통합:** 아키텍처를 바꾸지 않고 마스크 행렬만으로 양방향, 단방향, seq2seq를 전환
- **NLU와 NLG를 한 모델로:** GLUE, SQuAD 2.0, CoQA에서 BERT를 앞서고, CNN/DailyMail 요약과 Gigaword에서도 당시 SOTA 달성
- **공유 파라미터로 얻는 일반화:** 여러 목표를 함께 학습해 하나의 목표에만 과적합하는 것을 줄임

### 이 논문이 중요한 이유
"encoder는 이해, decoder는 생성"이라는 이분법에 처음으로 진지하게 문제를 제기한 논문입니다. 오늘날 LLM에서 쓰는 **prefix LM, causal mask, bidirectional prefix** 같은 개념(T5의 prefix-LM, GLM, UL2의 mixture-of-denoisers 등)이 모두 이 논문의 "마스크로 문맥을 설계한다"는 아이디어를 이어받았습니다. Attention mask를 직접 설계해 보는 경험은 AI 엔지니어에게 꼭 필요합니다.

### 사전 지식
- Transformer self-attention과 attention mask 동작 원리
- BERT의 Masked Language Modeling(MLM), GPT의 causal LM 차이
- Seq2seq 구조(encoder-decoder)와 생성 태스크 평가지표(ROUGE)

### 관련 논문
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training (Bao et al., 2020)](https://arxiv.org/abs/2002.12804)
- [UL2: Unifying Language Learning Paradigms (Tay et al., 2022)](https://arxiv.org/abs/2205.05131)
- [GLM: General Language Model Pretraining with Autoregressive Blank Infilling (Du et al., 2021)](https://arxiv.org/abs/2103.10360)

### 실무 적용
- **마스크 설계가 곧 기능 설계:** 챗봇의 system prompt를 양방향으로 처리하는 prefix-LM, 문서 검색과 요약을 하나로 처리하는 모델 등은 모두 마스크 전략에서 출발합니다.
- **멀티태스크 서비스의 모델 수 줄이기:** 분류, 추출, 요약을 각각 다른 모델로 서빙하는 대신 통합 모델 하나로 운영해 인프라 비용을 낮출 수 있습니다.
- 커스텀 학습 코드를 짤 때 `attention_mask`를 4D로 직접 구성하는 패턴(packing, prefix caching)을 이해하는 기초가 됩니다.

---

## Paper 2 (Classic): MPNet: Masked and Permuted Pre-training for Language Understanding
- **Authors:** Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu
- **Year:** 2020 (NeurIPS 2020)
- **arXiv:** https://arxiv.org/abs/2004.09297
- **PDF:** [./mpnet-song-2020.pdf](./mpnet-song-2020.pdf)
- **Citation Count:** ~1,300+

### 요약
MPNet은 BERT의 MLM과 XLNet의 Permuted LM(PLM)이 가진 약점을 함께 해결합니다. MLM은 마스킹된 토큰끼리의 의존성을 무시하고, PLM은 학습 중 전체 문장 위치 정보를 보지 못해 학습과 추론 사이에 차이(position discrepancy)가 생깁니다. MPNet은 순열(permutation)로 토큰 간 의존성을 학습하면서 **보조 위치 정보(auxiliary position)**를 함께 넣어 모델이 전체 문장 길이와 구조를 볼 수 있게 했습니다.

### 핵심 기여
- **MLM과 PLM의 장점 결합:** 예측 토큰 간 의존성(PLM 장점)과 전체 위치 정보(MLM 장점)를 함께 확보
- **Position discrepancy 해결:** 마스크 토큰에 위치 임베딩을 붙여 pre-training과 fine-tuning 입력 분포를 맞춤
- **같은 조건에서 BERT, XLNet, RoBERTa보다 우수:** GLUE와 SQuAD에서 일관된 개선을 보임

### 이 논문이 중요한 이유
MPNet은 논문보다 **실무에서 더 유명한 모델**입니다. Sentence-Transformers의 대표 모델 `all-mpnet-base-v2`의 backbone으로 쓰이면서 수년간 RAG와 시맨틱 검색의 기본 embedding 모델 역할을 했습니다. "pre-training 목표가 다운스트림 embedding 품질을 어떻게 바꾸는가"를 이해하는 좋은 사례이고, Module 9(RAG)로 이어지는 연결고리입니다.

### 사전 지식
- BERT의 MLM과 그 한계(마스크된 토큰끼리 독립 가정)
- XLNet의 Permutation Language Modeling과 two-stream attention
- 위치 임베딩(absolute / relative position)

### 관련 논문
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al., 2019)](https://arxiv.org/abs/1906.08237)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
- [MASS: Masked Sequence to Sequence Pre-training (Song et al., 2019)](https://arxiv.org/abs/1905.02450)

### 실무 적용
- **RAG 검색 품질 기준선:** `all-mpnet-base-v2`는 지금도 새 embedding 모델을 평가할 때 많이 쓰는 baseline입니다. 새 모델 도입 전 A/B 비교 대상으로 두면 좋습니다.
- **가볍고 빠른 온프레미스 검색:** 110M 파라미터 규모라 CPU나 작은 GPU에서도 서빙할 수 있어 B2B 고객사 내부 설치형 검색에 적합합니다.
- 분류, 중복 탐지, 클러스터링 같은 NLU 파이프라인에서 LLM 호출 비용을 줄이는 경량 대안이 됩니다.

---

## Paper 3 (Recent): Seq vs Seq: An Open Suite of Paired Encoders and Decoders (Ettin)
- **Authors:** Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, Benjamin Van Durme
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2507.11412
- **PDF:** [./ettin-seq-vs-seq-weller-2025.pdf](./ettin-seq-vs-seq-weller-2025.pdf)
- **Citation Count:** ~50+ (2025 공개, 빠르게 증가 중)

### 요약
Ettin은 **같은 데이터, 같은 레시피, 같은 크기**로 encoder-only 모델과 decoder-only 모델을 쌍으로 학습해 공개한 모델 모음입니다(17M~1B 파라미터, 최대 2T 토큰). 이 설정 덕분에 "encoder냐 decoder냐"를 처음으로 공정하게 비교할 수 있었습니다. 결과는 분명합니다. 분류와 검색은 encoder가, 생성은 decoder가 우세하고, 한쪽 모델을 반대 목표로 이어서 학습(continued training)해도 처음부터 그 목표로 학습한 모델을 따라잡지 못했습니다.

### 핵심 기여
- **통제된 비교 실험:** 데이터와 학습 조건을 동일하게 맞춘 encoder-decoder 쌍을 6개 크기로 공개
- **아키텍처 차이의 정량화:** 400M encoder가 1B decoder보다 MNLI에서 우수함 → 이해·검색 태스크에서 encoder의 효율성 입증
- **"변환"의 한계 확인:** decoder를 encoder로 바꾸는 방식(LLM2Vec 계열)은 네이티브 encoder보다 성능이 떨어짐
- **완전 공개:** 학습 데이터, 학습 순서, 200개가 넘는 체크포인트를 모두 공개해 재현 연구 가능

### 이 논문이 중요한 이유
2024~2025년 "LLM으로 embedding도 하면 되지 않나?"라는 흐름에 대해 데이터로 답한 논문입니다. ModernBERT 레시피를 기반으로 하고 있어, 이 주제의 이전 회차들(ModernBERT, NeoBERT, MLM 재검토 논문)과 함께 읽으면 **2025년 encoder 연구의 흐름**이 완성됩니다. AI 엔지니어가 "이 태스크에 어떤 아키텍처를 쓸 것인가"를 근거를 들어 결정할 수 있게 해 줍니다.

### 사전 지식
- Encoder-only(BERT 계열)와 decoder-only(GPT 계열)의 학습 목표 차이
- ModernBERT 아키텍처(RoPE, alternating attention, unpadding)
- Embedding 모델 평가(MTEB, BEIR)와 분류 벤치마크(GLUE)

### 관련 논문
- [Smarter, Better, Faster, Longer: ModernBERT (Warner et al., 2024)](https://arxiv.org/abs/2412.13663)
- [Should We Still Pretrain Encoders with Masked Language Modeling? (Gisserot-Boukhlef et al., 2025)](https://arxiv.org/abs/2507.00994)
- [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders (BehnamGhader et al., 2024)](https://arxiv.org/abs/2404.05961)
- [NeoBERT: A Next-Generation BERT (Le Breton et al., 2025)](https://arxiv.org/abs/2502.19587)

### 실무 적용
- **아키텍처 선택 가이드:** 검색, 분류, 라우팅, 가드레일 같은 "판단" 태스크는 작은 encoder로, 응답 생성은 decoder로 나누는 하이브리드 구조가 비용 대비 효과가 좋습니다.
- **Agent 시스템의 비용 절감:** intent 분류나 tool routing을 LLM 호출 대신 150M급 Ettin encoder로 처리하면 지연 시간과 비용을 크게 줄일 수 있습니다.
- 크기별로 encoder-decoder 쌍이 있어서, 자사 데이터로 "우리 태스크에서는 어느 쪽이 나은가"를 직접 실험하기 좋은 출발점입니다.

---

## 추천 읽기 순서
1. **UniLM (2019)** — "마스크가 곧 모델의 성격을 정한다"는 관점을 먼저 잡습니다. Attention mask 그림(Figure 1)을 꼭 직접 따라 그려 보세요.
2. **MPNet (2020)** — BERT(MLM)와 XLNet(PLM)의 한계를 어떻게 보완했는지 보며 pre-training 목표 설계의 trade-off를 이해합니다.
3. **Ettin (2025)** — 앞의 두 논문이 던진 "이해와 생성을 한 모델로?"라는 질문에 2025년 대규모 통제 실험이 어떤 답을 냈는지 확인합니다.

## 핵심 테이크어웨이
- **Q. Pre-trained LM의 성격은 무엇이 정하나?** → 아키텍처보다 **학습 목표와 attention mask**가 더 큰 영향을 줍니다(UniLM).
- **Q. 학습 목표를 바꾸면 무엇이 달라지나?** → 토큰 간 의존성과 위치 정보를 어떻게 다루느냐에 따라 다운스트림 품질, 특히 embedding 품질이 달라집니다(MPNet).
- **Q. 그럼 LLM 하나로 다 하면 되지 않나?** → 같은 조건에서 비교하면 이해·검색은 encoder, 생성은 decoder가 우세합니다. "하나로 통합"보다 **"태스크에 맞는 조합"**이 2025년의 답입니다(Ettin).
- **가설 제안:** 우리 서비스의 LLM 호출 중 분류·라우팅 성격이 큰 비중을 차지한다면, 소형 encoder로 바꿔도 품질은 비슷하고 비용과 지연은 크게 줄 것이다 → 로그 샘플로 검증해 볼 만합니다.

## 다음 토픽과의 연결
다음 주제는 **Speech Recognition Fundamentals**입니다. 텍스트에서 BERT가 한 "대규모 self-supervised pre-training → fine-tuning" 패러다임은 음성에서 wav2vec 2.0, HuBERT로 그대로 이어집니다. MPNet과 UniLM에서 본 "마스킹 전략" 역시 음성 프레임을 마스킹해 표현을 학습하는 방식과 맞닿아 있습니다. CTC와 Attention 기반 ASR을 공부할 때 오늘 배운 encoder-decoder 구조와 마스크 개념을 떠올려 보세요.
