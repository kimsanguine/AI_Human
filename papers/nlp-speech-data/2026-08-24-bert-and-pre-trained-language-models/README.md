# Daily AI Paper Recommendations

> **Date:** 2026-08-24
> **Module:** Module 4: NLP and Speech Data
> **Topic:** BERT and Pre-trained Language Models

---

## Paper 1 (Classic): SpanBERT: Improving Pre-training by Representing and Predicting Spans
- **Authors:** Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy
- **Year:** 2019 (TACL 2020)
- **arXiv:** https://arxiv.org/abs/1907.10529
- **PDF:** [./spanbert-joshi-2019.pdf](./spanbert-joshi-2019.pdf)
- **Citation Count:** 약 3,500+

### 요약
SpanBERT는 BERT의 마스킹 전략을 근본적으로 재설계한 논문이다. 토큰을 랜덤하게 하나씩 가리는 대신 연속된 구간(span)을 통째로 가리고, span 양쪽 경계 토큰의 표현만으로 가려진 내용 전체를 복원하도록 학습한다(Span Boundary Objective). 동일한 데이터와 모델 크기에서 BERT를 일관되게 능가하며, 특히 질의응답과 상호참조 해결처럼 "구간을 골라내는" 태스크에서 큰 폭의 향상을 보였다.

### 핵심 기여
- Span Masking: 기하분포로 길이를 뽑아 연속 구간을 마스킹 — 개별 토큰의 지역적 단서에 의존하는 지름길 학습을 차단
- Span Boundary Objective(SBO): 경계 토큰 표현에 span 전체 정보를 압축시켜, 구간 단위 추론 능력을 명시적으로 학습
- NSP 제거 + 단일 긴 시퀀스 학습이 오히려 더 좋다는 것을 통제 실험으로 입증 (RoBERTa의 결론과 독립적으로 수렴)
- SQuAD 1.1 F1 94.6, SQuAD 2.0 F1 88.7, OntoNotes 상호참조 79.6 F1로 당시 SOTA 달성

### 이 논문이 중요한 이유
"사전학습 목표(objective)를 바꾸면 모델 크기를 키우지 않고도 성능이 오른다"는 것을 가장 깔끔하게 보여준 사례다. AI 엔지니어가 파인튜닝만 반복하다 벽에 부딪혔을 때, 데이터·아키텍처가 아니라 **학습 과제 설계** 자체를 손봐야 한다는 사고를 심어준다. 또한 span 단위 표현은 오늘날 RAG의 청킹·추출형 QA·엔티티 링킹 설계와 직결된다.

### 사전 지식
- BERT의 MLM/NSP 구조와 [MASK] 토큰 처리 방식
- SQuAD 형태의 추출형 QA에서 start/end 포인터 예측 방식
- 기하분포 샘플링, 그리고 마스킹 비율(15%)이 왜 하이퍼파라미터로 다뤄지는지

### 관련 논문
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al., 2020)](https://arxiv.org/abs/2003.10555)

### 실무 적용
문서에서 특정 구간을 뽑아내는 제품 — 계약서 조항 추출, 이력서 파싱, 의료 기록의 소견 구간 추출 — 에서 SpanBERT 계열 인코더는 여전히 강력한 기본값이다. LLM 프롬프팅으로 처리하면 비용과 지연이 크고 근거 위치(offset)를 신뢰하기 어려운 반면, span 예측 모델은 원문 좌표를 그대로 반환하므로 하이라이팅·감사(audit) 기능을 붙이기 쉽다. RAG 파이프라인에서도 검색된 문단에서 정답 구간만 잘라 LLM 컨텍스트를 줄이는 리랭킹/압축 단계에 활용된다.

---

## Paper 2 (Classic): BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
- **Authors:** Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer
- **Year:** 2019 (ACL 2020)
- **arXiv:** https://arxiv.org/abs/1910.13461
- **PDF:** [./bart-lewis-2019.pdf](./bart-lewis-2019.pdf)
- **Citation Count:** 약 14,000+

### 요약
BART는 인코더-디코더(seq2seq) 트랜스포머를 "손상된 문서를 원문으로 복원하는" 디노이징 오토인코더로 사전학습한다. 토큰 마스킹, 토큰 삭제, 문장 순서 섞기, 문서 회전, 텍스트 인필링 등 임의의 노이즈를 허용하기 때문에 BERT(마스킹만)와 GPT(좌→우 생성만)의 제약에서 벗어난다. 결과적으로 이해(GLUE, SQuAD)에서는 RoBERTa급, 생성(요약, 대화, QA 생성)에서는 당시 최고 성능을 동시에 달성했다.

### 핵심 기여
- 통합 프레임워크: BERT(양방향 인코더)와 GPT(자기회귀 디코더)를 하나의 seq2seq 구조로 일반화
- 노이즈 함수 체계적 비교: text infilling(가변 길이 span을 하나의 [MASK]로 치환)이 가장 효과적임을 실험으로 규명 — 모델이 "몇 개가 빠졌는지"까지 추론해야 함
- 이해와 생성을 하나의 사전학습으로 동시에 커버 — CNN/DailyMail 요약에서 기존 대비 최대 6 ROUGE 향상
- 기계번역 응용: 인코더 앞에 새 인코더를 얹어 back-translation 없이 성능 개선

### 이 논문이 중요한 이유
오늘날 LLM은 대부분 디코더 온리지만, **"사전학습을 노이즈 복원 문제로 정의한다"**는 BART의 시각은 T5, 그리고 최근의 fill-in-the-middle(FIM) 코드 모델까지 이어지는 계보의 출발점이다. 또 BART는 "인코더-디코더가 필요한 순간"을 판단하는 기준을 제공한다 — 입력을 충분히 양방향으로 읽어야 하고 출력 길이가 입력과 크게 다를 때(요약, 번역, 정규화)가 그 지점이다.

### 사전 지식
- 트랜스포머 인코더-디코더 구조와 cross-attention의 역할
- 자기회귀 디코딩, teacher forcing, beam search
- ROUGE 지표의 의미와 한계
- BERT(인코더 온리)와 GPT(디코더 온리)의 구조적 차이

### 관련 논문
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer / T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Efficient Training of Language Models to Fill in the Middle (Bavarian et al., 2022)](https://arxiv.org/abs/2207.14255)

### 실무 적용
요약·번역·문장 교정처럼 입출력이 짝지어진 태스크에서 BART/mBART 파인튜닝은 여전히 비용 대비 성능이 뛰어나다. 수백만 건의 리뷰 요약이나 자막 정규화를 GPT급 API로 돌리면 단가가 무너지지만, 도메인 데이터 수만 건으로 파인튜닝한 BART-large는 자체 GPU에서 훨씬 저렴하게 처리한다. AI 더빙·자막 파이프라인에서는 ASR 출력의 구두점 복원과 문장 분할(ITN/punctuation restoration)에 seq2seq 디노이징 구조가 그대로 쓰인다.

---

## Paper 3 (Recent): Should We Still Pretrain Encoders with Masked Language Modeling?
- **Authors:** Hippolyte Gisserot-Boukhlef, Nicolas Boizard, Manuel Faysse, Duarte M. Alves, Emmanuel Malherbe, André F. T. Martins, Céline Hudelot, Pierre Colombo
- **Year:** 2025 (v4 revised 2026-05)
- **arXiv:** https://arxiv.org/abs/2507.00994
- **PDF:** [./mlm-vs-clm-encoder-pretraining-gisserot-boukhlef-2025.pdf](./mlm-vs-clm-encoder-pretraining-gisserot-boukhlef-2025.pdf)
- **Citation Count:** 약 40+ (2026년 8월 기준, 빠르게 증가 중)

### 요약
"디코더 LLM을 인코더로 재활용하면 전통적 인코더보다 낫다"는 최근 통념이 정말 목표 함수(CLM) 자체의 우위인지, 아니면 모델·데이터 규모라는 교란 변수 때문인지를 검증한 대규모 통제 실험이다. 2.1억~10억 파라미터 모델 38개를 사전학습하고 1.5만 회 이상의 파인튜닝·평가를 돌린 결과, MLM이 텍스트 표현 태스크에서 전반적으로 더 좋지만 CLM이 데이터 효율과 파인튜닝 안정성에서 앞섰다. 결론은 절충안이다 — **CLM으로 먼저, 그다음 MLM으로 이어 학습하는 2단계(biphasic) 전략**이 고정된 컴퓨트 예산에서 최적이었다.

### 핵심 기여
- 규모와 데이터를 통제한 상태에서 MLM vs CLM을 직접 비교한 최초의 대규모 ablation (38개 모델, 15,000+ 평가 run)
- "CLM이 우월하다"는 관찰이 상당 부분 규모 효과의 착시였음을 분리해서 입증
- Biphasic(CLM → MLM) 학습 레시피 제시 및 동일 컴퓨트 하 최적성 실험적 확인
- 기성 CLM 체크포인트에서 시작해 MLM으로 이어 학습하면 최고급 인코더를 훨씬 적은 비용으로 얻을 수 있음을 보임
- 전체 아티팩트 공개 (https://hf.co/MLMvsCLM)

### 이 논문이 중요한 이유
2024~2025년은 ModernBERT, NeoBERT, EuroBERT 등 "인코더의 귀환"이 이어진 시기였다. 이 논문은 그 흐름에 **어떤 목표 함수로 학습해야 하는가**라는 실무적 답을 준다. 특히 임베딩·분류·리랭킹 모델을 자체 구축하려는 팀에게는, 스크래치 MLM 사전학습 대신 공개된 CLM 모델에서 출발하는 것이 합리적이라는 근거를 제공한다 — 즉 GPU 예산 의사결정을 직접 바꾸는 논문이다.

### 사전 지식
- MLM과 CLM의 목표 함수 차이, 그리고 양방향 어텐션 vs 인과 마스킹
- LLM2Vec류의 "디코더를 인코더로 변환" 기법
- MTEB 등 텍스트 표현 벤치마크의 구성
- 스케일링 법칙과 컴퓨트 최적(compute-optimal) 학습의 개념

### 관련 논문
- [Smarter, Better, Faster, Longer / ModernBERT (Warner et al., 2024)](https://arxiv.org/abs/2412.13663)
- [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders (BehnamGhader et al., 2024)](https://arxiv.org/abs/2404.05961)
- [Seq vs Seq: An Open Suite of Paired Encoders and Decoders / Ettin (Weller et al., 2025)](https://arxiv.org/abs/2507.11412)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)

### 실무 적용
사내 검색·RAG용 임베딩 모델이나 도메인 분류기를 직접 학습할 때 바로 적용된다. 예를 들어 한국어 도메인 인코더가 필요하다면, 처음부터 MLM으로 태우는 대신 이미 공개된 한국어 CLM 모델(예: Qwen/Llama 계열 소형 체크포인트)을 가져와 MLM으로 continued pretraining하는 경로가 비용 대비 성능이 좋다는 뜻이다. 또한 "파인튜닝 안정성"이라는 축은 실무에서 과소평가되는 지표인데, 시드에 따라 성능이 요동치면 A/B 테스트 결과 해석 자체가 오염되므로 모델 선택 기준에 반드시 넣어야 한다.

---

## 추천 읽기 순서

1. **BART (Paper 2)** — 먼저 읽는다. "사전학습 = 노이즈 복원"이라는 큰 프레임을 잡아야 나머지 두 편이 같은 좌표계 안에서 읽힌다. 인코더 온리/디코더 온리/인코더-디코더의 지형도가 여기서 정리된다.
2. **SpanBERT (Paper 1)** — 그 프레임 안에서 "노이즈를 어떻게 설계하느냐"가 성능을 얼마나 바꾸는지 구체적으로 확인한다. 특히 SBO의 ablation 표를 꼼꼼히 볼 것.
3. **MLM vs CLM (Paper 3)** — 마지막으로 2025~2026년 시점의 답안지를 본다. 앞선 두 편이 제기한 "목표 함수 설계" 질문에 대규모 실증으로 답하는 구조이므로, 순서를 지키면 논문의 기여가 훨씬 선명해진다.

## 핵심 테이크어웨이

- **목표 함수는 하이퍼파라미터다.** 아키텍처와 데이터를 고정한 채 마스킹 방식만 바꿔도(SpanBERT) 유의미한 향상이 나온다. 모델을 키우기 전에 학습 과제를 다시 설계할 여지가 있는지 먼저 점검하라.
- **어려운 과제가 좋은 표현을 만든다.** BART의 text infilling과 SpanBERT의 SBO는 모두 "지역적 단서로는 못 풀게" 만들어 모델을 더 멀리 보게 강제한다. 이는 커리큘럼 설계나 합성 데이터 생성에도 그대로 적용되는 원리다.
- **인코더는 죽지 않았다.** 디코더 LLM이 모든 것을 흡수한 듯 보이지만, 분류·검색·구간 추출처럼 지연과 단가가 지배하는 지점에서는 인코더가 여전히 정답이다. 문제는 "인코더냐 디코더냐"가 아니라 "어느 계층에서 무엇을 담당하느냐"다.
- **통념은 통제 실험으로 검증해야 한다.** Paper 3은 "CLM이 낫다"는 업계 상식이 규모 효과의 착시였음을 보여준다. 벤치마크 수치를 볼 때 교란 변수(모델 크기, 데이터량, 파인튜닝 예산)가 통제됐는지 습관적으로 물어야 한다.
- **재활용이 스크래치보다 싸다.** 공개 CLM 체크포인트에서 MLM으로 이어 학습하는 전략은, 제한된 GPU 예산으로 도메인 특화 인코더를 확보하는 가장 현실적인 경로다.

## 다음 토픽과의 연결

다음 토픽은 **Speech Recognition Fundamentals(CTC, Listen-Attend-Spell)**다. 오늘 다룬 세 편은 모두 "입력을 어떻게 손상시키고 복원하게 만들 것인가", 그리고 "인코더와 디코더에 각각 어떤 역할을 줄 것인가"라는 질문을 다뤘다. 음성 인식은 바로 이 질문이 시간축 위에서 반복되는 영역이다 — CTC는 디코더 없이 인코더 출력만으로 정렬 문제를 푸는 접근이고, Listen-Attend-Spell은 BART와 동일한 인코더-디코더 + attention 구조를 음향 프레임 → 문자열 변환에 적용한 것이다. 또한 wav2vec 2.0의 마스킹 기반 사전학습은 오늘 읽은 MLM 계열 사고를 연속 신호에 옮긴 결과물이다. 즉 텍스트에서 익힌 "마스킹과 복원" 직관을 그대로 들고 음성으로 넘어가면 된다.
