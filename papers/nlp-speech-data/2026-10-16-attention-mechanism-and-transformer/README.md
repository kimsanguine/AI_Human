# Daily AI Paper Recommendations

> **Date:** 2026-10-16
> **Module:** Module 4: NLP and Speech Data
> **Topic:** Attention Mechanism and Transformer

---

## Paper 1 (Classic): Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)
- **Authors:** Ofir Press, Noah A. Smith, Mike Lewis
- **Year:** 2021 (ICLR 2022)
- **arXiv:** https://arxiv.org/abs/2108.12409
- **PDF:** [./alibi-press-2021.pdf](./alibi-press-2021.pdf)
- **Citation Count:** ~900+

### 요약
위치 임베딩을 입력에 더하는 대신, 어텐션 점수에 "쿼리-키 거리에 비례하는 선형 페널티"를 헤드별 기울기로 더하는 단순한 방법(ALiBi)을 제안한다. 짧은 길이로 학습해도 더 긴 시퀀스에서 성능이 무너지지 않는 길이 외삽(extrapolation)을 보여준다.

### 핵심 기여
- 학습 파라미터 없이 어텐션 logit에 `-m·|i-j|` 편향만 추가하는 위치 표현 방식 제안
- 1024 토큰으로 학습한 모델이 2048 토큰 추론에서 sinusoidal 위치 임베딩(2048 학습)과 동등한 perplexity 달성
- 학습 속도·메모리 측면에서 sinusoidal/rotary 대비 오버헤드가 거의 없음을 실험으로 확인

### 이 논문이 중요한 이유
"컨텍스트 길이를 학습 이후에 늘릴 수 있는가?"라는 질문을 처음으로 명확하게 정식화한 논문이다. 이후 RoPE scaling, Position Interpolation, YaRN 등 롱컨텍스트 기법의 비교 기준점이 되었고, BLOOM·MPT 같은 오픈 LLM이 실제로 채택했다.

### 사전 지식
- Transformer의 Scaled Dot-Product Attention 수식
- Sinusoidal / Learned / Rotary(RoPE) 위치 인코딩의 차이
- Perplexity 지표의 의미

### 관련 논문
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of LLMs via Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of LLMs (Peng et al., 2023)](https://arxiv.org/abs/2309.00071)

### 실무 적용
긴 문서 요약, 장시간 음성 전사(STT) 후처리, 대화 이력이 긴 에이전트처럼 "학습 때보다 긴 입력"이 들어오는 서비스에서 위치 인코딩 선택이 품질을 좌우한다. 자체 모델을 학습·파인튜닝할 때 ALiBi vs RoPE+스케일링 중 무엇을 쓸지 판단하는 근거가 된다.

---

## Paper 2 (Classic): Linformer: Self-Attention with Linear Complexity
- **Authors:** Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2006.04768
- **PDF:** [./linformer-wang-2020.pdf](./linformer-wang-2020.pdf)
- **Citation Count:** ~2,000+

### 요약
셀프 어텐션 행렬이 실제로는 저랭크(low-rank)라는 점을 이론·실험으로 보이고, 키·값을 길이 방향으로 n→k 차원 투영해 어텐션 비용을 O(n²)에서 O(n)으로 줄인다. 사전학습·다운스트림 성능은 표준 Transformer와 비슷하게 유지된다.

### 핵심 기여
- 어텐션 컨텍스트 행렬의 스펙트럼 분석으로 "저랭크 근사 가능성"을 제시 (Johnson–Lindenstrauss 기반 증명)
- K, V에 학습 가능한 선형 투영(E, F)을 적용하는 간단한 구조로 선형 시간·메모리 달성
- 헤드/레이어 간 투영 공유 등 파라미터 절감 변형과 긴 시퀀스에서의 속도 향상 실측

### 이 논문이 중요한 이유
효율적 어텐션 연구의 3대 축(희소화 · 커널/선형화 · 저랭크) 중 "저랭크" 계열의 대표 논문이다. 2024년 이후 등장한 DeepSeek의 MLA(Paper 3)도 "KV를 저차원 잠재공간으로 압축한다"는 같은 직관을 공유하므로, 둘을 함께 읽으면 아이디어가 어떻게 진화했는지 볼 수 있다.

### 사전 지식
- 행렬 랭크, SVD, 저랭크 근사의 기본 개념
- 어텐션의 시간/메모리 복잡도 O(n²) 의 원인
- 효율적 Transformer 계열(Reformer, Longformer, Performer) 개요

### 관련 논문
- [Efficient Transformers: A Survey (Tay et al., 2020)](https://arxiv.org/abs/2009.06732)
- [Rethinking Attention with Performers (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794)
- [Nyströmformer (Xiong et al., 2021)](https://arxiv.org/abs/2102.03902)

### 실무 적용
인과적(autoregressive) 디코딩에는 그대로 쓰기 어렵다는 한계가 있어, 실무에서는 주로 인코더 기반 분류·임베딩·긴 시퀀스 처리(예: 긴 오디오 프레임, 로그 분석)에 활용된다. "어디까지 압축해도 품질이 유지되는가"를 실험 설계할 때 좋은 기준선이 된다.

---

## Paper 3 (Recent): DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (Multi-head Latent Attention)
- **Authors:** DeepSeek-AI (Aixin Liu, Bei Feng, Bin Wang, et al.)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2405.04434
- **PDF:** [./deepseek-v2-mla-deepseek-ai-2024.pdf](./deepseek-v2-mla-deepseek-ai-2024.pdf)
- **Citation Count:** ~1,500+

### 요약
236B 총 파라미터(토큰당 21B 활성)의 MoE LLM으로, 핵심은 키·값을 저랭크 잠재 벡터로 함께 압축해 캐시하는 Multi-head Latent Attention(MLA)이다. 논문은 DeepSeek 67B 대비 KV 캐시 93.3% 절감, 학습 비용 42.5% 절감, 최대 생성 처리량 5.76배 향상을 보고한다(저자 측 수치).

### 핵심 기여
- MLA: KV를 공동 저랭크 압축 → MQA/GQA보다 작은 캐시로 MHA 이상의 성능을 주장
- Decoupled RoPE: 압축 구조와 회전 위치 인코딩을 충돌 없이 결합하는 설계
- DeepSeekMoE(세분화된 전문가 + 공유 전문가)와 결합해 128K 컨텍스트의 경제적 학습·추론 실현

### 이 논문이 중요한 이유
MHA → MQA → GQA로 이어진 "KV 캐시 다이어트" 계보의 다음 단계이며, DeepSeek-V3/R1에도 그대로 이어져 2025년 이후 오픈 LLM 아키텍처 논의의 중심이 되었다. 추론 비용이 곧 제품 단가인 AI 서비스에서 "어텐션 구조가 비용 구조를 바꾼다"는 것을 보여준 사례다.

### 사전 지식
- MHA / MQA / GQA의 차이와 KV 캐시 개념
- RoPE(회전 위치 인코딩)
- Mixture-of-Experts의 라우팅 기본

### 관련 논문
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437)
- [TransMLA: Multi-Head Latent Attention Is All You Need (Meng et al., 2025)](https://arxiv.org/abs/2502.07864)

### 실무 적용
같은 GPU에서 더 많은 동시 사용자·더 긴 컨텍스트를 처리할 수 있어 서빙 단가를 직접 낮춘다. vLLM·SGLang 등 추론 엔진이 MLA 전용 커널을 지원하므로, 모델 선택 시 "파라미터 수"만이 아니라 "토큰당 KV 캐시 크기"를 비교 지표로 넣어야 한다.

---

## 추천 읽기 순서
1. **Linformer** — 어텐션이 왜 O(n²)이고, 저랭크로 줄일 수 있는 이유를 먼저 이해
2. **ALiBi** — 효율 다음의 문제: 학습 길이를 넘는 입력에서 위치 정보를 어떻게 다룰지
3. **DeepSeek-V2 (MLA 섹션 2.1 중심)** — 저랭크 압축 + RoPE 문제를 실제 대형 LLM에서 어떻게 풀었는지 확인

## 핵심 테이크어웨이
- 어텐션 연구의 진짜 질문은 "얼마나 정확한가"보다 "어떤 정보를 얼마나 싸게 보관·재사용하는가"로 이동했다.
- 저랭크 아이디어(Linformer, 2020)는 시퀀스 길이 축 압축에서, KV 캐시의 헤드/차원 축 압축(MLA, 2024)으로 진화했다.
- 위치 인코딩은 롱컨텍스트 성능과 압축 구조 모두에 제약을 주는 숨은 설계 변수다(ALiBi, Decoupled RoPE).
- 탐구 질문: 우리 서비스의 병목은 계산(prefill)인가, 메모리(KV 캐시·decode)인가? 답에 따라 선택할 어텐션 변형이 달라진다.

## 다음 토픽과의 연결
다음 토픽 **BERT and Pre-trained Language Models**에서는 이 어텐션 블록을 쌓아 "어떤 목적함수로 사전학습하느냐"가 핵심이 된다. 오늘 본 위치 인코딩·효율적 어텐션 선택이 ModernBERT 같은 최신 인코더(RoPE, 긴 컨텍스트, 교대 로컬/글로벌 어텐션)에 어떻게 반영되었는지 비교해 보자.
