# Daily AI Paper Recommendations

> **Date:** 2026-08-19
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN, LSTM and Sequence Models

---

## Paper 1 (Classic): Efficiently Modeling Long Sequences with Structured State Spaces (S4)
- **Authors:** Albert Gu, Karan Goel, Christopher Ré
- **Year:** 2021 (ICLR 2022)
- **arXiv:** https://arxiv.org/abs/2111.00396
- **PDF:** [./structured-state-spaces-s4-gu-2021.pdf](./structured-state-spaces-s4-gu-2021.pdf)
- **Citation Count:** 약 3,500+ (2026년 기준)

### 요약
RNN은 긴 시퀀스에서 기울기 소실로, Transformer는 O(N²) 연산량으로 각각 막혀 있었다. S4는 제어이론의 상태공간모델(SSM)을 딥러닝 레이어로 가져오되, 상태 전이 행렬 A를 HiPPO 기반의 "대각 + 저계수(low-rank)" 구조로 재파라미터화해 코시(Cauchy) 커널 계산으로 환원시켰다. 그 결과 길이 16k의 Path-X를 최초로 풀어내며 Long Range Arena 전 과제 SOTA를 달성했고, 학습은 컨볼루션으로 병렬화, 추론은 재귀로 O(1) 메모리 처리가 가능해졌다.

### 핵심 기여
- SSM의 학습 병목이던 A 행렬 거듭제곱 계산을 NPLR(Normal Plus Low-Rank) 구조로 안정적으로 대각화하여 실용 속도를 확보
- "학습은 CNN처럼 병렬, 추론은 RNN처럼 순차"라는 이중 표현(dual representation) 확립 — 이후 모든 SSM 계열의 기본 문법
- Long Range Arena 전 과제 SOTA, 길이 16,000 Path-X 최초 해결로 장기 의존성 문제에 실증적 돌파구 제시
- 이미지·오디오·텍스트를 동일한 레이어로 처리하는 범용 시퀀스 백본 가능성 입증 (raw audio 분류에서 CNN 대비 우위)

### 이 논문이 중요한 이유
2024~2026년 LLM 아키텍처 논의의 절반은 "Attention을 무엇으로 대체/보완할 것인가"이고, 그 논의의 출발점이 S4다. Mamba, RWKV, Griffin, Jamba, Qwen3-Next의 하이브리드 설계까지 모두 S4가 정립한 SSM 문법 위에 서 있다. AI 엔지니어가 "왜 선형 어텐션 계열 모델이 KV 캐시 없이 긴 컨텍스트를 처리할 수 있는가"를 원리 수준에서 이해하려면 이 논문의 재귀-컨볼루션 이중성을 반드시 거쳐야 한다.

### 사전 지식
선형 시불변 시스템(LTI)과 상태공간 표현 x' = Ax + Bu, y = Cx, 이산화(ZOH/bilinear), 컨볼루션 정리와 FFT, 그리고 RNN의 기울기 소실 문제. HiPPO(arXiv:2008.07669)의 직교 다항식 기반 메모리 압축 개념을 먼저 훑으면 A 행렬 설계 의도가 훨씬 명확해진다.

### 관련 논문
- [HiPPO: Recurrent Memory with Optimal Polynomial Projections (Gu et al., 2020)](https://arxiv.org/abs/2008.07669)
- [Long Range Arena: A Benchmark for Efficient Transformers (Tay et al., 2020)](https://arxiv.org/abs/2011.04006)
- [On the Parameterization and Initialization of Diagonal State Space Models / S4D (Gu et al., 2022)](https://arxiv.org/abs/2206.11893)

### 실무 적용
장시간 센서 스트림, 원시 오디오, ECG 같은 초장기 시퀀스 파이프라인에서 Transformer 대비 메모리를 극적으로 줄이는 백본으로 쓰인다. 음성 파이프라인(STT 프론트엔드), 시계열 이상탐지, 그리고 온디바이스 추론처럼 KV 캐시를 감당할 수 없는 환경에서 특히 유효하다. 실무적으로는 S4 자체보다 이를 단순화한 S4D/Mamba 구현체를 쓰지만, 하이퍼파라미터(상태 차원 N, 이산화 스텝 Δ) 튜닝 감각은 이 논문에서 나온다.

---

## Paper 2 (Classic): Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Albert Gu, Tri Dao
- **Year:** 2023 (COLM 2024)
- **arXiv:** https://arxiv.org/abs/2312.00752
- **PDF:** [./mamba-selective-state-spaces-gu-2023.pdf](./mamba-selective-state-spaces-gu-2023.pdf)
- **Citation Count:** 약 6,000+ (2026년 기준)

### 요약
S4를 포함한 기존 SSM의 치명적 약점은 파라미터가 입력과 무관한 시불변(LTI) 구조라서 "내용 기반 추론", 즉 무엇을 기억하고 무엇을 버릴지를 토큰마다 판단하지 못한다는 점이었다. Mamba는 SSM 파라미터(Δ, B, C)를 입력의 함수로 만드는 선택 메커니즘(selection)을 도입해 이를 해결했고, 대신 잃어버린 컨볼루션 병렬성을 하드웨어 인지 병렬 스캔(SRAM 상에서 상태를 유지하는 커널)으로 되찾았다. 어텐션도 MLP 블록도 없는 단일 블록 구조로 Transformer++ 대비 5배 빠른 추론과 동급 이상의 언어모델 성능을 냈다.

### 핵심 기여
- Selective SSM: 입력 의존적 Δ/B/C로 시변 시스템을 구성해 SSM에 "선택적 망각·기억" 능력 부여 (Induction Head, Selective Copying 과제 해결)
- Hardware-aware parallel scan: 커널 퓨전과 재계산으로 확장된 상태를 HBM에 쓰지 않고 처리, FlashAttention식 IO 최적화를 SSM에 이식
- H3/게이트 MLP를 하나로 합친 단순화된 Mamba 블록 — 아키텍처 설계의 표준 단위 제시
- 언어·오디오·유전체에서 동일 크기 Transformer를 상회, 100만 토큰 길이까지 성능이 개선되는 스케일링 특성 실증

### 이 논문이 중요한 이유
"Attention is all you need"에 대한 가장 진지한 반론이자, 2024년 이후 등장한 거의 모든 하이브리드 LLM(Jamba, Zamba, Samba, Qwen3-Next, Falcon-H1)의 직접적 조상이다. 선형 시간·상수 메모리 추론이라는 특성은 에이전트가 긴 대화 히스토리와 툴 호출 로그를 계속 끌고 가야 하는 Agentic AI 제품의 비용 구조를 바꾼다. AI 엔지니어에게는 "왜 아직도 Transformer가 지배적인가, 그리고 어디서 무너지는가"를 판단하는 기준선이 되는 논문이다.

### 사전 지식
S4의 SSM 이산화와 재귀-컨볼루션 이중성이 선행 필수다. 추가로 병렬 스캔(prefix sum) 알고리즘, GPU 메모리 계층(HBM vs SRAM)과 커널 퓨전 개념, FlashAttention의 IO-aware 최적화 아이디어, 그리고 게이트 메커니즘(LSTM/GRU의 forget gate)을 알고 있으면 selection이 사실상 "학습 가능한 forget gate의 연속시간 일반화"임이 바로 보인다.

### 관련 논문
- [Hungry Hungry Hippos: Towards Language Modeling with State Space Models / H3 (Fu et al., 2022)](https://arxiv.org/abs/2212.14052)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality / Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)

### 실무 적용
긴 컨텍스트를 상수 메모리로 처리해야 하는 실시간 스트리밍 서비스(실시간 자막, 스트리밍 STT, 음성 에이전트)에서 KV 캐시 폭증 없이 서빙 비용을 낮춘다. 실무에서는 순수 Mamba보다 Attention 레이어를 일부 섞은 하이브리드(예: 6~7 : 1 비율)가 검색(retrieval) 성능 손실을 막으면서 처리량을 올리는 정석이며, 이 트레이드오프 판단 자체가 제품 아키텍처 의사결정 포인트다.

---

## Paper 3 (Recent): Gated Delta Networks: Improving Mamba2 with Delta Rule
- **Authors:** Songlin Yang, Jan Kautz, Ali Hatamizadeh (NVIDIA / MIT)
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2412.06464
- **PDF:** [./gated-delta-networks-yang-2024.pdf](./gated-delta-networks-yang-2024.pdf)
- **Citation Count:** 약 250+ (2026년 기준)

### 요약
선형 어텐션/SSM 계열의 남은 약점은 in-context retrieval, 즉 "긴 문맥에서 특정 사실을 정확히 끄집어내기"였다. 이 논문은 두 메모리 조작 방식이 상호보완적임을 관찰한다 — 게이팅(Mamba2)은 상태 전체를 빠르게 지우는 데 강하고, 델타 규칙(DeltaNet)은 특정 키-값 쌍만 정밀 갱신하는 데 강하다. 저자들은 둘을 결합한 gated delta rule과 이를 청크 단위로 병렬 학습하는 하드웨어 효율 알고리즘을 제안했고, Gated DeltaNet은 언어모델링·상식추론·검색·길이 외삽·장문맥 이해 전반에서 Mamba2와 DeltaNet을 모두 능가했다.

### 핵심 기여
- Gated delta rule 제안: 적응적 망각(α_t)과 정밀 연상기억 갱신(delta rule)을 하나의 상태 갱신식으로 통합
- WY 표현 기반 청크 병렬 알고리즘으로 델타 규칙의 순차 병목을 해소, Mamba2 수준의 학습 처리량 확보
- Gated DeltaNet-H1/H2 하이브리드(슬라이딩 윈도우 어텐션 + Mamba2 혼합) 설계로 순수 선형 모델의 검색 한계를 실전 수준까지 끌어올림
- S-NIAH 등 장문맥 검색 벤치마크에서 동급 선형 모델 대비 뚜렷한 우위 실증

### 이 논문이 중요한 이유
2025년 이후 상용 하이브리드 LLM(Qwen3-Next 계열 등)이 실제로 채택한 레이어가 gated delta 계열이다. 즉 이 논문은 학술적 제안에 그치지 않고 프로덕션 아키텍처로 넘어간 드문 사례다. 또한 "선형 어텐션 = 온라인 학습되는 연상기억(associative memory)"이라는 관점을 명확히 정리해, 메모리를 다루는 에이전트 아키텍처를 설계할 때 참고할 수 있는 이론적 프레임을 준다.

### 사전 지식
Mamba-2의 SSD(State Space Duality)와 선형 어텐션의 등가성, DeltaNet(arXiv:2406.06484)의 델타 규칙과 WY 표현, 그리고 Widrow-Hoff/온라인 경사하강 관점의 연상기억. 청크 단위 병렬화(chunkwise parallel form)와 Triton 커널 수준의 구현 감각이 있으면 알고리즘 절이 읽힌다.

### 관련 논문
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length / DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2406.06484)
- [Transformers are SSMs / Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [Titans: Learning to Memorize at Test Time (Behrouz et al., 2025)](https://arxiv.org/abs/2501.00663)

### 실무 적용
장문맥 RAG나 에이전트 메모리처럼 "긴 히스토리 + 정확한 사실 회수"가 동시에 필요한 워크로드에서, 순수 SSM이 놓치던 검색 정확도를 보완한다. 서빙 관점에서는 KV 캐시가 상수 크기로 유지되므로 동시 세션 수를 크게 늘릴 수 있어, 멀티턴 음성 에이전트나 대량 동시 사용자 챗봇의 단가 구조에 직접적으로 영향을 준다. 모델 선택 시 "이 하이브리드가 우리 검색 과제에서 실제로 어텐션 대비 몇 % 손실인지"를 S-NIAH류 자체 벤치로 먼저 재보는 것이 실무 순서다.

---

## 추천 읽기 순서
1. **S4 (2021)** — SSM의 기본 문법(이산화, 재귀-컨볼루션 이중성)을 먼저 잡는다. 수식이 가장 무겁지만 여기서 3장까지만 정확히 이해해도 나머지가 쉬워진다.
2. **Mamba (2023)** — S4의 한계(입력 무관 LTI)를 어떻게 깨는지 본다. 선택 메커니즘의 동기(Selective Copying, Induction Head 실험)를 먼저 읽고 알고리즘으로 넘어가면 이해가 빠르다.
3. **Gated DeltaNet (2024)** — 게이팅과 델타 규칙의 결합, 그리고 하이브리드 설계가 실제 제품에 어떻게 안착했는지 확인한다.

시간이 부족하면 Mamba → Gated DeltaNet 순으로 읽고 S4는 3장(방법론)만 참조해도 흐름은 잡힌다.

## 핵심 테이크어웨이
- **시퀀스 모델의 본질은 "고정 크기 상태에 무엇을 압축할 것인가"의 문제다.** RNN의 은닉 상태, SSM의 상태 행렬, Transformer의 KV 캐시는 모두 같은 질문에 대한 다른 답이며, 효율성과 회수 정확도는 정확히 이 지점에서 트레이드오프된다.
- **Transformer는 압축을 포기해서(전체 KV 보관) 정확도를 샀고, SSM은 압축을 택해 비용을 샀다.** 2024~2026년의 하이브리드 아키텍처는 이 둘을 레이어 단위로 섞는 실용적 타협이다.
- **선택(selection)과 망각(gating)은 결국 LSTM forget gate의 재발견이다.** 30년 된 아이디어가 하드웨어 인지 알고리즘을 만나 다시 최전선으로 돌아왔다는 점이, 아키텍처 연구에서 "이론적 우수성보다 하드웨어 적합성이 승패를 가른다"는 교훈을 보여준다.
- **선형 모델의 진짜 약점은 성능이 아니라 in-context retrieval이다.** 도입을 검토한다면 일반 벤치마크 점수가 아니라 자사 과제의 needle-in-a-haystack류 회수 정확도로 판단해야 한다.

## 다음 토픽과의 연결
다음 토픽인 **최적화와 정규화(Optimization and Regularization)**는 오늘 본 아키텍처들이 "실제로 학습되게 만드는" 장치를 다룬다. S4의 Δ 파라미터에 별도 학습률과 weight decay 제외 규칙을 적용해야 수렴하는 것, Mamba의 안정적 학습을 위한 초기화 전략 등은 모두 최적화 설계 문제다. 또한 오늘 반복해서 등장한 "기울기 소실/폭발"과 "상태 안정성"은 다음 모듈의 정규화 논의로 자연스럽게 이어진다.
