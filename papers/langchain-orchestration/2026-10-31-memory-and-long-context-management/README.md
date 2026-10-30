# Daily AI Paper Recommendations

> **Date:** 2026-10-31
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** Memory and Long-Context Management

---

## Paper 1 (Classic): Recurrent Memory Transformer
- **Authors:** Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev
- **Year:** 2022 (NeurIPS 2022)
- **arXiv:** https://arxiv.org/abs/2207.06881
- **PDF:** [./recurrent-memory-transformer-bulatov-2022.pdf](./recurrent-memory-transformer-bulatov-2022.pdf)
- **Citation Count:** ~300+

### 요약
긴 입력을 여러 세그먼트로 나누고, 입력 앞뒤에 특수한 "메모리 토큰"을 붙여 세그먼트 사이에 정보를 전달하는 방식입니다. 이전 세그먼트의 출력 메모리 토큰이 다음 세그먼트의 입력 메모리가 되므로, 트랜스포머 구조를 바꾸지 않고도 RNN처럼 정보를 이어갈 수 있습니다. 후속 연구(Scaling Transformer to 1M tokens, 2023)에서는 BERT에 RMT를 적용해 100만 토큰 이상까지 확장했습니다.

### 핵심 기여
- 모델 아키텍처 변경 없이 "메모리 토큰"만 추가해 세그먼트 단위 반복(recurrence) 구현
- 세그먼트 간 역전파(BPTT)로 무엇을 기억할지 모델이 스스로 학습
- Transformer-XL 대비 적은 메모리로 비슷하거나 더 나은 장거리 의존성 성능 (copy, reverse, associative retrieval 과제)

### 이 논문이 중요한 이유
"컨텍스트 윈도우를 늘리는 것"과 "압축된 상태로 기억을 이어가는 것"이라는 두 갈래 중, 후자의 대표적 출발점입니다. 오늘날 에이전트가 대화 요약을 다음 턴에 넘기는 방식(summary memory)을 모델 내부에서 학습 가능한 형태로 구현한 것과 같아서, 에이전트 메모리 설계를 이해하는 개념적 틀이 됩니다.

### 사전 지식
- Transformer 구조와 self-attention 비용(O(n²))
- RNN과 BPTT(Backpropagation Through Time)의 개념
- Transformer-XL의 세그먼트 반복 아이디어

### 관련 논문
- [Transformer-XL (Dai et al., 2019)](https://arxiv.org/abs/1901.02860)
- [Scaling Transformer to 1M tokens and beyond with RMT (Bulatov et al., 2023)](https://arxiv.org/abs/2304.11062)
- [Memorizing Transformers (Wu et al., 2022)](https://arxiv.org/abs/2203.08913)

### 실무 적용
긴 문서를 청크로 나눠 처리하면서 "이전 청크에서 기억할 핵심"을 다음 청크로 넘기는 파이프라인(예: LangGraph의 rolling summary 노드)의 설계 원리와 동일합니다. 직접 RMT를 학습하지 않더라도, 메모리 크기(토큰 수)와 정보 손실 간 트레이드오프를 판단하는 기준으로 활용할 수 있습니다.

---

## Paper 2 (Classic): Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2309.17453
- **PDF:** [./streamingllm-attention-sinks-xiao-2023.pdf](./streamingllm-attention-sinks-xiao-2023.pdf)
- **Citation Count:** ~1,000+

### 요약
LLM은 의미가 거의 없는 맨 앞 몇 개 토큰에 큰 attention 점수를 몰아주는데, 저자들은 이를 "attention sink"라고 부릅니다. 슬라이딩 윈도우로 오래된 KV 캐시를 버리면 이 sink 토큰도 사라져 성능이 무너집니다. StreamingLLM은 처음 4개 토큰의 KV와 최근 윈도우 KV만 유지해, 추가 학습 없이 최대 400만 토큰 스트리밍을 안정적으로 처리합니다.

### 핵심 기여
- Attention sink 현상의 발견과 원인 분석 (softmax가 합이 1이 되어야 하는 특성)
- 학습 없이 적용 가능한 KV 캐시 정책: sink 토큰 + 최근 윈도우
- 사전학습 시 전용 sink 토큰을 넣으면 스트리밍 성능이 더 좋아진다는 제안
- 슬라이딩 윈도우 재계산 대비 최대 22.2배 속도 향상

### 이 논문이 중요한 이유
24시간 돌아가는 챗봇이나 에이전트처럼 "끝없이 길어지는 대화"를 메모리 폭발 없이 처리하는 방법을 보여줍니다. 다만 StreamingLLM은 컨텍스트를 "늘리는" 것이 아니라 "안정적으로 버리는" 방법이라는 점이 중요합니다. 버려진 정보는 기억하지 못하므로, 외부 메모리(RAG, 요약)와 함께 써야 한다는 설계 판단을 내리게 해줍니다.

### 사전 지식
- KV 캐시가 무엇이고 왜 메모리를 많이 쓰는지
- Softmax attention의 작동 방식
- 슬라이딩 윈도우 attention (Longformer 등)

### 관련 논문
- [Longformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference (Zhang et al., 2023)](https://arxiv.org/abs/2306.14048)
- [Vision Transformers Need Registers (Darcet et al., 2023)](https://arxiv.org/abs/2309.16588)

### 실무 적용
vLLM, TensorRT-LLM, HuggingFace Transformers의 `SinkCache` 등 주요 서빙 스택에 반영되어 있습니다. 장시간 음성 비서, 실시간 자막/더빙, 고객상담 봇처럼 세션이 길어지는 서비스에서 GPU 메모리 상한을 고정하는 데 바로 쓸 수 있습니다. OpenAI gpt-oss 등 최신 모델도 attention sink 개념을 아키텍처에 포함하고 있습니다.

---

## Paper 3 (Recent): RULER: What's the Real Context Size of Your Long-Context Language Models?
- **Authors:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg
- **Year:** 2024 (COLM 2024)
- **arXiv:** https://arxiv.org/abs/2404.06654
- **PDF:** [./ruler-real-context-size-hsieh-2024.pdf](./ruler-real-context-size-hsieh-2024.pdf)
- **Citation Count:** ~500+

### 요약
"128K 컨텍스트 지원"이라는 숫자가 실제로 얼마나 쓸모 있는지 측정하는 합성 벤치마크입니다. 단순한 needle-in-a-haystack을 넘어 다중 needle 검색, 변수 추적(multi-hop tracing), 집계(aggregation), 질의응답 등 4개 카테고리 13개 과제를 길이별로 평가합니다. 32K 이상을 지원한다고 주장하는 모델 중 실제로 32K에서 기준 성능을 유지한 것은 절반 정도였습니다.

### 핵심 기여
- "주장된 컨텍스트 길이"와 "유효 컨텍스트 길이(effective context length)"의 구분을 정량화
- 길이와 과제 난이도를 자유롭게 조절할 수 있는 오픈소스 합성 벤치마크 제공
- NIAH 점수가 거의 만점인 모델도 길이와 복잡도가 늘면 성능이 크게 떨어진다는 실증

### 이 논문이 중요한 이유
"컨텍스트가 길면 RAG나 메모리 설계가 필요 없다"는 가정을 데이터로 검증할 수 있게 해줍니다. 에이전트 아키텍처를 설계할 때 "전부 컨텍스트에 넣을지, 외부 메모리로 뺄지"를 결정하는 근거가 되며, 모델 선택 시 마케팅 수치 대신 실제 유효 길이로 비교하는 습관을 길러줍니다.

### 사전 지식
- Needle-in-a-Haystack 평가 방식
- Lost in the Middle 현상 (긴 컨텍스트 중간 정보 손실)
- RoPE 기반 컨텍스트 확장 기법 개요

### 관련 논문
- [Lost in the Middle (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)
- [LongBench (Bai et al., 2023)](https://arxiv.org/abs/2308.14508)
- [∞Bench: Extending Long Context Evaluation Beyond 100K Tokens (Zhang et al., 2024)](https://arxiv.org/abs/2402.13718)

### 실무 적용
사내 모델 선정 시 RULER 코드(github.com/hsiehjackson/RULER)를 돌려 우리 서비스의 실제 입력 길이에서 성능 절벽이 어디인지 확인할 수 있습니다. 그 지점을 기준으로 컨텍스트 예산을 정하고, 넘치는 부분은 요약 메모리나 RAG로 넘기는 정책을 세우면 됩니다.

---

## 추천 읽기 순서
1. **StreamingLLM** — KV 캐시와 attention 동작을 직관적으로 이해하기 가장 쉽고, 실무 적용도 바로 연결됩니다.
2. **Recurrent Memory Transformer** — "버리는 대신 압축해서 넘긴다"는 대안적 접근을 비교하며 읽습니다.
3. **RULER** — 앞의 두 접근을 포함한 긴 컨텍스트 기법들이 실제로 얼마나 효과가 있는지 평가하는 관점으로 마무리합니다.

## 핵심 테이크어웨이
- 긴 컨텍스트 관리에는 세 가지 질문이 있습니다: **무엇을 버릴까(StreamingLLM), 무엇을 압축해 넘길까(RMT), 실제로 얼마나 기억하나(RULER)**.
- 컨텍스트 길이 스펙은 "유효 길이"와 다릅니다. 우리 데이터로 직접 측정해야 합니다.
- 모델 내부 메모리만으로는 부족하며, 에이전트 수준의 외부 메모리(요약, 벡터 검색, 그래프)와 함께 설계해야 합니다.

## 다음 토픽과의 연결
다음 모듈은 **RAG(Module 9)**이며 첫 주제는 **Dense Retrieval and Embedding Search**입니다. 오늘 RULER에서 확인한 "긴 컨텍스트의 한계"는 필요한 정보만 골라 넣는 검색 기반 접근이 왜 필요한지 보여줍니다. StreamingLLM이 버리는 정보를 외부 벡터 저장소에서 다시 찾아오는 구조가 바로 RAG입니다.
