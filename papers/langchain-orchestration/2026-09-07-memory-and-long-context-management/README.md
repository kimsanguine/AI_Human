# Daily AI Paper Recommendations

> **Date:** 2026-09-07
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** Memory and Long-Context Management

---

## Paper 1 (Classic): Neural Turing Machines
- **Authors:** Alex Graves, Greg Wayne, Ivo Danihelka
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1410.5401
- **PDF:** [./neural-turing-machines-graves-2014.pdf](./neural-turing-machines-graves-2014.pdf)
- **Citation Count:** ~3,000회 (approximate)

### 요약
신경망에 외부 메모리 행렬(external memory matrix)을 붙이고, 그 메모리를 미분 가능한 읽기/쓰기 헤드로 접근하게 만든 최초의 본격적인 시도입니다. 컨트롤러(LSTM 또는 MLP)가 "어디를 읽고 어디에 쓸지"를 소프트 어텐션 가중치로 결정하기 때문에, 전체 시스템을 end-to-end 경사하강법으로 학습할 수 있습니다. 복사(copy), 반복 복사, 연상 회상, 정렬 같은 알고리즘적 과제에서 LSTM 단독 대비 훨씬 긴 시퀀스로 일반화되는 것을 보였습니다.

### 핵심 기여
- 연산(컨트롤러)과 저장(메모리)을 분리한 아키텍처를 제안 — 파라미터 수를 늘리지 않고 기억 용량을 늘릴 수 있음
- 내용 기반(content-based) 주소 지정 + 위치 기반(location-based) 주소 지정을 결합한 미분 가능한 메모리 접근 메커니즘 설계
- 신경망이 명시적 프로그래밍 없이 데이터만으로 간단한 알고리즘(복사, 정렬)을 "학습"할 수 있음을 실증

### 이 논문이 중요한 이유
오늘날 우리가 부르는 "에이전트 메모리"의 개념적 원형입니다. LLM 에이전트에서 벡터 스토어를 조회하고 요약을 다시 써 넣는 read/write 루프는, NTM이 제안한 컨트롤러-메모리 분리 구조를 자연어와 도구 호출로 재구현한 것에 가깝습니다. 왜 컨텍스트 윈도우를 키우는 것만으로는 부족하고 별도의 외부 저장소가 필요한지를 원리 수준에서 이해하려면 이 논문이 출발점입니다. 또한 "주소 지정을 어떻게 할 것인가(내용 기반 vs 위치 기반)"라는 질문은 지금의 semantic search vs recency 기반 회상 설계 논쟁과 정확히 같은 문제입니다.

### 사전 지식
- RNN/LSTM의 동작 원리와 장기 의존성 문제
- 소프트 어텐션과 softmax 가중합의 미분 가능성
- 시퀀스-투-시퀀스 학습의 기본 셋업
- (선택) 튜링 머신의 테이프/헤드 개념

### 관련 논문
- [Hybrid computing using a neural network with dynamic external memory / DNC (Graves et al., 2016)](https://www.nature.com/articles/nature20101)
- [Memory Networks (Weston, Chopra & Bordes, 2014)](https://arxiv.org/abs/1410.3916)
- [End-To-End Memory Networks (Sukhbaatar et al., 2015)](https://arxiv.org/abs/1503.08895)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)

### 실무 적용
LangGraph나 LlamaIndex로 에이전트를 만들 때 "메모리 상태를 어떤 슬롯 구조로 나눌 것인가"를 설계하는 감각이 여기서 나옵니다. NTM의 메모리 행렬은 실무에서 (a) 벡터 DB의 임베딩 슬롯, (b) 세션 상태 딕셔너리, (c) 스크래치패드 파일로 나뉘어 구현됩니다. 특히 NTM이 보여준 "쓰기에는 erase + add 두 단계가 필요하다"는 통찰은, 에이전트 메모리에서 오래된 사실을 그냥 덮어쓰지 말고 무효화(invalidate) 후 갱신해야 한다는 실무 규칙과 직결됩니다.

---

## Paper 2 (Classic): Compressive Transformers for Long-Range Sequence Modelling
- **Authors:** Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap
- **Year:** 2019 (ICLR 2020)
- **arXiv:** https://arxiv.org/abs/1911.05507
- **PDF:** [./compressive-transformer-rae-2019.pdf](./compressive-transformer-rae-2019.pdf)
- **Citation Count:** ~1,000회 (approximate)

### 요약
Transformer-XL이 오래된 활성값(activation)을 그냥 버리는 대신, 이를 압축해 2차 메모리(compressed memory)에 보관하자는 제안입니다. 짧은 기간의 세밀한 메모리와 긴 기간의 거칠지만 오래 유지되는 메모리를 계층으로 두어, 메모리 비용을 선형 이하로 유지하면서 컨텍스트 범위를 크게 확장합니다. WikiText-103과 Enwik8에서 SOTA를 기록했고, 책 단위 장기 문맥 평가를 위한 PG-19 벤치마크를 함께 공개했습니다.

### 핵심 기여
- 다단계 메모리 계층(활성 → 단기 메모리 → 압축 메모리)이라는 설계 패턴 제안
- 압축 함수(max/mean pooling, 1D conv, dilated conv)와 압축 손실(attention-reconstruction loss)을 비교 실험으로 정리
- PG-19 데이터셋 공개 — 문단이 아니라 책 전체 수준의 장기 문맥을 평가하는 표준 벤치마크 확립
- 희소한(rare) 토큰일수록 압축 메모리의 이득이 크다는 분석 제시

### 이 논문이 중요한 이유
"오래된 정보를 버릴 것인가, 압축해서 남길 것인가"는 LLM 에이전트 메모리 설계의 핵심 트레이드오프이며, 이 논문이 그 문제를 가장 깔끔하게 정식화했습니다. 실무에서 대화가 길어질 때 하는 rolling summarization은 사실상 Compressive Transformer의 아이디어를 토큰 공간에서 다시 구현한 것입니다. 또한 "무엇을 압축 손실로 삼을 것인가(재구성 vs 어텐션 재현)"라는 질문은, 요약 프롬프트가 무엇을 보존해야 하는지 결정할 때 그대로 적용됩니다.

### 사전 지식
- Transformer의 self-attention과 KV 캐시 구조
- Transformer-XL의 세그먼트 순환(segment recurrence)과 상대 위치 인코딩
- 언어 모델 평가 지표(perplexity, bits-per-character)
- 메모리 복잡도 O(n²) 문제와 이를 완화하는 접근들

### 관련 논문
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](https://arxiv.org/abs/1901.02860)
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention (Munkhdalai et al., 2024)](https://arxiv.org/abs/2404.07143)
- [Titans: Learning to Memorize at Test Time (Behrouz et al., 2025)](https://arxiv.org/abs/2501.00663)

### 실무 적용
대화형 에이전트의 컨텍스트 관리 전략을 세울 때 바로 쓰입니다. 최근 N턴은 원문 그대로(단기 메모리), 그 이전은 턴 단위 요약(1차 압축), 더 이전은 세션 요약(2차 압축)으로 두는 3단 구조가 이 논문의 계층 메모리를 그대로 옮긴 것입니다. LangChain의 `ConversationSummaryBufferMemory`가 사실상 2단 버전이며, 압축 비율(compression rate)을 얼마로 둘지, 어떤 정보를 손실로 허용할지 결정할 때 이 논문의 ablation이 좋은 기준선이 됩니다.

---

## Paper 3 (Recent): A-MEM: Agentic Memory for LLM Agents
- **Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.12110
- **PDF:** [./a-mem-agentic-memory-xu-2025.pdf](./a-mem-agentic-memory-xu-2025.pdf)
- **Citation Count:** ~150회 (approximate, Semantic Scholar)

### 요약
기존 에이전트 메모리 시스템이 사람이 미리 정한 고정 스키마(예: "사용자 선호도", "과거 행동")에 의존한다는 문제를 지적하고, 메모리 구조 자체를 에이전트가 동적으로 만들어가는 A-MEM을 제안합니다. 제텔카스텐(Zettelkasten) 방법론에서 착안해, 새 메모리가 들어올 때마다 컨텍스트 설명·키워드·태그를 담은 구조화 노트를 생성하고, 의미적으로 가까운 기존 노트와 링크를 자동 생성합니다. 여기서 그치지 않고 새 노트와 연결된 기존 노트들의 표현까지 함께 갱신하는 memory evolution 단계를 둔 것이 특징입니다.

### 핵심 기여
- 사전 정의된 메모리 스키마 없이 에이전트가 메모리 구조를 스스로 조직하는 agentic memory 프레임워크 제안
- 노트 생성 → 링크 생성 → 메모리 진화의 3단계 파이프라인 설계 (새 지식이 기존 지식의 해석을 바꾸는 구조)
- LoCoMo 등 장기 대화 벤치마크의 멀티홉 추론 과제에서 기존 메모리 baseline 대비 큰 폭의 성능 향상 입증
- 여러 백본 LLM에 걸친 일반성 검증과 오픈소스 구현 공개

### 이 논문이 중요한 이유
2025년 이후 에이전트 메모리 논의의 무게중심이 "무엇을 저장할까(storage)"에서 "저장된 것들을 어떻게 연결하고 갱신할까(organization)"로 옮겨갔는데, 그 전환점에 있는 논문입니다. 실무에서 RAG 기반 메모리를 붙였는데도 에이전트가 여러 대화에 흩어진 사실을 연결하지 못하는 문제, 그리고 과거 정보가 최신 정보와 모순될 때 잘못된 답을 하는 문제 — 이 두 가지가 바로 A-MEM이 겨냥한 실패 모드입니다. AI 엔지니어에게는 "벡터 검색만으로는 메모리가 되지 않는다"는 점을 실험적으로 확인시켜 주는 레퍼런스입니다.

### 사전 지식
- RAG의 기본 구조(임베딩, 벡터 검색, top-k 회상)
- LLM 에이전트 루프와 도구 호출
- 지식 그래프의 노드/엣지 개념과 멀티홉 추론
- LoCoMo 같은 장기 대화 벤치마크의 평가 방식
- MemGPT, Mem0 등 선행 메모리 시스템의 개념

### 관련 논문
- [MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)](https://arxiv.org/abs/2310.08560)
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (Chhikara et al., 2025)](https://arxiv.org/abs/2504.19413)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory (Rasmussen et al., 2025)](https://arxiv.org/abs/2501.13956)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs (Gutiérrez et al., 2024)](https://arxiv.org/abs/2405.14831)

### 실무 적용
장기 사용자 관계를 다루는 제품(AI 코치, CS 에이전트, 개인 비서)에서 즉시 적용 가능합니다. 구현 관점의 체크리스트는 세 가지입니다. (1) 메모리를 raw 텍스트 청크가 아니라 키워드·태그·요약을 포함한 구조화 노트로 저장할 것, (2) 저장 시점에 기존 노트와의 링크를 만들어 검색이 그래프 순회로 확장되게 할 것, (3) 새 정보가 들어올 때 관련 노트를 갱신하는 배치 잡을 둘 것. 다만 노트 생성과 진화 단계마다 LLM 호출이 발생하므로 쓰기 비용이 단순 임베딩 방식보다 크게 늘어납니다. 실무에서는 모든 발화가 아니라 "사실성 있는 발화"만 필터링해 메모리로 승격시키는 게이트를 앞단에 두는 설계가 현실적입니다.

---

## 추천 읽기 순서

1. **Neural Turing Machines (2014)** — 먼저 읽으세요. 외부 메모리라는 개념 자체와 읽기/쓰기를 어떻게 미분 가능하게 만드는지를 이해하면, 이후 모든 메모리 논문이 "이 문제의 변주"로 보입니다.
2. **Compressive Transformers (2019)** — 다음으로 읽으세요. 메모리를 무한히 키울 수 없다는 현실적 제약 아래에서 "압축과 계층화"라는 해법을 배웁니다. 오늘날의 요약 기반 컨텍스트 관리의 이론적 근거입니다.
3. **A-MEM (2025)** — 마지막에 읽으세요. 앞의 두 논문이 파라미터 공간에서 다룬 문제를 LLM 시대에는 자연어와 그래프 구조로 어떻게 다시 푸는지 확인하게 됩니다.

## 핵심 테이크어웨이

- **연산과 저장은 분리되어야 한다.** NTM 이후 30여 년간 반복되는 결론입니다. 컨텍스트 윈도우를 키우는 것은 저장 용량을 늘리는 것이지 메모리 시스템을 만드는 것이 아닙니다.
- **메모리 설계의 본질은 "버리기"의 설계다.** Compressive Transformer가 보여주듯, 무엇을 원문으로 남기고 무엇을 압축하고 무엇을 폐기할지에 대한 정책이 곧 메모리 아키텍처입니다.
- **저장보다 조직화가 어렵다.** A-MEM의 메시지는 명확합니다. top-k 벡터 검색은 회상(recall)은 되지만 연결(association)은 안 됩니다. 멀티홉 질문이 필요한 제품이라면 링크 구조를 반드시 설계해야 합니다.
- **메모리는 갱신되어야 한다.** 정적으로 쌓이기만 하는 메모리는 시간이 지나면 모순 덩어리가 됩니다. 무효화와 진화 메커니즘 없이는 장기 운영이 불가능합니다.
- **쓰기 비용은 읽기 비용보다 과소평가된다.** 구조화 노트 생성, 링크 생성, 진화는 모두 LLM 호출입니다. 프로덕션에서는 메모리 승격 게이트와 비동기 배치가 필수입니다.

## 다음 토픽과의 연결

다음 모듈은 **RAG(Retrieval-Augmented Generation)**로 넘어가며, 첫 토픽은 Dense Retrieval과 임베딩 검색입니다. 오늘 다룬 메모리는 "에이전트 내부에 축적되는 경험"이었다면, RAG는 "외부에 이미 존재하는 지식"을 다룹니다. 두 문제는 회상 메커니즘(임베딩 검색, 랭킹, 컨텍스트 주입)을 공유하지만, 쓰기 주체와 신뢰도 관리 방식이 다릅니다. 오늘 A-MEM에서 본 링크 기반 조직화가 다음 모듈의 GraphRAG·HippoRAG 계열과 어떻게 수렴하는지를 염두에 두고 읽으면 두 흐름이 하나로 이어집니다.
