# Daily AI Paper Recommendations

> **Date:** 2026-05-24
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** RAG Architecture and Optimization

---

## Paper 1 (Classic): Improving language models by retrieving from trillions of tokens (RETRO)
- **Authors:** Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. (DeepMind)
- **Year:** 2022 (ICML 2022)
- **arXiv:** https://arxiv.org/abs/2112.04426
- **PDF:** [./retro-borgeaud-2022.pdf](./retro-borgeaud-2022.pdf)
- **Citation Count:** ~1,500+ (DeepMind의 retrieval-augmented LM 대표작)

### 요약
RETRO는 2조 개(2 trillion) 토큰 규모의 외부 데이터베이스에서 가까운 이웃 청크(neighbor chunk)를 검색해 자기회귀 언어모델에 주입하는 retrieval-enhanced transformer다. 핵심은 "chunked cross-attention" 메커니즘인데, 입력을 일정 청크 단위로 자르고 각 청크에 대해 BERT 임베딩 기반으로 검색한 이웃 문서를 cross-attention으로 결합한다. 7.5B 파라미터의 RETRO가 GPT-3(175B) 및 Jurassic-1(178B)에 필적하는 Pile 성능을 25배 적은 파라미터로 달성했다.

### 핵심 기여
- **Trillion-scale 외부 메모리**: 2조 토큰 규모의 retrieval index를 frozen BERT 임베딩으로 구축, 모델 파라미터에 모든 지식을 압축할 필요 없이 "비파라메트릭 메모리"로 분리
- **Chunked Cross-Attention (CCA)**: 입력을 m=64 토큰 청크로 자르고 청크별로 k개 이웃을 검색, 검색 결과를 시간순서 보존하며 cross-attention으로 통합하는 새로운 어텐션 메커니즘
- **데이터 누수 분석 프레임워크**: 평가 데이터셋과 retrieval corpus의 중복도를 정량 측정하는 방법론을 제안해 retrieval-augmented LM의 "벤치마크 부풀리기" 문제를 처음으로 체계적으로 다룸
- **Scaling 곡선 검증**: 모델 크기(150M~7B), retrieval DB 크기, 검색 이웃 수(k)에 따른 성능 변화를 광범위하게 측정해 retrieval이 단순 fine-tuning 효과가 아닌 본질적 capacity 확장임을 입증

### 이 논문이 중요한 이유
"파라미터 vs 외부 메모리"라는 LLM 아키텍처의 근본적 트레이드오프를 정량적으로 풀어낸 첫 대규모 연구다. 오늘날 RAG가 비용 효율적인 이유, 작은 모델 + 좋은 검색이 큰 모델을 이기는 이유의 이론적 근거가 모두 이 논문에 있다. AI 엔지니어가 "왜 13B + RAG가 70B closed-book보다 나은가"를 설명할 때 인용해야 할 first principle이다. 또한 chunked cross-attention 설계는 이후 Atlas, REPLUG, In-Context RALM 등 대부분의 retrieval-augmented architecture의 출발점이 되었다.

### 사전 지식
- Transformer decoder-only 구조와 cross-attention 메커니즘
- BERT 기반 임베딩 검색과 ScaNN/FAISS approximate nearest neighbor
- Language modeling 평가 지표 (perplexity, bits-per-byte)
- Scaling laws (Kaplan 2020, Chinchilla 2022) — 본 논문이 Chinchilla 팀과 같은 DeepMind 작품이라는 맥락

### 관련 논문
- [Generalization through Memorization: Nearest Neighbor Language Models / kNN-LM (Khandelwal et al., 2020)](https://arxiv.org/abs/1911.00172)
- [REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al., 2020)](https://arxiv.org/abs/2002.08909)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)

### 실무 적용
RETRO의 직접적 구현은 학습 비용 때문에 산업에서 드물지만, 핵심 아이디어는 거의 모든 enterprise RAG에 녹아있다. (1) "청크 단위로 검색하고 cross-attention으로 결합" → 현재 LangChain/LlamaIndex의 chunk-based retrieval + context stuffing 패턴, (2) "frozen retriever + tunable generator" → 대부분의 사내 챗봇이 임베딩 모델은 고정, generator만 fine-tune하는 분업 구조, (3) "데이터 누수 측정" → 사내 RAG 평가 시 학습 데이터와 retrieval corpus 중복 체크의 표준 절차. Nvidia의 ChatQA, Anthropic의 contextual retrieval 같은 최신 enterprise solution도 RETRO의 설계 철학을 계승한다.

---

## Paper 2 (Classic): Atlas: Few-shot Learning with Retrieval Augmented Language Models
- **Authors:** Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, Edouard Grave (Meta AI / FAIR)
- **Year:** 2022 (JMLR 2023)
- **arXiv:** https://arxiv.org/abs/2208.03299
- **PDF:** [./atlas-izacard-2022.pdf](./atlas-izacard-2022.pdf)
- **Citation Count:** ~900+ (Meta의 RAG 대표작, FiD 후속)

### 요약
Atlas는 retriever(Contriever)와 generator(T5 + Fusion-in-Decoder)를 **공동으로 사전학습**해 few-shot 환경에서도 강력한 지식 집약 태스크 성능을 달성한 retrieval-augmented LM이다. 단 64개 학습 샘플만으로 Natural Questions에서 42% 이상 정확도를 달성해, 540B 파라미터의 PaLM을 50배 작은 11B 모델로 능가했다. 핵심은 retriever와 generator를 분리 학습하지 않고 4가지 loss(ADist, EMDR², PDist, LOOP)로 공동 최적화하는 학습 레시피와 MLM·prefix LM·title-section generation 등 다양한 사전학습 objective의 비교 분석이다.

### 핵심 기여
- **Retriever-Generator Joint Pretraining**: 라벨링된 supervision 없이 retriever와 generator를 4종 손실로 공동 사전학습하는 방법 제시 — 특히 EMDR²(Expectation-Maximization Dense Retriever)이 안정적
- **Few-shot SOTA**: NaturalQuestions, TriviaQA, MMLU, KILT 벤치마크에서 64-shot setting으로 540B closed-book 모델을 능가, "RAG는 few-shot에서 더 강력하다"는 명제를 입증
- **인덱스 업데이트 가능성 실증**: 학습 후에도 retrieval corpus를 갈아끼우는 것만으로 새로운 도메인/시점의 지식 반영이 가능 — TempLAMA temporal benchmark에서 재학습 없이 시간 업데이트 검증
- **Fusion-in-Decoder (FiD) 통합**: 100개 이상의 retrieved passage를 효율적으로 처리하는 decoder-level fusion을 retrieval-augmented LM의 표준 generator 구조로 정립

### 이 논문이 중요한 이유
"RAG는 단지 inference 시 검색을 붙이는 것이 아니라, 학습 단계부터 검색 능력을 갖춘 모델을 만드는 것"이라는 관점을 정립했다. 이는 오늘날 ChatGPT의 Browse with Bing, Perplexity, Claude의 contextual retrieval, Gemini의 Search Grounding으로 이어지는 "검색 능력이 내장된 LM"의 직접적 조상이다. 또한 few-shot 능력은 enterprise RAG의 가장 큰 가치 제안 — "고객사 데이터로 fine-tuning 없이 빠르게 도메인 적응" — 의 학술적 근거가 된다. AI 엔지니어가 LoRA-tuning과 RAG 중 어느 쪽을 택할지 결정할 때 Atlas의 few-shot 곡선이 결정적 참고자료다.

### 사전 지식
- DPR/Contriever 같은 dual-encoder retriever와 contrastive learning
- Fusion-in-Decoder (Izacard & Grave 2020, arXiv:2007.01282)
- T5 encoder-decoder 구조와 prefix LM / MLM 사전학습
- EM 알고리즘, latent variable model (retrieved document를 latent로 다룸)
- Few-shot evaluation 프로토콜 (PaLM, GPT-3 스타일)

### 관련 논문
- [Leveraging Passage Retrieval with Generative Models / FiD (Izacard & Grave, 2020)](https://arxiv.org/abs/2007.01282)
- [Unsupervised Dense Information Retrieval with Contrastive Learning / Contriever (Izacard et al., 2021)](https://arxiv.org/abs/2112.09118)
- [RETRO: Improving language models by retrieving from trillions of tokens (Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426)
- [REPLUG: Retrieval-Augmented Black-Box Language Models (Shi et al., 2023)](https://arxiv.org/abs/2301.12652)

### 실무 적용
Atlas의 joint training은 비용이 커서 일반 기업이 직접 재현하지 않지만, 그 학습 레시피와 평가 프로토콜은 enterprise RAG 평가의 표준이 되었다. (1) **few-shot evaluation harness**: 사내 RAG 시스템을 평가할 때 학습 데이터 0~64개 구간에서 성능 곡선을 그려보는 것은 Atlas에서 정립된 관행, (2) **인덱스 hot-swap 패턴**: 매 시점 새 뉴스/문서로 corpus를 갱신하는 production RAG의 표준 워크플로우, (3) **FiD-style multi-passage fusion**: Cohere Command-R, Mistral RAG-optimized 모델 등 retrieval-tuned 모델들이 채택. Meta의 Llama 3.1에 포함된 long-context RAG capability도 Atlas 라인업의 직계 후손이다.

---

## Paper 3 (Recent): A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions
- **Authors:** Shailja Gupta, Rajesh Ranjan, Surya Narayan Singh
- **Year:** 2024 (October)
- **arXiv:** https://arxiv.org/abs/2410.12837
- **PDF:** [./rag-comprehensive-survey-gupta-2024.pdf](./rag-comprehensive-survey-gupta-2024.pdf)
- **Citation Count:** ~150+ (빠르게 인용 누적 중)

### 요약
2024년 10월에 출간된 본 서베이는 Naive RAG → Advanced RAG → Modular RAG → Agentic RAG로 이어지는 RAG 아키텍처의 진화를 체계적으로 정리한다. 단순 retrieve-then-generate에서 시작해, query rewriting·reranking·hybrid search 같은 advanced 기법, 그리고 retriever/generator/memory/router를 독립 모듈로 조립하는 modular 패러다임까지의 발전 경로를 단계별로 다룬다. 또한 hallucination, scalability, multi-modal RAG, evaluation framework(RAGAS, ARES) 등 현재 산업이 직면한 핵심 과제와 향후 연구 방향을 제시한다.

### 핵심 기여
- **RAG 진화 분류 체계**: Naive → Advanced → Modular RAG의 3세대 분류를 도식화하고 각 세대별 대표 기법(query expansion, HyDE, RAG-Fusion, Self-RAG, CRAG 등)을 매핑
- **Modular RAG 컴포넌트 분해**: Retrieval, Memory, Routing, Fusion, Predict, Task Adapter 등 7개 모듈로 RAG를 분해하고 LangChain/LlamaIndex/Haystack에서의 매핑을 제시
- **Adaptive Retrieval 정리**: 쿼리 특성에 따라 검색 횟수/소스/전략을 동적으로 결정하는 adaptive RAG (FLARE, Self-RAG, IR-CoT) 비교 분석
- **Evaluation Landscape**: faithfulness, answer relevance, context precision/recall 등 RAG 전용 메트릭과 RAGAS, ARES, TruLens 같은 평가 프레임워크의 강약점 비교
- **Future Directions**: multi-modal RAG, real-time RAG, federated RAG, privacy-preserving retrieval, agentic RAG 등 6대 미래 연구 영역 제시

### 이 논문이 중요한 이유
2024년 시점에서 RAG의 "전체 지형도"를 한 장으로 보여주는 최신 서베이다. 실무에서 RAG 아키텍처 결정을 할 때 (Naive로 충분한가? Advanced 기법이 필요한가? Modular까지 가야 하는가?) 의사결정 프레임워크를 제공한다. 특히 enterprise RAG가 PoC(Naive RAG)에서 production(Modular RAG)로 넘어가는 단계에서 어떤 추가 모듈을 우선 도입해야 하는지에 대한 학술적 가이드라인 역할을 한다. CPO/엔지니어링 리더가 팀에 "왜 우리는 단순 RAG에서 멈추면 안 되는가"를 설명할 때 첫 번째로 공유할 자료.

### 사전 지식
- 기본 RAG 파이프라인 (embedding → retrieval → context stuffing → generation)
- Vector database 기초 (FAISS, Pinecone, Weaviate, Qdrant)
- BM25 vs dense retrieval vs hybrid search
- LangChain/LlamaIndex의 기본 컴포넌트 (Retriever, Reranker, Chain)
- LLM evaluation 일반 (LLM-as-judge, reference-based vs reference-free)

### 관련 논문
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation / CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (Singh et al., 2025)](https://arxiv.org/abs/2501.09136)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217)

### 실무 적용
이 서베이는 그 자체가 "실무 적용 매뉴얼"이다. (1) **RAG 성숙도 모델**로 활용: 사내 RAG 시스템을 Naive/Advanced/Modular 중 어디에 위치시킬지 진단 → 다음 단계 로드맵 수립, (2) **Modular RAG 컴포넌트 체크리스트**: query rewriter, reranker, fusion strategy, fallback router 중 우리 제품에 빠진 것이 무엇인지 점검, (3) **평가 프레임워크 선택**: RAGAS vs ARES vs TruLens 중 우리 사례에 맞는 evaluation stack 선택, (4) **로드맵 정당화**: PM/CPO가 "왜 다음 분기에 Self-RAG·CRAG 같은 adaptive retrieval에 투자해야 하는가"를 경영진에 설명할 때 인용. 특히 Agentic AI 제품을 리딩하는 입장에서 modular→agentic 전이의 학술적 기반을 제공한다.

---

## 추천 읽기 순서
1. **Gupta 2024 서베이 (Paper 3)** 먼저 — RAG 전체 지형도를 30분 안에 잡은 뒤 어디에 깊이 들어갈지 결정
2. **RETRO (Paper 1)** — 왜 retrieval-augmentation이 단순 prompting이 아닌 architectural choice인지 first principle 이해
3. **Atlas (Paper 2)** — joint training과 few-shot 능력이라는 RAG의 차세대 방향 학습 (실험 섹션은 선택적)

### 시간이 부족하다면
- 30분 코스: Gupta 2024의 Section 3(Evolution)과 Section 5(Modular RAG)만
- 2시간 코스: 위 + RETRO의 Section 2(Method)와 Figure 2의 chunked cross-attention
- 반나절 코스: 3편 모두, Atlas는 Section 3(Training Objectives) 집중

---

## 핵심 테이크어웨이
- **외부 메모리는 파라미터 확장의 대체재이자 보완재다.** RETRO가 보여줬듯 7.5B + 2T 토큰 검색 ≈ 175B closed-book이다. "더 큰 모델"이 아닌 "더 나은 검색"이 ROI가 높은 길임을 항상 기억할 것.
- **Retriever와 generator를 분리 vs 공동 학습은 비용 곡선의 문제다.** PoC에서는 분리(off-the-shelf embedding + LLM)가 빠르지만, scale-up 단계에서는 Atlas 식 joint tuning이 데이터 효율을 결정한다.
- **Naive RAG는 production의 출발점이지 종착점이 아니다.** Gupta 서베이가 정리하듯, 사용자 만족도가 정체되기 시작하면 query rewriting → reranking → adaptive retrieval → agentic loop 순으로 모듈을 점진 추가하는 것이 검증된 진화 경로다.
- **평가가 곧 제품이다.** RAGAS/ARES 같은 RAG 전용 메트릭 없이 LLM-as-judge로만 의사결정하면 hallucination·context-utilization의 회귀를 놓친다. 평가 인프라 투자가 모델 교체보다 우선순위가 높을 수 있다.

---

## 다음 토픽과의 연결
다음 day 26은 **Advanced RAG: Self-RAG / Corrective RAG**다. 오늘 다룬 RETRO·Atlas가 "검색을 architecture에 통합"하는 방향이었다면, 내일은 "LLM이 스스로 검색이 필요한지·검색 결과가 정확한지 판단"하는 한 단계 진화한 adaptive RAG를 다룬다. Gupta 서베이의 Section 5(Adaptive Retrieval) 부분을 미리 훑어두면 자연스럽게 이어진다. 또한 Self-RAG·CRAG가 결국 agent loop(reflection + tool use)로 발전한다는 점에서, Module 8(LangChain·Agents)에서 다룬 ReAct·Tree-of-Thoughts와도 직접 연결된다.
