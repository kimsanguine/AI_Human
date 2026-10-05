# Daily AI Paper Recommendations

> **Date:** 2026-10-06
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** RAG Architecture and Optimization

---

## Paper 1 (Classic): RA-DIT: Retrieval-Augmented Dual Instruction Tuning
- **Authors:** Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, Scott Yih
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.01352
- **PDF:** [./ra-dit-lin-2023.pdf](./ra-dit-lin-2023.pdf)
- **Citation Count:** ~250+

### 요약
RA-DIT는 이미 학습된 LLM과 리트리버를 "나중에" 연결해 RAG 시스템으로 개조(retrofit)하는 경량 파인튜닝 방법이다. ① LLM은 검색된 문서가 붙은 지시문에서 정답 확률을 높이도록, ② 리트리버는 LLM이 실제로 도움을 받는 문서를 더 높게 점수 매기도록(KL 최소화) 각각 한 번씩 튜닝한다. RA-DIT 65B는 지식 집약형 제로샷 벤치마크에서 기존 In-Context RALM 대비 평균 +8.9%p 향상을 보였다.

### 핵심 기여
- **Dual Instruction Tuning:** 언어모델 튜닝(LM-ft)과 리트리버 튜닝(R-ft)을 분리된 두 단계로 설계해, 값비싼 end-to-end 사전학습(REALM/Atlas류) 없이 RAG 성능 확보
- **LLM 선호 기반 리트리버 학습:** "사람이 보기에 관련 있는 문서"가 아니라 "LLM이 정답을 내는 데 실제로 기여한 문서"를 기준으로 리트리버를 정렬 (LSR: LM-Supervised Retrieval)
- **노이즈 내성 학습:** 학습 시 일부러 관련 없는 문서를 섞어, LLM이 검색 결과가 틀렸을 때 자체 지식으로 답하는 능력을 유지하도록 함

### 이 논문이 중요한 이유
RAG 아키텍처를 "리트리버 + 생성기"의 두 모듈로 보고, 각 모듈을 서로에게 맞춰 정렬한다는 관점을 정립했다. 실무에서 "검색은 잘 되는데 답이 별로"이거나 "답은 그럴듯한데 근거 문서가 엉뚱한" 문제의 원인이 두 모듈 간 미스얼라인먼트라는 점을 이해하는 데 핵심이다.

### 사전 지식
- RAG (Lewis et al., 2020), REPLUG의 LM-supervised retrieval 개념
- Instruction Tuning, KL Divergence
- Dense Retriever (DPR/Dragon) 구조

### 관련 논문
- [REPLUG: Retrieval-Augmented Black-Box Language Models (Shi et al., 2023)](https://arxiv.org/abs/2301.12652)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)
- [In-Context Retrieval-Augmented Language Models (Ram et al., 2023)](https://arxiv.org/abs/2302.00083)

### 실무 적용
도메인 RAG(사내 문서, 법률, 의료)에서 오픈소스 LLM + 임베딩 모델을 함께 파인튜닝할 때의 레시피로 쓰인다. 특히 "검색 문서를 붙인 SFT 데이터 구성"과 "정답 문서가 아닌 노이즈 문서를 섞는 비율"은 사내 RAG 파인튜닝 데이터셋 설계에 그대로 적용 가능하다.

---

## Paper 2 (Classic): RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
- **Authors:** Fangyuan Xu, Weijia Shi, Eunsol Choi
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.04408
- **PDF:** [./recomp-xu-2023.pdf](./recomp-xu-2023.pdf)
- **Citation Count:** ~250+

### 요약
RECOMP는 검색된 문서를 그대로 프롬프트에 넣지 않고, 먼저 짧은 텍스트 요약으로 "압축"한 뒤 LLM에 전달하는 방법이다. 유용한 문장만 고르는 추출형(extractive) 압축기와 여러 문서를 종합해 요약하는 생성형(abstractive) 압축기를 제안하며, 문서가 도움이 안 되면 빈 문자열을 출력해 아예 검색 결과를 쓰지 않는 "선택적 증강"을 구현한다. 원문의 약 6% 길이로 압축하면서도 성능 손실이 거의 없었다.

### 핵심 기여
- **컨텍스트 압축을 RAG의 독립 단계로 정의:** Retrieve → Compress → Generate 파이프라인 정립
- **End-task 기반 압축기 학습:** 요약 품질이 아니라 "다운스트림 LLM의 정답률/퍼플렉서티"를 신호로 압축기를 학습 (작은 모델로도 off-the-shelf 요약기보다 우수)
- **Selective Augmentation:** 관련 없는 문서면 빈 출력 → 노이즈 주입 방지 + 토큰 비용 절감

### 이 논문이 중요한 이유
RAG 최적화의 핵심 트레이드오프인 "더 많은 컨텍스트 vs 비용·지연·Lost-in-the-Middle"을 정면으로 다룬다. Top-k를 늘리는 것만이 답이 아니며, 컨텍스트를 줄이는 것이 정확도와 비용 모두를 개선할 수 있다는 점을 보여준 대표 논문이다.

### 사전 지식
- RAG 기본 파이프라인, Top-k 검색
- 요약 모델(T5 등)과 지식 증류(distillation)
- Lost in the Middle 현상 (긴 컨텍스트에서 중간 정보 누락)

### 관련 논문
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (Jiang et al., 2023)](https://arxiv.org/abs/2310.06839)
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)
- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context (Yoran et al., 2023)](https://arxiv.org/abs/2310.01558)

### 실무 적용
대량 문서 RAG에서 토큰 비용과 응답 지연을 줄이는 "컨텍스트 압축 레이어"의 원형이다. LangChain의 ContextualCompressionRetriever, LLMLingua 등 실무 도구의 이론적 배경이며, 소형 모델(예: 1B급)로 압축기를 두고 대형 LLM은 생성에만 쓰는 비용 최적화 아키텍처에 바로 응용된다.

---

## Paper 3 (Recent): RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs
- **Authors:** Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, Bryan Catanzaro
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2407.02485
- **PDF:** [./rankrag-yu-2024.pdf](./rankrag-yu-2024.pdf)
- **Citation Count:** ~200+

### 요약
RankRAG는 하나의 LLM을 "컨텍스트 재순위화(reranking)"와 "답변 생성" 두 역할에 동시에 쓰도록 instruction tuning하는 프레임워크다. 학습 데이터에 소량의 랭킹 데이터만 섞어도, 대량 랭킹 데이터로 학습한 전문 리랭커보다 뛰어난 성능을 보였다. Llama3-RankRAG는 9개 지식 집약 벤치마크에서 Llama3-ChatQA-1.5와 GPT-4를 크게 앞섰고, 바이오메디컬 도메인 5개 벤치마크에서도 별도 튜닝 없이 GPT-4와 비슷한 성능을 냈다.

### 핵심 기여
- **Retrieve → Rerank → Generate를 단일 LLM으로 통합:** 별도 크로스인코더 리랭커 없이 LLM 자체가 top-N 중 관련 문서를 골라냄
- **데이터 효율성:** 랭킹 데이터를 "조금만" 섞어도 랭킹·생성 능력이 상호 강화됨을 실증
- **Top-k 딜레마 해결:** k가 작으면 recall 부족, 크면 노이즈 증가 → 넓게 검색 후 LLM이 좁히는 구조로 해결

### 이 논문이 중요한 이유
2024년 이후 RAG 최적화의 핵심 레버가 "리트리버 개선"에서 "리랭킹 + 생성기 정렬"로 이동했음을 보여준다. RA-DIT(정렬), RECOMP(압축)의 문제의식을 "랭킹"이라는 축으로 이어받아 하나의 모델로 통합한 최신 흐름의 대표 사례다.

### 사전 지식
- Cross-encoder 리랭커(monoT5, BGE-reranker 등) 개념
- Instruction tuning, ChatQA 계열 모델
- Recall@k, EM/F1 등 RAG 평가 지표

### 관련 논문
- [ChatQA: Surpassing GPT-4 on Conversational QA and RAG (Liu et al., 2024)](https://arxiv.org/abs/2401.10225)
- [RankGPT: Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents (Sun et al., 2023)](https://arxiv.org/abs/2304.09542)
- [Searching for Best Practices in Retrieval-Augmented Generation (Wang et al., 2024)](https://arxiv.org/abs/2407.01219)

### 실무 적용
프로덕션 RAG에서 "임베딩 검색 top-50 → 리랭커 top-5 → LLM" 구조의 리랭커 단계를 LLM으로 흡수해 모델 서빙 수를 줄일 수 있다. 또한 자체 파인튜닝 시 "랭킹 태스크 데이터 일부 혼합"이라는 저비용 레시피로 근거 선택 정확도를 올리는 데 활용 가능하다.

---

## 추천 읽기 순서
1. **RECOMP** — 가장 직관적. "검색 결과를 그대로 넣으면 왜 문제인가?"라는 질문에서 출발해 RAG 최적화 감각을 잡는다.
2. **RA-DIT** — 리트리버와 LLM을 서로에게 맞춰 정렬한다는 아키텍처 관점을 익힌다.
3. **RankRAG** — 위 두 아이디어(노이즈 제거 + 모듈 정렬)가 2024년에 단일 LLM 랭킹·생성 통합으로 어떻게 진화했는지 확인한다.

## 핵심 테이크어웨이
- **Q. RAG 성능이 안 나올 때 가장 먼저 의심할 것은?** → 리트리버 자체보다 "리트리버와 생성기의 미스얼라인먼트"와 "컨텍스트 노이즈"인 경우가 많다.
- **Q. 컨텍스트는 많을수록 좋은가?** → 아니다. 압축(RECOMP)·재순위화(RankRAG)로 줄이는 것이 정확도·비용·지연을 동시에 개선한다.
- **Q. 무엇을 학습 신호로 써야 하나?** → 세 논문 모두 "사람 기준 관련성"이 아니라 "LLM의 최종 답변 품질"을 최적화 목표로 삼는다. 이것이 RAG 최적화의 공통 원리다.
- **Q. 언제 검색 결과를 버려야 하나?** → 선택적 증강(빈 압축 출력), 노이즈 내성 학습처럼 "검색을 안 쓰는 판단"도 아키텍처의 일부다.

## 다음 토픽과의 연결
다음 토픽 **Advanced RAG (Self-RAG, Corrective RAG)** 는 오늘 다룬 "언제 검색 결과를 쓰고 버릴지"를 모델이 스스로 판단·비평하게 만드는 방향으로 확장된다. RECOMP의 선택적 증강 → Self-RAG의 reflection token, RankRAG의 컨텍스트 선별 → CRAG의 검색 품질 평가기로 자연스럽게 이어진다.
