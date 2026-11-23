# Daily AI Paper Recommendations

> **Date:** 2026-11-24
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): Learning How to Ask: Querying LMs with Mixtures of Soft Prompts
- **Authors:** Guanghui Qin, Jason Eisner
- **Year:** 2021 (NAACL-HLT 2021, Best Short Paper)
- **arXiv:** https://arxiv.org/abs/2104.06599
- **PDF:** [./soft-prompt-mixtures-qin-2021.pdf](./soft-prompt-mixtures-qin-2021.pdf)
- **Citation Count:** ~600+

### 요약
사람이 손으로 쓴 이산(discrete) 프롬프트 대신, 실제 단어가 아닌 연속 벡터("soft words")로 이루어진 프롬프트를 경사하강법으로 학습해 언어모델에서 사실 지식을 끌어내는 방법을 제안합니다. 여러 개의 soft prompt를 학습된 가중치로 섞는 "mixture of prompts"를 함께 사용해, 기존 수작업/탐색 기반 프롬프트보다 지식 추출(LAMA 등) 성능을 크게 끌어올렸습니다.

### 핵심 기여
- 프롬프트를 이산 토큰 공간이 아닌 연속 임베딩 공간에서 직접 최적화하는 "soft prompt" 개념을 지식 질의(probing)에 체계적으로 적용
- 여러 프롬프트를 가중 앙상블하는 Mixture-of-Prompts로 단일 프롬프트의 취약성을 완화
- 랜덤 초기화도 정보 기반 초기화와 거의 비슷한 성능 → "좋은 질문 문장"보다 "최적화 가능한 질문 공간"이 중요함을 시사
- LM이 가진 사실 지식이 기존 방식으로는 과소평가되어 왔다는 실증적 증거 제시

### 이 논문이 중요한 이유
"프롬프트 성능이 낮다 = 모델이 모른다"가 아니라 "질문을 잘못했다"일 수 있다는 관점을 정량적으로 보여줍니다. Prefix-Tuning, Prompt Tuning, P-Tuning과 같은 시기에 등장한 연속 프롬프트 계열의 핵심 논문으로, 이후 PEFT(파라미터 효율 미세조정) 흐름을 이해하는 기반이 됩니다.

### 사전 지식
- 단어 임베딩과 Transformer 입력 구조
- Masked LM(BERT 계열)과 LAMA 같은 지식 probing 벤치마크
- 경사하강법, softmax 가중 앙상블 개념

### 관련 논문
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)
- [AutoPrompt (Shin et al., 2020)](https://arxiv.org/abs/2010.15980)
- [How Can We Know What Language Models Know? (Jiang et al., 2020)](https://arxiv.org/abs/1911.12543)

### 실무 적용
오픈 웨이트 모델을 운영할 때, 모델 전체를 미세조정하지 않고 작은 soft prompt 벡터만 학습해 태스크별 어댑터처럼 교체하는 방식으로 활용됩니다. 멀티테넌트 SaaS에서 고객/태스크마다 수 KB 수준의 프롬프트 벡터만 저장해 서빙 비용을 줄이는 패턴의 원형입니다. 단, API형 폐쇄 모델에서는 임베딩 접근이 불가하므로 적용이 제한됩니다.

---

## Paper 2 (Classic): PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization
- **Authors:** Xinyuan Wang, Chenxi Li, Zhen Wang, Fan Bai, Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric P. Xing, Zhiting Hu
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.16427
- **PDF:** [./promptagent-wang-2023.pdf](./promptagent-wang-2023.pdf)
- **Citation Count:** ~200+

### 요약
프롬프트 최적화를 "전략적 계획(planning) 문제"로 보고, 몬테카를로 트리 탐색(MCTS)으로 프롬프트 공간을 탐색합니다. 모델이 틀린 사례를 분석해 "오류 피드백"을 만들고, 이를 반영해 프롬프트를 수정하는 행동을 반복하면서 미래 보상을 시뮬레이션해 전문가 수준의 상세한 프롬프트에 도달합니다.

### 핵심 기여
- 프롬프트 최적화를 상태(프롬프트)·행동(오류 기반 수정)·보상(검증 성능)으로 정식화한 MDP + MCTS 프레임워크
- 단순 재작성(paraphrase)이 아닌, 오류 원인 분석에 기반한 "도메인 지식 주입형" 프롬프트 생성
- BBH, 의료·생물 도메인, 일반 NLP 등 12개 태스크에서 CoT 및 APE 등 기존 최적화 기법 대비 큰 성능 향상
- 최적화된 프롬프트가 다른 모델(GPT-4, PaLM 2 등)로도 일정 부분 전이됨을 보임

### 이 논문이 중요한 이유
APE/OPRO가 "짧고 일반적인 지시문"을 찾는 데 머물렀다면, PromptAgent는 실무 전문가가 쓰는 것처럼 예외 처리·도메인 규칙이 담긴 긴 프롬프트를 자동으로 만들어냅니다. 이후 TextGrad, GEPA 등 "반성(reflection)+탐색" 기반 최적화기의 직접적인 선행 연구입니다.

### 사전 지식
- 몬테카를로 트리 탐색(UCT 선택, 확장, 시뮬레이션, 역전파)
- APE, ProTeGi(텍스트 그래디언트) 등 LLM 기반 프롬프트 최적화
- 검증 세트 기반 평가와 과적합 개념

### 관련 논문
- [Automatic Prompt Optimization with "Gradient Descent" and Beam Search (Pryzant et al., 2023)](https://arxiv.org/abs/2305.03495)
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [Reasoning with Language Model is Planning with World Model / RAP (Hao et al., 2023)](https://arxiv.org/abs/2305.14992)
- [Large Language Models Are Human-Level Prompt Engineers / APE (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)

### 실무 적용
CS 분류, 의료 문서 태깅, 계약서 조항 추출처럼 "도메인 규칙이 많고 오류 패턴이 반복되는" 태스크에서, 소량의 라벨 데이터와 실패 로그만으로 시스템 프롬프트를 자동 고도화하는 파이프라인에 적합합니다. 탐색 비용(LLM 호출 수)이 크므로 오프라인 배치 최적화 → 결과 프롬프트를 버전 관리해 배포하는 형태가 현실적입니다.

---

## Paper 3 (Recent): Self-Supervised Prompt Optimization (SPO)
- **Authors:** Jinyu Xiang, Jiayi Zhang, et al. (11 authors)
- **Year:** 2025 (Findings of EMNLP 2025)
- **arXiv:** https://arxiv.org/abs/2502.06855
- **PDF:** [./self-supervised-prompt-optimization-xiang-2025.pdf](./self-supervised-prompt-optimization-xiang-2025.pdf)
- **Code:** https://github.com/FoundationAgents/SPO
- **Citation Count:** ~50+ (2025년 발표, 빠르게 증가 중)

### 요약
기존 프롬프트 최적화는 정답 라벨이나 사람 평가가 필요했습니다. SPO는 정답 없이 "최적화 → 실행 → 평가" 루프를 돌리며, LLM이 두 프롬프트의 출력을 쌍대 비교(pairwise)로 판정하고 LLM 최적화기가 현재 최고 프롬프트와 그 출력을 보고 프롬프트를 다시 씁니다. 저자들은 기존 방법 대비 매우 낮은 비용으로 동등 이상의 성능을, 3개 수준의 샘플만으로도 달성한다고 보고합니다.

### 핵심 기여
- Ground truth 없이 출력 자체를 비교 신호로 쓰는 "자기지도(self-supervised)" 프롬프트 최적화 프레임워크
- Output-vs-Output 쌍대 비교로 절대 점수 평가의 불안정성 완화
- 극소량 샘플·저비용으로 폐쇄형/개방형 태스크 모두에서 경쟁력 있는 성능
- 오픈소스(MetaGPT 생태계)로 공개되어 바로 실험 가능

### 이 논문이 중요한 이유
실제 제품에서는 "정답 라벨이 있는 평가셋"이 없는 경우가 대부분입니다(카피라이팅, 요약, 에이전트 응답 품질 등). SPO는 라벨 수집 비용이라는 프롬프트 최적화의 가장 큰 현실적 장벽을 정면으로 다루며, LLM-as-a-Judge와 프롬프트 최적화를 결합하는 최신 흐름을 보여줍니다.

### 사전 지식
- LLM-as-a-Judge, 쌍대 비교(pairwise preference) 평가
- OPRO/TextGrad 등 LLM을 최적화기로 쓰는 방식
- 평가자 편향(위치 편향, 길이 편향) 개념

### 관련 논문
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (Agrawal et al., 2025)](https://arxiv.org/abs/2507.19457)
- [AFlow: Automating Agentic Workflow Generation (Zhang et al., 2024)](https://arxiv.org/abs/2410.10762)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)

### 실무 적용
라벨 없는 신규 기능(예: AI 아바타 대본 생성, 더빙 자막 톤 조정, 에이전트 답변 스타일)에 대해 몇 개의 대표 입력만 준비하면 시스템 프롬프트를 자동 개선할 수 있습니다. 다만 Judge 모델의 편향이 그대로 최적화 방향이 되므로, 최종 채택 전에는 소규모 휴먼 리뷰나 온라인 A/B 테스트로 검증하는 것을 권장합니다.

---

## 추천 읽기 순서
1. **Qin & Eisner (2021)** — "프롬프트를 최적화한다"는 개념의 가장 직관적 형태(연속 벡터 + 경사하강법)부터 이해
2. **PromptAgent (2023)** — 이산 텍스트 공간에서 탐색(MCTS)과 오류 피드백으로 최적화하는 방식으로 확장
3. **SPO (2025)** — 정답 라벨까지 제거한 최신 실용형 최적화로 마무리

## 핵심 테이크어웨이
- 프롬프트 최적화는 **무엇을 최적화하는가(연속 벡터 vs 텍스트)**, **어떻게 탐색하는가(그래디언트 vs 탐색/진화/반성)**, **무엇으로 평가하는가(라벨 vs LLM Judge)**의 세 축으로 정리할 수 있습니다.
- 최신 흐름은 "그래디언트 접근 불가한 API 모델 + 라벨 부족" 현실에 맞춰 **텍스트 공간 탐색 + LLM 기반 평가**로 수렴하고 있습니다.
- 자동 최적화의 품질 상한은 결국 **평가 신호의 품질**에 좌우됩니다. 평가셋/Judge 설계가 프롬프트 설계보다 중요해지는 지점입니다.

## 다음 토픽과의 연결
다음 모듈 **LangChain and LLM Orchestration**의 "LLM Application Frameworks and Orchestration"에서는 단일 프롬프트를 넘어 여러 LLM 호출·도구·검색을 엮은 **Compound AI System**을 다룹니다. 오늘 본 최적화 기법(특히 SPO, DSPy, AFlow 계열)은 이런 다단계 파이프라인 전체의 프롬프트와 워크플로를 자동 튜닝하는 방향으로 확장되고 있어, 오케스트레이션 설계 시 "최적화 가능한 구조"를 염두에 두는 관점으로 이어집니다.
