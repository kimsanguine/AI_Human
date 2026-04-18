# Daily AI Paper Recommendations

> **Date:** 2026-04-19
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Efficient LLM — Quantization and Distillation

---

## Paper 1 (Classic): LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2106.09685
- **PDF:** [./lora-low-rank-adaptation-hu-2021.pdf](./lora-low-rank-adaptation-hu-2021.pdf)
- **Citation Count:** 15,000+ (2026년 기준, Google Scholar)

### 요약
LoRA는 사전학습된 거대 언어모델의 가중치를 동결(freeze)한 채, 각 Transformer 레이어의 가중치 행렬에 저차원(rank) 분해 행렬을 주입하여 그 부분만 학습하는 파라미터 효율적 파인튜닝(PEFT) 기법이다. GPT-3 175B 기준으로 학습 파라미터를 1만분의 1 수준까지 줄이면서도 풀 파인튜닝 대비 동등하거나 더 나은 성능을 달성했고, 추론 시 병합이 가능해 추가 지연(latency)도 발생하지 않는다.

### 핵심 기여
- ΔW = B·A (rank r ≪ d) 저차원 분해로 학습 파라미터를 최대 10,000배까지 축소
- 추론 시 W ← W + BA로 병합 가능해 어댑터 방식과 달리 지연 비용 0
- 여러 다운스트림 태스크마다 작은 LoRA 모듈만 스왑하면 되어 멀티태스크 배포에 최적
- 풀 파인튜닝, 어댑터, 프리픽스 튜닝 대비 동등/우위의 성능을 GPT-3, RoBERTa, DeBERTa에서 실증

### 이 논문이 중요한 이유
AI 엔지니어에게 LoRA는 "거대 모델을 적은 GPU로 커스터마이즈"하는 사실상의 표준 레시피다. Hugging Face PEFT, Unsloth, Axolotl 등 거의 모든 오픈소스 파인튜닝 스택의 기본 선택지이며, SD/Flux 같은 이미지 생성 모델까지 확장돼 있어 이 원리를 모르면 최신 모델 커스터마이징/배포 파이프라인 자체를 이해할 수 없다.

### 사전 지식
- Transformer 구조 (self-attention의 Q, K, V, O 프로젝션 행렬)
- 행렬의 랭크(rank)와 SVD 기반 저차원 근사
- Full fine-tuning vs. adapter/prefix-tuning의 파라미터·지연 트레이드오프

### 관련 논문
- [Adapter Tuning: Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)

### 실무 적용
사내 도메인 지식으로 Llama/Qwen/Gemma 계열을 튜닝할 때 단일 A100/H100 한 장으로도 7B~13B 모델을 학습 가능. LoRA 어댑터만 저장(수십 MB)하여 베이스 모델 하나에 고객별·테넌트별 어댑터를 스왑 서빙하는 멀티테넌트 아키텍처(예: vLLM의 multi-LoRA, LoRAX)가 가능하다. AI Dubbing/Avatar 같은 제품에서도 캐릭터·화자 스타일을 LoRA로 관리하면 저장·운영 비용을 크게 절감할 수 있다.

---

## Paper 2 (Classic): QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.14314
- **PDF:** [./qlora-efficient-finetuning-quantized-llms-dettmers-2023.pdf](./qlora-efficient-finetuning-quantized-llms-dettmers-2023.pdf)
- **Citation Count:** 4,500+ (2026년 기준, Google Scholar)

### 요약
QLoRA는 4-bit로 양자화된 베이스 모델(frozen) 위에 LoRA 어댑터를 얹어 역전파하는 방식으로, 65B 모델을 단일 48GB GPU 하나에서 파인튜닝 가능하게 만든 기법이다. 핵심은 (1) 정규분포에 최적화된 4-bit NormalFloat(NF4) 데이터 타입, (2) 양자화 상수까지 다시 양자화하는 Double Quantization, (3) 메모리 스파이크를 흡수하는 Paged Optimizer. 이를 통해 16-bit 풀 파인튜닝과 동등한 성능(Guanaco, Vicuna 벤치마크)을 달성했다.

### 핵심 기여
- 4-bit NF4: 가중치의 정규분포 특성에 맞게 정보이론적으로 최적화한 데이터 타입
- Double Quantization으로 양자화 상수 메모리를 추가로 0.37 bit/param 절감
- NVIDIA Unified Memory를 활용한 Paged Optimizers로 long-seq OOM 방지
- 65B 모델 파인튜닝 메모리를 780GB → 48GB 미만으로 축소, 16-bit 대비 성능 유지 입증

### 이 논문이 중요한 이유
QLoRA는 "GPU 한 장으로 수백억 파라미터 모델을 튜닝"이라는 시대를 열었고, 이후 bitsandbytes·Hugging Face PEFT의 기본 경로가 되었다. AI 엔지니어가 현실의 하드웨어 제약에서 LLM 커스터마이징을 설계할 때 가장 먼저 고려하는 레퍼런스이며, QLoRA의 NF4·Paged Optimizer는 이후 모든 4-bit 파인튜닝 스택의 기반이 된다.

### 사전 지식
- Post-training vs. quantization-aware training의 차이
- INT8, FP16, BF16, FP4 등 저정밀 데이터 타입
- LoRA의 저차원 분해 원리 (Paper 1)
- GPU 메모리 구조: weights, activations, gradients, optimizer states

### 관련 논문
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)

### 실무 적용
B2B SaaS의 고객사별 파인튜닝 상품화에서 단일 GPU 서버로 13B~70B 모델 커스터마이징이 가능해져 파인튜닝 단가가 극적으로 떨어졌다. Agentic AI에서 툴 사용·도메인 특화 에이전트를 테넌트별로 학습하고 4-bit 추론 서버(vLLM, TGI, llama.cpp)에 얹는 워크플로우는 QLoRA가 표준 경로다. ROI 관점에서는 "풀 파인튜닝 대비 5~10배 저비용으로 동등 품질" 벤치마크의 근거 논문.

---

## Paper 3 (Recent): Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities
- **Authors:** Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, Dacheng Tao
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2408.07666
- **PDF:** [./model-merging-llms-survey-yang-2024.pdf](./model-merging-llms-survey-yang-2024.pdf)
- **Citation Count:** 200+ (2026년 기준, Google Scholar — survey 특성상 빠르게 상승 중)

### 요약
이 서베이는 여러 개의 파인튜닝된 모델을 추가 학습 없이(또는 최소 학습으로) 하나로 결합하는 "모델 머징(model merging)" 기법을 체계적으로 정리한 최신 논문이다. Pre-merging(가중치 정렬), During-merging(가중치 결합: Task Arithmetic, TIES, DARE, Model Soups 등), Theories & Analysis의 3축 분류 체계를 제시하고, LLM·MLLM·Continual Learning·Multi-task·Federated 시나리오별 응용 사례와 한계를 정리한다.

### 핵심 기여
- 모델 머징 기법을 Pre / During / Post의 3단계로 통합 분류하는 새로운 택소노미 제안
- Task Arithmetic, TIES-Merging, DARE, Model Soups, SLERP, Fisher Merging 등 주요 알고리즘을 수식 수준에서 비교
- LLM alignment, multilingual LLM, MoE 구성, MLLM 통합, 연속학습 등 실제 응용 시나리오 분석
- 학습 없이 기능 조합이 가능한 "No-training fine-tuning" 패러다임의 이론적 기반 정리

### 이 논문이 중요한 이유
2024~2025년 오픈소스 LLM 생태계(HF Open LLM Leaderboard 상위 모델 상당수, Mergekit 활용 모델, MoE 구성)의 성능 향상을 설명하는 가장 압축된 레퍼런스다. 파인튜닝 없이 "좋은 체크포인트들을 합쳐서 더 좋은 모델을 만드는" 이 접근은 AI Engineer/CPO 관점에서 보면 모델 출시 주기·비용 구조 자체를 바꾸는 전략적 무기이며, 필수로 최신 동향을 파악해야 한다.

### 사전 지식
- LoRA/Full fine-tuning과 delta parameter(δ = W_ft − W_pt) 개념
- Loss landscape와 linear mode connectivity
- Multi-task learning의 gradient/task interference 문제

### 관련 논문
- [Editing Models with Task Arithmetic (Ilharco et al., 2022)](https://arxiv.org/abs/2212.04089)
- [TIES-Merging: Resolving Interference When Merging Models (Yadav et al., 2023)](https://arxiv.org/abs/2306.01708)
- [Language Models are Super Mario: DARE (Yu et al., 2023)](https://arxiv.org/abs/2311.03099)
- [Model Soups: Averaging Weights of Multiple Fine-tuned Models (Wortsman et al., 2022)](https://arxiv.org/abs/2203.05482)

### 실무 적용
기능 분야별(코딩, 다국어, 안전성)로 각각 튜닝된 LoRA/풀 파인튜닝 결과를 Mergekit(TIES+DARE)로 합쳐 "훈련 비용 0"으로 범용 제품 모델을 구성할 수 있다. Agentic AI에서 툴 사용·RAG·함수 호출을 각각 튜닝한 뒤 머지하여 릴리스 후보를 빠르게 A/B 테스트하는 실험 파이프라인을 구축할 수 있다. 모델 버전 관리(체크포인트 슈퍼포지션) 관점에서도 Git-LFS + Mergekit 조합이 새로운 ML Ops 패턴으로 떠오르는 중.

---

## 추천 읽기 순서
1. **LoRA** → 먼저 "왜 저차원 분해가 통하는가"의 직관과 기본 수식(ΔW = BA)을 확실히 잡는다. 본 커리큘럼 Day 17의 모든 하류 기법의 뿌리.
2. **QLoRA** → 그 위에 4-bit 양자화(NF4, Double Quant, Paged Optimizer)가 어떻게 결합되어 "단일 GPU 65B 튜닝"을 가능하게 했는지 따라간다. 여기서 실전 가능성이 열린다.
3. **Model Merging Survey** → 마지막으로 "학습하지 않고 모델을 합친다"는 새로운 축을 잡는다. LoRA·QLoRA와 수직 관계에 있는 2024~2025 최신 패러다임.

## 핵심 테이크어웨이
- 거대 모델의 효율화는 이제 "quantization × low-rank adaptation × merging" 3축의 곱셈 게임이다. 어느 한 축만 알면 최신 오픈소스 스택을 이해할 수 없다.
- 4-bit NF4 + LoRA(QLoRA)는 사실상 현재 산업 표준 파인튜닝 레시피이며, 대부분의 B2B LLM 커스터마이징 ROI 계산의 출발점이다.
- 모델 머징은 학습 비용을 0에 가깝게 만드는 "기능 합성(feature composition)" 관점의 기법으로, 프로덕트 팀이 릴리스 속도를 근본적으로 바꿀 수 있는 레버.
- 세 기법 모두 공통적으로 "원본 모델의 대부분은 건드리지 않고, 작은 변화(delta)만으로 큰 효과를 낸다"는 철학을 공유한다.

## 다음 토픽과의 연결
내일(Day 18)은 Module 7의 시작인 **"Chain-of-Thought and Few-Shot Prompting"**이다. 오늘까지 Module 6에서 "모델 자체를 어떻게 효율적으로 만들고 정렬(align)하는가"를 다뤘다면, 내일부터는 "이미 잘 만들어진 모델을 어떻게 프롬프트만으로 더 잘 쓰는가"의 영역으로 넘어간다. 특히 LoRA/QLoRA로 튜닝할지, 프롬프트 엔지니어링으로 해결할지, 아니면 모델 머징으로 조합할지 — 이 세 가지 옵션을 놓고 문제별로 비용·성능을 비교 판단하는 능력이 CPO/AI 엔지니어의 핵심 역량이다.
