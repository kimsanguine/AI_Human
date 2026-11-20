# Daily AI Paper Recommendations

> **Date:** 2026-11-21
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Efficient LLM Quantization and Distillation

> 이번 사이클에서는 이전에 다룬 논문(LoRA, QLoRA, Hinton KD, GPTQ, BitNet b1.58, LLM.int8(), AWQ, QuaRot, SmoothQuant, DistilBERT, Deep Compression, Integer-only Inference, AQLM, k-bit Scaling Laws, MiniLLM, Sequence-Level KD, AdaRound, KIVI, SparseGPT, TinyBERT, Minitron 등)과 겹치지 않도록 **"출력만이 아니라 중간 표현도 가르칠 수 있는가(FitNets)" → "양자화와 증류를 한 파이프라인으로 묶을 수 있는가(ZeroQuant)" → "2비트까지 내려가도 쓸 만한 모델을 만들 수 있는가(QuIP#)"** 라는 흐름으로 골랐습니다. 오늘의 질문은 하나입니다: **"모델을 작게 만들 때, 우리는 정확히 무엇을 보존하려는 것인가?"**

---

## Paper 1 (Classic): FitNets: Hints for Thin Deep Nets
- **Authors:** Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio
- **Year:** 2014 (ICLR 2015)
- **arXiv:** https://arxiv.org/abs/1412.6550
- **PDF:** [./fitnets-romero-2014.pdf](./fitnets-romero-2014.pdf)
- **Citation Count:** ~4,500+

### 요약
Hinton의 지식 증류(KD)는 교사 모델의 "최종 출력(soft label)"만 학생에게 가르쳤습니다. FitNets는 한 걸음 더 나아가 교사의 **중간 레이어 표현(hint)** 을 학생의 중간 레이어(guided layer)가 따라 하도록 먼저 학습시킨 뒤, 그 다음에 KD로 전체를 학습하는 2단계 방식을 제안합니다. 그 결과 교사보다 **더 얇고(폭이 좁고) 더 깊은** 학생 네트워크가 파라미터는 훨씬 적으면서 교사와 비슷하거나 더 좋은 성능을 냈습니다.

### 핵심 기여
- **Hint 기반 학습:** 교사의 중간 특징 맵을 회귀(regressor)로 맞추는 손실을 도입해, "답"뿐 아니라 "생각의 중간 과정"을 전달
- **얇고 깊은 학생 구조:** 폭을 줄이고 깊이를 늘린 학생이 깊이 때문에 생기는 학습 난이도를 hint로 극복할 수 있음을 증명
- **2단계 학습 절차:** (1) hint로 하위 레이어 초기화 → (2) KD로 전체 미세조정. 이후 수많은 feature-based KD 연구의 출발점
- CIFAR-10/100, SVHN, MNIST에서 약 10배 적은 파라미터로 교사 수준 이상의 성능

### 이 논문이 중요한 이유
오늘날 LLM 증류(Minitron, DistilBERT, TinyBERT 등)에서 쓰는 "히든 스테이트 정렬", "어텐션 맵 정렬" 손실의 원형이 바로 FitNets입니다. "학생에게 무엇을 보여줄 것인가(출력 vs 중간 표현)"라는 증류 설계의 핵심 질문을 처음 명확히 던진 논문이라, 증류 파이프라인을 설계하는 AI 엔지니어라면 반드시 알아야 합니다.

### 사전 지식
- Hinton의 Knowledge Distillation (온도 T, soft target) — 이전 사이클 Day 17
- CNN의 중간 특징 맵(feature map) 개념
- 깊은 네트워크 학습의 어려움(기울기 소실)과 좋은 초기화의 중요성

### 관련 논문
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Paying More Attention to Attention: Attention Transfer (Zagoruyko & Komodakis, 2016)](https://arxiv.org/abs/1612.03928)
- [TinyBERT: Distilling BERT for Natural Language Understanding (Jiao et al., 2019)](https://arxiv.org/abs/1909.10351)
- [Do Deep Nets Really Need to be Deep? (Ba & Caruana, 2013)](https://arxiv.org/abs/1312.6184)

### 실무 적용
사내 LLM을 작은 모델로 증류할 때, 교사의 최종 로짓만 쓰면 학생이 "답의 형식"은 흉내 내도 내부 추론은 따라가지 못하는 경우가 많습니다. 교사·학생의 히든 차원이 다르면 FitNets처럼 작은 선형 projection을 두고 중간 레이어 히든 스테이트를 MSE로 맞추는 손실을 추가해 보세요. 특히 같은 계열 모델(예: 큰 모델 → 레이어를 줄인 작은 모델)로 증류할 때 수렴 속도와 최종 품질이 함께 좋아지는 경우가 많습니다.

---

## Paper 2 (Classic): ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
- **Authors:** Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He
- **Year:** 2022 (NeurIPS 2022)
- **arXiv:** https://arxiv.org/abs/2206.01861
- **PDF:** [./zeroquant-yao-2022.pdf](./zeroquant-yao-2022.pdf)
- **Citation Count:** ~700+

### 요약
GPT-3급 대형 모델을 INT8로 양자화하려면 보통 재학습(QAT)이 필요했지만, 이는 비용이 너무 큽니다. ZeroQuant는 Microsoft DeepSpeed 팀이 만든 **학습 없는(post-training) 엔드투엔드 양자화 파이프라인**으로, (1) 가중치는 그룹 단위·활성값은 토큰 단위로 세밀하게 양자화하고, (2) 원본 데이터 없이도 레이어 하나씩 증류하는 **LKD(Layer-by-layer Knowledge Distillation)** 로 정확도를 회복하며, (3) 이를 실제로 빠르게 돌리는 시스템 커널까지 함께 제공합니다.

### 핵심 기여
- **세밀한 양자화 단위:** 가중치는 group-wise, 활성값은 token-wise 동적 양자화로 이상치(outlier)의 영향을 국소화
- **LKD (레이어 단위 증류):** 전체 모델을 한 번에 증류하지 않고 레이어 하나씩 원본 레이어의 출력을 따라 하게 해, GPU 메모리가 적어도 대형 모델을 INT4/INT8로 복구 가능
- **시스템 공동 설계:** 양자화/역양자화 연산을 앞뒤 연산과 융합한 커널로 BERT·GPT 계열에서 FP16 대비 최대 5.2배 추론 가속
- 양자화와 증류가 "별개 기법"이 아니라 **하나의 압축 파이프라인 안에서 결합**될 수 있음을 보여줌

### 이 논문이 중요한 이유
GPTQ, SmoothQuant, AWQ로 이어지는 LLM PTQ 흐름의 초기 이정표이며, 이후 ZeroQuant-V2/FP/HERO 시리즈와 DeepSpeed-Inference의 기반이 됐습니다. 특히 "알고리즘만 좋다고 빨라지지 않는다, 커널까지 설계해야 실제 속도가 난다"는 메시지는 GGUF·vLLM·TensorRT-LLM 등 서빙 스택을 고르는 엔지니어에게 중요한 관점입니다.

### 사전 지식
- 양자화 기본: 스케일(scale), zero-point, 대칭/비대칭, per-tensor vs per-channel
- PTQ(Post-Training Quantization)와 QAT(Quantization-Aware Training)의 차이
- 활성값 이상치 문제 (LLM.int8(), SmoothQuant — 이전 사이클)

### 관련 논문
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438)
- [ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation (Yao et al., 2023)](https://arxiv.org/abs/2303.08302)
- [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction (Li et al., 2021)](https://arxiv.org/abs/2102.05426)

### 실무 적용
자체 모델을 INT8/INT4로 서빙할 때 "양자화 후 정확도가 떨어졌다"면 곧바로 전체 재학습을 하기보다, ZeroQuant의 LKD처럼 **레이어(또는 블록) 단위로 원본 출력과의 오차를 줄이는 보정 단계**를 먼저 시도해 보세요. 소량의 대표 프롬프트만 있으면 되고, 단일 GPU로도 가능합니다. 또한 양자화 방식을 고를 때는 정확도 표만 보지 말고, 실제 사용하는 서빙 엔진이 해당 포맷의 융합 커널을 지원하는지(실제 TPS·지연시간)를 함께 측정해야 합니다.

---

## Paper 3 (Recent): QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- **Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **Year:** 2024 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2402.04396
- **PDF:** [./quip-sharp-tseng-2024.pdf](./quip-sharp-tseng-2024.pdf)
- **Citation Count:** ~300+

### 요약
가중치를 2~4비트로 극단적으로 줄이면 대부분의 기법은 품질이 급격히 무너집니다. QuIP#은 세 가지를 결합해 이를 돌파합니다: (1) **랜덤 Hadamard 변환**으로 가중치의 이상치를 고르게 퍼뜨리는 "비일관성(incoherence) 처리", (2) 숫자를 하나씩 반올림하는 대신 8개씩 묶어 **E8 격자(lattice) 코드북**의 가장 가까운 점으로 양자화하는 벡터 양자화, (3) 양자화 후 남은 연속 파라미터를 미세조정하는 단계. 그 결과 Llama 2 70B를 **2비트**로 줄여도 실용적인 품질을 유지하는 첫 PTQ 방법 중 하나가 됐습니다.

### 핵심 기여
- **Hadamard 비일관성 처리:** 기존 QuIP의 Kronecker 직교 행렬보다 빠르고 이론적 보장이 더 좋은 랜덤 Hadamard 변환 사용 (QuaRot·SpinQuant 등 회전 기반 기법과 같은 계보)
- **E8P 격자 코드북:** 8차원에서 가장 촘촘한 구 채우기(sphere packing)인 E8 격자를 활용, 코드북이 작아 캐시에 올라가므로 디코딩이 빠름
- **양자화 후 미세조정:** 블록 단위로 원본 출력에 맞추는 파인튜닝으로 2비트 품질을 추가 회복
- 2비트에서 기존 PTQ 대비 큰 폭의 perplexity 개선, 3비트 QuIP# 모델이 4비트 기존 모델보다 더 좋은 "비트당 효율" 달성

### 이 논문이 중요한 이유
GGUF의 IQ2/IQ3 계열, AQLM, QTIP 등 최근 "2~3비트 실사용" 흐름의 핵심 아이디어(회전 + 벡터 양자화 + 미세조정)를 가장 깔끔하게 정리한 논문입니다. "같은 메모리 예산이라면 작은 모델을 4비트로 쓸까, 큰 모델을 2비트로 쓸까?"라는 실무 질문에 데이터로 답할 수 있게 해 주며, 온디바이스·저비용 GPU 서빙을 고민하는 팀에게 특히 중요합니다.

### 사전 지식
- 스칼라 양자화 vs 벡터 양자화(VQ)의 차이, 코드북 개념
- 직교 변환(회전)이 이상치를 분산시키는 원리 (QuaRot, SpinQuant — 이전 사이클)
- GPTQ의 2차 정보(Hessian) 기반 오차 보정 아이디어

### 관련 논문
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees (Chee et al., 2023)](https://arxiv.org/abs/2307.13304)
- [Extreme Compression of Large Language Models via Additive Quantization / AQLM (Egiazarian et al., 2024)](https://arxiv.org/abs/2401.06118)
- [QTIP: Quantization with Trellises and Incoherence Processing (Tseng et al., 2024)](https://arxiv.org/abs/2406.11235)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Ashkboos et al., 2024)](https://arxiv.org/abs/2404.00456)

### 실무 적용
24GB 단일 GPU나 Mac 같은 제한된 환경에서 70B급 모델을 돌려야 한다면, "8B를 FP16으로"와 "70B를 2~3비트로"를 같은 평가셋으로 직접 비교해 보세요. QuIP#·AQLM·GGUF IQ 포맷은 이 선택지를 현실로 만들었습니다. 단, 2비트 모델은 perplexity보다 **긴 추론·코드·수학 과제에서 먼저 성능이 무너지는** 경향이 있으므로, 어제(Day 16) 다룬 실행형 벤치마크(SWE-bench 류)나 자사 도메인 과제로 검증한 뒤 도입해야 합니다.

---

## 추천 읽기 순서
1. **FitNets** — 13쪽으로 짧고 직관적. "출력만이 아니라 중간 표현도 가르친다"는 증류의 핵심 아이디어를 먼저 잡기
2. **ZeroQuant** — 양자화(group/token-wise)와 증류(LKD)가 한 파이프라인에서 어떻게 만나는지, 그리고 커널 설계가 왜 필요한지 확인
3. **QuIP#** — 회전 + 벡터 양자화 + 미세조정으로 2비트까지 내려가는 최신 기법. 수식은 E8 격자 부분을 가볍게 넘기고 3절(Incoherence)과 실험 결과 위주로 읽기

## 핵심 테이크어웨이
- **Q. 모델을 작게 만들 때 무엇을 보존해야 하나?** 최종 출력만이 아니라 중간 표현(FitNets), 레이어별 출력(ZeroQuant LKD), 블록 출력(QuIP# 미세조정)처럼 "원본과 같은 계산 과정"을 얼마나 지키느냐가 품질을 좌우한다.
- **Q. 양자화와 증류는 다른 기법인가?** 아니다. 최신 압축은 "양자화로 줄이고, 증류(원본 출력 맞추기)로 복구"하는 하나의 파이프라인이다.
- **Q. 비트를 줄이면 무엇이 문제인가?** 이상치(outlier). 그룹 단위 스케일, 회전(Hadamard), 벡터 양자화는 모두 이상치를 다루는 서로 다른 해법이다.
- **Q. 정확도만 보면 되나?** 아니다. 실제 서빙 엔진에서 해당 포맷의 커널이 있어야 속도가 나고, 저비트 모델은 장문 추론·코드에서 먼저 무너지므로 과제 기반 평가가 필수다.

## 다음 토픽과의 연결
다음 토픽은 **Module 7: Prompt Engineering — Chain-of-Thought and Few-Shot Prompting**입니다. 경량화된 작은 모델일수록 프롬프트 설계(예시 선택, 단계별 추론 유도)에 따라 성능 편차가 커집니다. 또한 최근 증류는 큰 모델의 Chain-of-Thought 추론 과정을 작은 모델에 옮기는 방향(추론 증류)으로 발전하고 있어, 오늘 배운 "중간 과정을 가르친다(FitNets)"는 아이디어가 프롬프트 수준에서 CoT로 다시 등장한다고 볼 수 있습니다.
