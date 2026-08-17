# Daily AI Paper Recommendations

> **Date:** 2026-08-18
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- **Authors:** Mingxing Tan, Quoc V. Le
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1905.11946
- **PDF:** [./efficientnet-tan-2019.pdf](./efficientnet-tan-2019.pdf)
- **Citation Count:** ~30,000+ (approximate)

### 요약
CNN의 성능을 높이기 위해 depth(층 수), width(채널 수), resolution(입력 해상도)을 개별적으로 키우던 관행에 대해, 세 축을 고정된 비율로 동시에 확장하는 "compound scaling" 법칙을 제안한 논문이다. 여기에 NAS로 찾은 효율적인 베이스 모델 EfficientNet-B0를 결합해, EfficientNet-B7이 ImageNet에서 84.3% top-1을 달성하면서도 당시 SOTA 대비 파라미터는 8.4배, 추론 연산은 6.1배 적었다. "더 큰 모델"이 아니라 "균형 잡힌 모델"이 정확도/비용 곡선을 지배한다는 것을 실증했다.

### 핵심 기여
- Depth·Width·Resolution을 φ 하나의 계수로 동시에 확장하는 compound scaling 법칙 정식화
- Mobile inverted bottleneck(MBConv) + Squeeze-and-Excitation 기반의 NAS 베이스 모델 EfficientNet-B0 설계
- B0~B7 패밀리로 정확도-FLOPs 파레토 프론티어를 크게 개선하고, 전이학습 8개 데이터셋에서도 SOTA 검증

### 이 논문이 중요한 이유
AI 엔지니어가 실제로 부딪히는 문제는 "정확도를 얼마나 올릴 수 있나"가 아니라 "주어진 지연시간/메모리/비용 예산 안에서 최대 정확도를 어떻게 뽑나"이다. EfficientNet은 이 트레이드오프를 임의의 튜닝이 아니라 하나의 스케일링 법칙으로 다룰 수 있음을 보여줬고, 이 사고방식이 이후 LLM 스케일링 법칙 논의로도 이어진다. 모델 패밀리를 B0~B7처럼 등급별로 제공하는 제품 설계 패턴(엣지/서버 티어 분리)의 원형이기도 하다.

### 사전 지식
- CNN 기본 구조(합성곱, 풀링, 잔차 연결)와 ResNet 계열 아키텍처
- Depthwise separable convolution과 MobileNet의 inverted residual 블록
- FLOPs·파라미터 수와 실제 지연시간의 차이, ImageNet 벤치마크 읽는 법
- NAS(Neural Architecture Search)의 개념 수준 이해

### 관련 논문
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)
- [Squeeze-and-Excitation Networks (Hu et al., 2017)](https://arxiv.org/abs/1709.01507)
- [EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)](https://arxiv.org/abs/2104.00298)

### 실무 적용
온디바이스 비전(모바일 앱 이미지 분류, 실시간 카메라 필터), 대량 배치 이미지 태깅 파이프라인처럼 GPU 비용이 곧 마진인 서비스에서 여전히 기본 백본으로 쓰인다. 실무에서는 "정확도 1%p를 위해 추론 비용 3배"를 감수할지 판단할 때 B0~B7 곡선을 그대로 의사결정 근거로 활용할 수 있다.

---

## Paper 2 (Classic): Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- **Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2103.14030
- **PDF:** [./swin-transformer-liu-2021.pdf](./swin-transformer-liu-2021.pdf)
- **Citation Count:** ~30,000+ (approximate)

### 요약
ViT는 이미지 전체를 하나의 해상도에서 flat하게 처리해 detection·segmentation 같은 dense prediction에 약했다. Swin Transformer는 self-attention을 로컬 윈도우 안으로 제한해 계산량을 이미지 크기에 대해 제곱이 아닌 선형으로 낮추고, 레이어마다 윈도우를 절반씩 이동시키는 shifted window로 윈도우 간 정보 교환을 확보했다. 여기에 CNN처럼 단계별로 해상도를 낮추는 계층 구조를 더해, 분류뿐 아니라 COCO detection·ADE20K segmentation에서 모두 SOTA를 달성한 범용 비전 백본이 되었다.

### 핵심 기여
- Window-based Multi-head Self-Attention(W-MSA)으로 attention 복잡도를 O(N²)에서 O(N)으로 감소
- Shifted Window(SW-MSA)로 윈도우 경계를 넘는 연결성을 추가 비용 거의 없이 확보
- 4-stage 계층적 피라미드 구조로 FPN·U-Net 계열 헤드에 그대로 꽂아 쓸 수 있는 범용 백본 정립

### 이 논문이 중요한 이유
"Transformer가 CNN을 대체할 것인가"라는 질문에, "대체가 아니라 CNN의 귀납 편향(지역성·다중 스케일)을 Transformer 안으로 가져오면 된다"는 실용적 답을 제시했다. 오늘날 SAM, DINOv2/v3, 대부분의 멀티모달 LLM 비전 인코더가 이 계층적·효율적 attention 설계 계보 위에 있다. 아키텍처 설계에서 "무엇을 버리고 무엇을 유지할지"를 판단하는 좋은 사례 연구다.

### 사전 지식
- Self-attention과 Transformer 인코더 구조
- ViT의 패치 임베딩 방식과 그 한계
- Object detection/semantic segmentation의 기본 파이프라인(FPN, Mask R-CNN 등)
- Attention 연산의 계산 복잡도 분석

### 관련 논문
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale / ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [A ConvNet for the 2020s / ConvNeXt (Liu et al., 2022)](https://arxiv.org/abs/2201.03545)
- [Swin Transformer V2: Scaling Up Capacity and Resolution (Liu et al., 2021)](https://arxiv.org/abs/2111.09883)

### 실무 적용
의료 영상 분할, 위성/항공 이미지 분석, 문서 레이아웃 파싱처럼 고해상도 입력에서 픽셀 단위 예측이 필요한 도메인의 사실상 표준 백본이다. 실무에서는 Swin 백본 + task-specific head 조합으로 사전학습 가중치를 재사용해 라벨이 적은 도메인에서도 빠르게 baseline을 세울 수 있다.

---

## Paper 3 (Recent): Multimodal Autoregressive Pre-training of Large Vision Encoders (AIMv2)
- **Authors:** Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, Alaaeldin El-Nouby (Apple)
- **Year:** 2024 (CVPR 2025)
- **arXiv:** https://arxiv.org/abs/2411.14402
- **PDF:** [./aimv2-multimodal-autoregressive-pretraining-fini-2024.pdf](./aimv2-multimodal-autoregressive-pretraining-fini-2024.pdf)
- **Citation Count:** ~200+ (approximate, 빠르게 증가 중)

### 요약
CLIP/SigLIP 계열의 대조학습(contrastive learning) 대신, 이미지 패치와 텍스트 토큰을 **하나의 autoregressive 목적함수로 함께 생성**하도록 비전 인코더를 사전학습한 연구다. 인코더가 뽑은 표현을 causal multimodal decoder에 넣어 이미지 패치를 회귀하고 이어서 텍스트 토큰을 예측하게 하며, 이 단순한 레시피가 대규모 배치나 복잡한 배치 간 통신 없이도 안정적으로 스케일된다. AIMv2-3B는 frozen trunk 상태로 ImageNet-1k 89.5%를 기록했고, multimodal LLM의 비전 인코더로 붙였을 때 OCR·grounding 등에서 CLIP/SigLIP을 앞섰다.

### 핵심 기여
- 대조학습 없이 순수 autoregressive 손실만으로 학습된 대규모 범용 비전 인코더 계열(AIMv2) 제안
- 이미지 패치 → 텍스트 토큰 순으로 생성하는 causal multimodal decoder 설계로 밀집(dense) 감독 신호 확보
- 거대 배치·특수 인프라 의존성을 제거해 학습 레시피를 단순화하고, 분류·localization·grounding·MLLM 통합까지 폭넓게 검증

### 이 논문이 중요한 이유
멀티모달 LLM 시대에 "비전 인코더를 어떻게 학습시킬 것인가"는 제품 품질(특히 OCR·문서 이해·UI 인식)에 직접 연결되는 문제다. AIMv2는 LLM에서 이미 검증된 next-token prediction 패러다임을 비전 인코더에 그대로 이식할 수 있음을 보여, 텍스트와 비전의 학습 스택을 통일하는 방향을 제시했다. AI 엔지니어 입장에서는 학습 인프라 복잡도를 낮추면서 성능을 얻는 실용적 선택지의 근거가 된다.

### 사전 지식
- CLIP/SigLIP 등 대조학습 기반 이미지-텍스트 정렬의 원리와 배치 크기 의존성
- Autoregressive language modeling(next-token prediction)과 causal attention mask
- ViT 기반 인코더와 frozen trunk / linear probing 평가 방식
- Multimodal LLM(예: LLaVA 계열)에서 비전 인코더가 붙는 위치

### 관련 논문
- [Learning Transferable Visual Models From Natural Language Supervision / CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training / SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [Scalable Pre-training of Large Autoregressive Image Models / AIM (El-Nouby et al., 2024)](https://arxiv.org/abs/2401.08541)

### 실무 적용
문서 이해·영수증 OCR·UI 스크린샷 에이전트처럼 텍스트가 섞인 이미지를 다루는 제품에서 비전 인코더 교체만으로 체감 품질이 크게 달라진다. 자체 멀티모달 제품을 만들 때 "CLIP 계열을 그대로 쓸지, autoregressive 사전학습 인코더로 갈아탈지"를 결정하는 벤치마크 근거로 쓸 수 있고, 대규모 GPU 클러스터 없이도 학습 파이프라인을 재현하기 쉽다는 점이 스타트업 관점에서 특히 중요하다.

---

## 추천 읽기 순서

1. **EfficientNet (2019)** — 먼저 읽는다. "정확도 vs 비용"이라는 축을 세워야 나머지 두 논문의 설계 선택이 왜 그렇게 됐는지 보인다. 4~5장의 compound scaling 수식과 Table 2 파레토 곡선에 집중.
2. **Swin Transformer (2021)** — 다음. CNN의 계층 구조를 Transformer로 옮기는 과정을 보며, EfficientNet이 다룬 효율성 문제를 attention에서 어떻게 다시 푸는지 대조하며 읽는다. Figure 1~3(윈도우 이동 메커니즘)이 핵심.
3. **AIMv2 (2024)** — 마지막. 아키텍처가 아니라 **사전학습 목적함수**가 주인공으로 바뀐 시점을 확인한다. 표현 학습의 무게중심이 어디로 이동했는지 파악.

시간이 부족하면: EfficientNet 4장 → Swin 3장 → AIMv2 3장 + 실험 테이블만 봐도 흐름은 잡힌다.

## 핵심 테이크어웨이

- **아키텍처 개선의 본질은 파레토 프론티어 이동이다.** EfficientNet은 새로운 연산자를 발명한 게 아니라, 기존 축들을 균형 있게 확장하는 법칙을 찾아 곡선 전체를 끌어올렸다. 제품 관점에서 의미 있는 개선은 "더 정확한 모델"이 아니라 "같은 비용에서 더 정확한 모델"이다.
- **귀납 편향은 버리는 게 아니라 옮기는 것이다.** ViT가 CNN의 지역성·다중 스케일을 버렸다가 dense prediction에서 대가를 치렀고, Swin은 그것을 attention 구조 안으로 다시 들여와 해결했다. 새 패러다임이 나올 때 기존 설계의 어떤 부분이 "관행"이고 어떤 부분이 "본질"인지 구분하는 훈련이 필요하다.
- **경쟁의 축이 아키텍처에서 학습 목적함수·데이터로 이동했다.** 2019→2021은 블록 설계 싸움이었지만, 2024~2025의 AIMv2·DINOv3·SigLIP2는 모두 "무엇을 예측하게 할 것인가"로 승부한다. 백본은 ViT로 수렴했고 차별화는 pre-training objective에서 나온다.
- **frozen trunk 성능이 실무 지표다.** AIMv2가 frozen 상태 89.5%를 강조하는 이유는, 실제 제품에서는 인코더를 재학습하지 않고 얼려둔 채 여러 downstream 헤드를 붙이기 때문이다. 파인튜닝 후 최고 성능보다 이 숫자가 배포 현실에 가깝다.

## 다음 토픽과의 연결

다음 토픽인 **RNN·LSTM과 시퀀스 모델**로 넘어가면, 오늘 본 "공간(spatial) 구조를 어떻게 모델에 넣을 것인가"라는 질문이 "시간(temporal) 구조를 어떻게 넣을 것인가"로 바뀐다. Swin의 로컬 윈도우 attention은 시퀀스 쪽의 sliding-window attention·state space model(Mamba)과 정확히 같은 문제의식 — 긴 입력에서 제곱 복잡도를 어떻게 피할 것인가 — 을 공유한다. 두 흐름이 결국 어디서 만나는지는 Module 4의 Attention/Transformer 편에서 이어진다.
