# Daily AI Paper Recommendations

> **Date:** 2026-09-14
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Squeeze-and-Excitation Networks
- **Authors:** Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
- **Year:** 2017 (arXiv) / 2018 (CVPR)
- **arXiv:** https://arxiv.org/abs/1709.01507
- **PDF:** [./squeeze-and-excitation-networks-hu-2018.pdf](./squeeze-and-excitation-networks-hu-2018.pdf)
- **Citation Count:** 약 35,000회 이상

### 요약
CNN이 공간(spatial) 차원의 특징 추출에는 많은 발전을 이뤘지만, 채널(channel) 간의 상호 의존성은 거의 다뤄지지 않았다는 문제의식에서 출발한 논문이다. 저자들은 채널별 특징 응답을 명시적으로 재보정(recalibration)하는 Squeeze-and-Excitation(SE) 블록을 제안했다. SE-ResNet 기반 모델로 ILSVRC 2017 분류 부문에서 1위를 차지하며 top-5 error를 2.251%까지 낮췄다.

### 핵심 기여
- **Squeeze 연산:** Global Average Pooling으로 각 채널의 공간 정보를 하나의 스칼라로 압축하여, 얕은 층에서도 전역(global) 수용 영역을 확보
- **Excitation 연산:** 2개의 FC 레이어 + 시그모이드로 채널별 가중치를 학습하고, 원본 feature map에 곱해 중요한 채널을 강조하고 불필요한 채널을 억제
- **아키텍처 무관 플러그인:** ResNet, Inception, ResNeXt 등 기존 어떤 백본에도 삽입 가능하며, 연산량은 1% 미만 증가시키면서 정확도를 유의미하게 개선

### 이 논문이 중요한 이유
"어텐션(attention)"이라는 개념이 CNN 안으로 들어온 결정적인 시점을 만든 논문이다. Transformer의 self-attention이 토큰 간 관계를 다룬다면, SE 블록은 채널 간 관계를 다루는 가장 단순하고 효율적인 형태의 어텐션이다. 이후 CBAM, ECA-Net, EfficientNet, MobileNetV3 등 대부분의 현대 효율 아키텍처가 SE 블록 또는 그 변형을 내장하고 있어, AI 엔지니어라면 코드 레벨에서 반드시 이해하고 있어야 하는 모듈이다. 또한 "10줄짜리 모듈 추가로 SOTA를 갱신한다"는, 비용 대비 효과가 극단적으로 높은 아키텍처 설계 사례로서 제품 관점에서도 배울 점이 크다.

### 사전 지식
- CNN의 기본 구조 (convolution, pooling, feature map의 채널 개념)
- ResNet의 residual block과 skip connection
- Global Average Pooling의 역할
- 어텐션 메커니즘의 기본 아이디어 (가중치를 학습해 중요한 것에 집중)

### 관련 논문
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [CBAM: Convolutional Block Attention Module (Woo et al., 2018)](https://arxiv.org/abs/1807.06521)
- [ECA-Net: Efficient Channel Attention for Deep CNNs (Wang et al., 2019)](https://arxiv.org/abs/1910.03151)
- [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)

### 실무 적용
SE 블록은 오늘날 온디바이스 비전 모델의 사실상 표준 부품이다. MobileNetV3, EfficientNet-Lite 계열이 스마트폰 카메라의 장면 인식, 실시간 세그멘테이션, AR 필터에 탑재될 때 SE 블록이 그대로 들어간다. 실무에서는 기존 백본의 성능이 아쉬울 때 "구조를 바꾸지 않고 SE 블록만 삽입해 재학습"하는 것이 가장 저렴한 첫 번째 개선 시도로 자주 쓰인다. 또한 SE의 채널 가중치를 시각화하면 모델이 어떤 특징 채널에 의존하는지 해석할 수 있어, 모델 디버깅과 데이터 편향 진단에도 활용된다.

---

## Paper 2 (Classic): MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1704.04861
- **PDF:** [./mobilenets-efficient-cnn-howard-2017.pdf](./mobilenets-efficient-cnn-howard-2017.pdf)
- **Citation Count:** 약 30,000회 이상

### 요약
모바일 및 임베디드 환경에서 동작 가능한 경량 CNN 아키텍처를 제안한 논문이다. 핵심은 표준 컨볼루션을 depthwise convolution과 1×1 pointwise convolution으로 분해하는 depthwise separable convolution이다. 여기에 width multiplier와 resolution multiplier라는 두 개의 하이퍼파라미터를 도입해, 개발자가 지연시간(latency)과 정확도 사이를 직접 조절할 수 있게 했다.

### 핵심 기여
- **Depthwise Separable Convolution:** 표준 컨볼루션 대비 연산량을 약 8~9배 줄이면서 정확도 손실은 1% 수준으로 억제
- **Width Multiplier (α):** 모든 레이어의 채널 수를 균일하게 축소해 모델 크기를 제곱 비례로 줄이는 단순한 스케일링 노브
- **Resolution Multiplier (ρ):** 입력 해상도를 조절해 연산량을 추가로 제어
- **트레이드오프의 명시적 설계:** "하나의 최적 모델"이 아니라 "제약 조건에 맞춰 고를 수 있는 모델 패밀리"라는 관점을 제시

### 이 논문이 중요한 이유
AI 엔지니어의 현실은 "가장 정확한 모델"이 아니라 "주어진 지연시간·메모리·배터리 예산 안에서 가장 정확한 모델"을 만드는 일이다. MobileNet은 그 문제를 정면으로 다룬 최초의 대표적 논문이며, 이후 MobileNetV2/V3, ShuffleNet, EfficientNet, MnasNet으로 이어지는 효율 아키텍처 계보의 출발점이다. 특히 depthwise separable convolution은 지금도 온디바이스 모델, 실시간 비디오 처리, 심지어 일부 오디오/음성 모델에서까지 기본 연산 단위로 쓰인다. 또한 "정확도-비용 곡선(Pareto front)에서 제품 요구사항에 맞는 점을 고른다"는 사고방식 자체가 PM/CPO 관점에서 매우 중요한 프레이밍이다.

### 사전 지식
- 표준 convolution의 연산량 계산 (D_K × D_K × M × N × D_F × D_F)
- 채널(channel), 필터(filter), 커널 크기의 관계
- FLOPs와 실제 지연시간(latency)이 항상 비례하지는 않는다는 점
- 모델 경량화의 다른 축들 (양자화, 프루닝, 지식 증류)과의 차이

### 관련 논문
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)
- [Searching for MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
- [ShuffleNet: An Extremely Efficient CNN for Mobile Devices (Zhang et al., 2017)](https://arxiv.org/abs/1707.01083)
- [Xception: Deep Learning with Depthwise Separable Convolutions (Chollet, 2016)](https://arxiv.org/abs/1610.02357)

### 실무 적용
MobileNet 계열은 TensorFlow Lite, Core ML, ONNX Runtime의 대표 벤치마크 모델로 자리잡았고, 스마트폰 갤러리 앱의 자동 태깅, 실시간 배경 제거(화상회의), 리테일 매장의 엣지 카메라 분석, 드론의 온보드 객체 탐지 등에 광범위하게 쓰인다. 실무 워크플로우에서는 MobileNet 백본으로 먼저 파이프라인을 빠르게 구성해 지연시간 예산을 검증한 뒤, 여유가 있으면 더 큰 백본으로 교체하는 방식이 일반적이다. width multiplier는 A/B 테스트로 "정확도 1%p를 얻기 위해 지연시간 몇 ms를 지불할 것인가"를 정량적으로 결정하는 도구가 된다.

---

## Paper 3 (Recent): Scaling Parallel Sequence Models to Foundation-Scale Vision Encoders (C-GSPN)
- **Authors:** Yitong Jiang, Hongjun Wang, Collin McCarthy, Hanrong Ye, David Wehr, Xinhao Li, Qi Dou, Tianfan Xue, Ka Chun Cheung, Simon See, Wonmin Byeon, Ke Chen, Kai Han, Jinwei Gu, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Sifei Liu (NVIDIA, CUHK, HKU, UCSD)
- **Year:** 2026
- **arXiv:** https://arxiv.org/abs/2606.00746
- **PDF:** [./c-gspn-foundation-scale-vision-encoders-jiang-2026.pdf](./c-gspn-foundation-scale-vision-encoders-jiang-2026.pdf)
- **Citation Count:** 신규 논문 (2026년 5월 공개), 인용 수 집계 초기 단계

### 요약
비전 파운데이션 모델은 self-attention의 이차(quadratic) 연산 비용 때문에 사용 가능한 해상도가 제한되고 대규모 사전학습 비용이 급증하는 병목을 겪는다. 선형 어텐션이나 SSM(Mamba 계열)은 비용을 낮추지만 이미지를 1D 토큰 열로 직렬화하면서 2D 공간 구조를 잃는다. 이 논문은 2D 격자 위에서 line-scan 재귀로 직접 컨텍스트를 전파하는 GSPN(Generalized Spatial Propagation Network)을 파운데이션 스케일로 끌어올린 C-GSPN을 제안한다.

### 핵심 기여
- **고속 GSPN 커널 (시스템 효율):** 수천 번의 개별 커널 실행을 warp-specialized 단일 CUDA 커널로 융합하고 shared-memory 타일링을 적용해, 이론적 메모리 대역폭의 90% 이상을 달성하고 기존 구현 대비 40~52배 빠른 속도 확보
- **효율적인 ViT 블록 (아키텍처 효율):** 압축된 잠재 공간(latent space)에서 전파를 수행하고 정규화를 융합해, 커널 수준의 속도 향상을 블록·모델 수준의 실질 속도로 전환
- **교차 연산자 증류 (학습 효율):** 새 연산자는 사전학습된 어텐션 가중치를 상속할 수 없다는 문제를, 서브레이어 정렬 → end-to-end two-tap feature distillation의 2단계 증류 레시피로 해결
- **성과:** 600M 이미지-텍스트 쌍으로 증류한 결과, 동형(isomorphic) ViT 베이스라인을 파라미터 15% 적게 쓰면서 따라잡고, ADE20K 세그멘테이션 +2.1%, 2K 해상도에서 타일링 없는 단일 패스 추론으로 4배 end-to-end 블록 속도 향상

### 이 논문이 중요한 이유
"어텐션을 무엇으로 대체할 것인가"라는 질문은 LLM에서는 Mamba/SSM 논쟁으로 이미 익숙하지만, 비전에서는 답이 달라야 한다는 점을 명확히 보여주는 논문이다. 이미지는 본질적으로 2D이고, 이를 1D로 펴는 순간 잃는 정보가 있다는 문제 제기와, 그 해법으로 2D 공간 전파를 제시한 점이 핵심이다. 더 중요한 실무적 기여는 **"새 아키텍처를 처음부터 학습하지 않고 기존 어텐션 모델로부터 증류해서 얻는다"**는 레시피다. 이는 파운데이션 모델 시대에 아키텍처 혁신을 시도하는 비용 구조 자체를 바꾸는 접근이며, 자본이 제한된 팀도 새로운 백본을 실험할 수 있게 한다. 고해상도 처리가 필요한 문서 이해, 의료 영상, 위성 영상 도메인에서는 특히 직접적인 임팩트가 있다.

### 사전 지식
- Vision Transformer(ViT)의 구조와 self-attention의 O(n²) 복잡도
- State Space Model / Mamba의 기본 아이디어와 비전 적용 시의 한계 (VisionMamba, MambaVision)
- 지식 증류(Knowledge Distillation)의 기본 개념, 특히 feature-level distillation
- CUDA 커널 최적화 개념 (메모리 대역폭, shared memory tiling, 커널 융합) — 세부 구현까지는 아니어도 "왜 이론 복잡도와 실제 속도가 다른가"에 대한 감각
- CLIP 스타일 이미지-텍스트 대규모 사전학습

### 관련 논문
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale / ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone (Hatamizadeh & Kautz, 2024)](https://arxiv.org/abs/2407.08083)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)

### 실무 적용
고해상도 입력이 필수인 제품에서 가장 직접적으로 유효하다. 문서/영수증/도면 이해(VLM에 2K 이상 이미지를 타일 분할 없이 넣기), 병리 슬라이드 및 의료 영상 분석, 위성·항공 이미지 분석, 고해상도 비디오 스트림 처리 등이 대표적이다. 기존에는 큰 이미지를 타일로 쪼개 각각 인코딩한 뒤 다시 합치는 방식이 표준이었는데, 이 과정에서 타일 경계의 컨텍스트 손실과 엔지니어링 복잡도가 발생했다. 단일 패스 추론이 가능해지면 파이프라인이 단순해지고 GPU 비용도 줄어든다. 또한 교차 연산자 증류 레시피는 사내에 이미 학습된 ViT 인코더가 있는 팀이 그 자산을 버리지 않고 더 빠른 아키텍처로 옮겨갈 수 있는 실질적 마이그레이션 경로를 제시한다.

---

## 추천 읽기 순서

1. **MobileNets (2017)** — 먼저 읽는다. depthwise separable convolution이라는 구체적인 연산 분해를 통해 "연산량을 어떻게 계산하고 어떻게 줄이는가"의 감각을 잡는다. 논문이 6페이지로 짧고 수식이 직관적이라 진입 장벽이 낮다.
2. **Squeeze-and-Excitation Networks (2017)** — 다음으로 읽는다. MobileNet이 "연산을 줄이는" 방향이라면 SE는 "같은 연산으로 더 잘 쓰는" 방향이다. 두 논문을 이어 읽으면 효율 아키텍처 설계의 두 축이 대비되어 보인다. MobileNetV3가 이 둘을 결합한다는 점도 확인할 것.
3. **C-GSPN (2026)** — 마지막에 읽는다. 앞의 두 고전이 다룬 "효율성"이라는 주제가 파운데이션 모델 시대에 어떻게 재해석되는지 확인한다. 처음 읽을 때는 CUDA 커널 최적화 파트(4장)를 건너뛰고 5장(아키텍처·학습 효율)과 6.4절(파운데이션 스케일 결과)에 집중해도 좋다.

## 핵심 테이크어웨이

- **효율성은 아키텍처 설계의 부수 조건이 아니라 일급 목표다.** MobileNet의 width multiplier부터 C-GSPN의 커널 융합까지, 좋은 비전 아키텍처 논문은 항상 "정확도 vs 비용"의 Pareto front 자체를 밀어올린다. 단일 SOTA 숫자보다 곡선 전체를 보는 습관이 필요하다.
- **"어디에 집중할 것인가"는 저렴하게 배울 수 있다.** SE 블록은 파라미터 1% 미만 증가로 의미 있는 성능 향상을 만들었다. 어텐션은 반드시 비싸야 하는 것이 아니며, 문제에 맞는 최소한의 어텐션 형태를 찾는 것이 설계자의 일이다.
- **이론적 복잡도와 실제 속도는 다르다.** C-GSPN이 명확히 보여주듯, 좋은 asymptotic complexity를 가진 연산자도 하드웨어 친화적으로 구현되지 않으면 무용지물이다. FLOPs가 아니라 실측 지연시간과 처리량으로 판단해야 한다.
- **데이터의 구조적 사전지식(inductive bias)은 여전히 유효하다.** 이미지는 2D다. 이 단순한 사실을 무시하고 1D로 직렬화하면 대가를 치른다. 도메인 구조를 아키텍처에 반영하는 것은 스케일링으로 완전히 대체되지 않는다.
- **새 아키텍처의 진입 비용은 증류로 낮출 수 있다.** 파운데이션 스케일에서 from-scratch 학습은 대부분의 팀에게 불가능하다. 기존 모델로부터의 교차 연산자 증류는 아키텍처 실험을 민주화하는 실질적 도구다.

## 다음 토픽과의 연결

다음 토픽은 **RNN, LSTM and Sequence Models**다. 오늘 다룬 C-GSPN이 이미지를 시퀀스로 다루는 접근(선형 어텐션, SSM, 공간 전파 재귀)의 한계와 가능성을 논의했다면, 다음 토픽에서는 그 시퀀스 모델링의 원류인 LSTM과 seq2seq를 되짚고, Mamba를 비롯한 현대 state space model이 왜 다시 재귀(recurrence)로 돌아왔는지를 살펴본다. 특히 C-GSPN의 line-scan recurrence는 본질적으로 2D로 확장된 재귀 연산이므로, LSTM의 게이팅 메커니즘과 병렬 스캔(parallel scan) 개념을 이해하고 나면 오늘 논문의 커널 최적화 부분이 훨씬 명확해진다. 두 토픽을 잇는 질문: *"컨볼루션, 어텐션, 재귀 — 이 세 가지 연산 원시(primitive) 중 무엇이 어떤 데이터 구조에 왜 적합한가?"*
