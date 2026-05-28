# Daily AI Paper Recommendations

> **Date:** 2026-05-29
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Going Deeper with Convolutions (GoogLeNet / Inception)
- **Authors:** Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- **Year:** 2014 (CVPR 2015)
- **arXiv:** https://arxiv.org/abs/1409.4842
- **PDF:** [./going-deeper-with-convolutions-szegedy-2014.pdf](./going-deeper-with-convolutions-szegedy-2014.pdf)
- **Citation Count:** 60,000+

### 요약
구글이 ILSVRC14에서 1위를 차지한 22층 깊이의 GoogLeNet(Inception v1) 아키텍처를 제안한 논문이다. "Network in Network" 아이디어를 발전시킨 Inception 모듈을 도입하여, 동일 레이어에서 1x1·3x3·5x5 합성곱과 풀링을 병렬로 수행하고 그 결과를 concat한다. 1x1 합성곱으로 채널 차원을 축소(bottleneck)함으로써 파라미터 수와 연산량을 극적으로 줄였으며, AlexNet 대비 12배 적은 파라미터로 더 높은 정확도를 달성했다.

### 핵심 기여
- **Inception 모듈:** 여러 크기의 필터(1x1, 3x3, 5x5)와 풀링을 병렬 분기로 결합해 멀티스케일 특징을 동시 추출
- **1x1 합성곱 bottleneck:** 채널을 줄여 연산량을 통제하면서 깊은 네트워크 구성 가능 — 이후 모든 효율적 CNN의 표준 패턴이 됨
- **Auxiliary classifier:** 중간 레이어에 보조 분류기를 두어 깊은 네트워크의 gradient vanishing 완화
- **Global Average Pooling 사용:** FC 레이어를 제거해 파라미터 수 대폭 절감 (AlexNet 60M → GoogLeNet 5M)

### 이 논문이 중요한 이유
"무작정 깊고 넓게" 만들던 시대에서 "계산 효율성을 고려한 아키텍처 설계" 시대로 넘어가는 분기점이 된 논문이다. 1x1 conv를 통한 차원 축소 기법은 ResNet의 bottleneck block, MobileNet의 pointwise conv, 최근의 efficient transformer까지 이어지는 핵심 아이디어다. AI 엔지니어가 모바일·엣지 환경에 모델을 배포할 때 "어떻게 파라미터와 FLOPs를 줄일 것인가"를 고민하는 모든 출발점이 여기에 있다.

### 사전 지식
- 합성곱 신경망(CNN) 기본 구조 (Conv, Pooling, FC)
- ImageNet/ILSVRC 벤치마크의 의미
- AlexNet, VGG의 구조와 한계 (파라미터 폭증, 메모리 문제)
- 1x1 convolution의 수학적 의미 (채널 mixing)

### 관련 논문
- [Network In Network (Lin et al., 2013)](https://arxiv.org/abs/1312.4400)
- [Very Deep Convolutional Networks / VGG (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [Rethinking the Inception Architecture / Inception v3 (Szegedy et al., 2015)](https://arxiv.org/abs/1512.00567)
- [Xception: Deep Learning with Depthwise Separable Convolutions (Chollet, 2016)](https://arxiv.org/abs/1610.02357)

### 실무 적용
- **모바일/엣지 비전 모델:** Inception 계열 변형은 여전히 임베디드 디바이스의 이미지 분류·OCR 백본으로 활용됨
- **TensorFlow/Keras 표준 모델:** `tf.keras.applications.InceptionV3` 등으로 즉시 fine-tuning 가능
- **AI Dubbing/Avatar 워크플로:** 얼굴·립싱크 전처리 단계에서 가벼운 CNN 백본이 필요할 때 Inception 계열로 추론 비용 절감
- **멀티스케일 인식:** 의료 영상, 위성 영상처럼 객체 크기 편차가 큰 도메인에서 Inception의 multi-branch 사고방식이 여전히 유효

---

## Paper 2 (Classic): U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors:** Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Year:** 2015 (MICCAI 2015)
- **arXiv:** https://arxiv.org/abs/1505.04597
- **PDF:** [./u-net-convolutional-networks-ronneberger-2015.pdf](./u-net-convolutional-networks-ronneberger-2015.pdf)
- **Citation Count:** 90,000+

### 요약
의료 영상 세그멘테이션을 위해 제안된 인코더-디코더 구조의 완전 합성곱 네트워크(FCN)다. 다운샘플링하는 contracting path와 업샘플링하는 expansive path가 U자 모양으로 대칭을 이루며, 같은 해상도의 인코더-디코더 특징을 skip connection으로 직접 연결한다. 30장의 학습 데이터와 강력한 augmentation만으로 ISBI 세포 추적 챌린지에서 1위를 차지했다.

### 핵심 기여
- **U자형 인코더-디코더 + Skip Connection:** 디코더에서 픽셀 단위 위치 정보를 복원할 때, 인코더의 고해상도 특징을 직접 가져와 디테일 손실 방지
- **소량 데이터 + Elastic Deformation Augmentation:** 의료 영상처럼 라벨이 부족한 환경에서도 학습 가능함을 입증
- **End-to-End Pixel-wise Prediction:** 슬라이딩 윈도우 방식 대비 속도와 정확도 모두 압도
- **Overlap-tile Strategy:** 큰 이미지를 타일 단위로 처리하면서 경계 정보 손실 최소화

### 이 논문이 중요한 이유
세그멘테이션을 넘어 "dense prediction"이 필요한 모든 영역의 기본 골격이 되었다. Stable Diffusion·DALL-E의 UNet, 의료 영상 분할, 자율주행의 BEV 인식, 깊이 추정, 매팅(matting), nnU-Net을 통한 의료 AI까지 — U-Net의 변형이 들어가지 않은 컴퓨터 비전 시스템을 찾기가 어렵다. 특히 생성형 AI 시대에 와서 diffusion 모델의 backbone으로 다시 한 번 주목받게 된 점이 인상적이다.

### 사전 지식
- Fully Convolutional Networks(FCN, Long et al., 2015)의 의미
- 의미적 분할(semantic segmentation) vs 인스턴스 분할(instance segmentation)
- Transposed convolution / Upsampling 기법
- Data augmentation의 역할

### 관련 논문
- [Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)](https://arxiv.org/abs/1411.4038)
- [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation (Isensee et al., 2020)](https://arxiv.org/abs/1904.08128)
- [Denoising Diffusion Probabilistic Models / DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models / Stable Diffusion (Rombach et al., 2021)](https://arxiv.org/abs/2112.10752)

### 실무 적용
- **생성형 이미지/비디오 모델:** Stable Diffusion·SDXL·AnimateDiff의 노이즈 예측 backbone이 U-Net 구조
- **AI Avatar/Dubbing 파이프라인:** 얼굴 세그멘테이션, 배경 제거, 립싱크 마스크 생성에 직접 활용
- **의료 AI SaaS:** 흉부 X-ray, 병리 슬라이드, 안과 OCT 등 픽셀 단위 진단 모델의 표준 구조
- **문서 AI:** 영수증·계약서의 영역 분할, 표 구조 인식

---

## Paper 3 (Recent): Depth Anything V2
- **Authors:** Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.09414
- **PDF:** [./depth-anything-v2-yang-2024.pdf](./depth-anything-v2-yang-2024.pdf)
- **Citation Count:** 1,500+ (2026년 5월 기준)

### 요약
단안(monocular) 깊이 추정을 위한 강력한 파운데이션 모델로, V1 대비 더 미세하고 강건한 깊이 예측을 보여준다. 핵심 인사이트는 "실제 라벨 이미지 대신 합성 이미지로만 teacher를 학습 → 거대 teacher를 만든 뒤 → 수백만 장의 의사 라벨(pseudo-labeled) 실 이미지로 student를 distillation"하는 3단계 전략이다. 25M에서 1.3B 파라미터까지 다양한 사이즈를 제공하며, V1 대비 약 10배 빠르면서도 더 정확하다.

### 핵심 기여
- **합성 데이터의 재발견:** 실제 데이터의 라벨 노이즈와 모호성이 오히려 정밀도를 해친다는 것을 정량적으로 입증, 합성 이미지만 사용한 teacher가 더 우수함을 보임
- **대규모 Pseudo-label Distillation:** 거대 teacher가 라벨링한 6,200만 장의 실제 이미지로 student를 학습 → 일반화 성능과 효율성 모두 확보
- **DA-2K 벤치마크 제안:** 기존 벤치마크의 라벨 노이즈 문제를 해결하는 정밀 평가셋
- **Production-ready 모델 패밀리:** Small/Base/Large/Giant까지 동일한 학습 파이프라인으로 제공 — 실제 배포 환경에 맞춰 선택 가능

### 이 논문이 중요한 이유
"비전 파운데이션 모델 = 거대 트랜스포머 + 많은 진짜 데이터"라는 통념을 깨고, **합성 데이터 + 지식 증류**가 단일 태스크 파운데이션 모델을 만드는 더 좋은 방법이 될 수 있음을 보여준다. AI 엔지니어 입장에서 (1) 라벨이 부족한 도메인에서 어떻게 모델을 만들 것인가, (2) 거대 모델의 지식을 어떻게 작은 모델로 옮길 것인가, (3) 합성 데이터 파이프라인을 어떻게 설계할 것인가 — 이 세 가지 핵심 질문에 대한 2024년 최신 해답이다.

### 사전 지식
- 단안 깊이 추정(Monocular Depth Estimation) 태스크 정의
- Knowledge Distillation의 기본 (teacher-student, soft label)
- DINOv2 등 self-supervised vision backbone
- Sim-to-Real domain gap 문제

### 관련 논문
- [Depth Anything V1 (Yang et al., 2024)](https://arxiv.org/abs/2401.10891)
- [MiDaS: Towards Robust Monocular Depth Estimation (Ranftl et al., 2019)](https://arxiv.org/abs/1907.01341)
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Segment Anything 2 / SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714)

### 실무 적용
- **AR/VR·3D 콘텐츠 생성:** 단일 이미지에서 즉시 3D 메시·깊이맵 생성, AI Avatar의 3D 합성에 활용
- **AI 영상 편집/Dubbing:** 깊이 정보 기반 배경 분리, depth-aware 시각 효과, parallax video
- **자율주행·로보틱스 prototyping:** LiDAR 없이도 합리적 수준의 깊이 인식
- **이미지 생성 모델의 컨디셔닝:** ControlNet에 depth map을 조건으로 제공해 구도 일관성 확보

---

## 추천 읽기 순서
1. **GoogLeNet (Inception)** — CNN 아키텍처 설계의 사고방식(효율성, 멀티스케일)을 이해
2. **U-Net** — dense prediction과 skip connection이라는 핵심 패턴 학습 (현대 diffusion 모델의 기반)
3. **Depth Anything V2** — 위 두 가지 기반 위에서, 2024년형 "파운데이션 모델 만드는 법"을 체득

처음 두 편은 합쳐 1시간 이내로 읽고 핵심 도식만 노트화. Depth Anything V2는 합성 데이터·distillation 파이프라인을 직접 그려보며 읽는 것을 권장.

## 핵심 테이크어웨이
- **Q: 깊은 CNN의 파라미터 폭발을 어떻게 해결할 수 있을까?**
  - A: 1x1 conv bottleneck과 multi-branch 모듈로 같은 표현력을 더 적은 파라미터로 달성한다(Inception).
- **Q: 픽셀 단위 예측에서 디테일을 살리려면?**
  - A: 인코더-디코더 + skip connection으로 저수준 특징을 보존한다(U-Net).
- **Q: 파운데이션 모델은 꼭 거대한 실 데이터가 필요한가?**
  - A: 깨끗한 합성 데이터로 teacher를 만들고, 그 teacher로 만든 pseudo label로 student를 distillation하면 더 강력해질 수 있다(Depth Anything V2).
- **공통 원칙:** 더 큰 모델이 아니라, **더 똑똑한 데이터·아키텍처 설계**가 성능을 만든다.

## 다음 토픽과의 연결
다음 토픽인 **"RNN·LSTM과 시퀀스 모델"**로 넘어가면, 이미지의 공간 구조를 다루던 CNN에서 시간·시퀀스 구조를 다루는 모델로 시야를 확장한다. 특히 U-Net의 skip connection·인코더-디코더 아이디어는 seq2seq의 attention과 직접 연결되며, GoogLeNet의 multi-branch 사고는 이후 트랜스포머의 multi-head attention으로 진화한다. 깊이 추정처럼 한 모달에서의 파운데이션 모델 패턴은, 다음 단계에서 다룰 음성·텍스트 파운데이션 모델의 학습 전략을 이해하는 데에도 그대로 적용된다.
