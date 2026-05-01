# Daily AI Paper Recommendations

> **Date:** 2026-05-02
> **Module:** Module 3 - Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)
- **Authors:** Karen Simonyan, Andrew Zisserman
- **Year:** 2014 (ICLR 2015)
- **arXiv:** https://arxiv.org/abs/1409.1556
- **PDF:** [./vgg-very-deep-cnn-simonyan-2014.pdf](./vgg-very-deep-cnn-simonyan-2014.pdf)
- **Citation Count:** 100,000회 이상

### 요약
VGGNet은 3×3 작은 필터만을 반복적으로 쌓아 네트워크 깊이를 16~19층까지 확장한 CNN 구조다. "더 깊을수록 더 강하다"는 가설을 체계적인 아블레이션 실험으로 검증했고, ILSVRC-2014에서 분류 2위·로컬라이제이션 1위를 차지하며 깊이의 중요성을 입증했다.


### 핵심 기여
- 큰 필터(7×7, 11×11) 대신 3×3 필터를 누적해 같은 receptive field를 더 적은 파라미터·더 많은 비선형성으로 구현
- 깊이(11→13→16→19층)에 따른 성능 변화를 통제 변수 실험으로 정량화한 첫 연구
- VGG-16/VGG-19 사전학습 모델을 공개하여 후속 연구의 표준 백본(backbone)으로 자리잡음

### 이 논문이 중요한 이유
AI 엔지니어에게 VGGNet은 "단순함의 힘"을 보여주는 교과서다. 작은 필터 반복이라는 단일 디자인 원칙만으로 SOTA를 달성한 사례는 모델 설계 시 복잡도와 성능의 트레이드오프를 고민할 때 항상 참조된다. 또한 perceptual loss, style transfer, feature extractor 등 다양한 다운스트림 태스크에서 여전히 baseline으로 쓰이며, "왜 깊이가 중요한가"라는 근본 질문에 대한 실증적 답변을 제공한다.

### 사전 지식
- CNN 기본 개념 (convolution, pooling, fully-connected layer)
- ImageNet 데이터셋과 ILSVRC 챌린지의 의미
- AlexNet(2012) 구조와 한계
- Receptive field 개념과 stacking의 효과


### 관련 논문
- [ImageNet Classification with Deep CNNs / AlexNet (Krizhevsky et al., 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [Going Deeper with Convolutions / GoogLeNet (Szegedy et al., 2014)](https://arxiv.org/abs/1409.4842)
- [Deep Residual Learning / ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)

### 실무 적용
- **사전학습 백본:** Object detection(Faster R-CNN, SSD), semantic segmentation(FCN) 초기 버전의 기본 backbone
- **Perceptual loss:** Style transfer, super-resolution, GAN training에서 VGG feature 기반 손실 함수가 표준
- **Feature extraction:** 의료 영상·산업 비전 등 데이터가 적은 도메인에서 transfer learning 출발점
- **모델 설계 원칙:** 모듈러·반복적 구조 설계 패턴은 ResNet, DenseNet, EfficientNet에 계승됨

---

## Paper 2 (Classic): An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)
- **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **Year:** 2020 (ICLR 2021)
- **arXiv:** https://arxiv.org/abs/2010.11929
- **PDF:** [./vit-image-worth-16x16-words-dosovitskiy-2020.pdf](./vit-image-worth-16x16-words-dosovitskiy-2020.pdf)
- **Citation Count:** 60,000회 이상

### 요약
ViT는 이미지를 16×16 패치로 쪼개 토큰처럼 다루고, NLP에서 검증된 순수 Transformer 인코더만으로 이미지 분류를 수행한 모델이다. 충분히 큰 데이터(JFT-300M)로 사전학습할 경우 CNN의 inductive bias 없이도 SOTA를 달성할 수 있음을 보여주며 "비전=CNN" 패러다임을 종식시켰다.


### 핵심 기여
- 이미지를 패치 시퀀스로 변환하여 NLP Transformer를 vision에 그대로 적용한 최초의 대규모 성공 사례
- "Convolution은 필수가 아니다 — 데이터 규모가 inductive bias를 대체할 수 있다"는 스케일링 법칙 입증
- Patch embedding + class token + positional embedding 구조는 이후 모든 vision Transformer의 표준 레시피가 됨

### 이 논문이 중요한 이유
ViT는 멀티모달·파운데이션 모델 시대의 출발점이다. 이미지·텍스트·오디오를 모두 토큰으로 통일하는 사고방식 덕분에 CLIP, DALL·E, SAM, GPT-4V 같은 멀티모달 모델이 가능해졌다. AI 엔지니어가 이미지+LLM을 통합 파이프라인으로 다룰 때 거의 모든 비전 인코더가 ViT 계열이며, 토크나이저·attention·스케일링 직관을 비전에도 그대로 적용할 수 있게 해준 결정적 논문이다.

### 사전 지식
- Transformer 구조 (Self-attention, Multi-head attention, Layer norm)
- BERT/GPT의 토큰 임베딩과 positional encoding
- Inductive bias 개념과 CNN(translation equivariance, locality)의 의미
- Pre-training & fine-tuning 패러다임

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Masked Autoencoders Are Scalable Vision Learners / MAE (He et al., 2021)](https://arxiv.org/abs/2111.06377)
- [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)

### 실무 적용
- **멀티모달 모델 백본:** CLIP, BLIP, LLaVA, GPT-4V 등 거의 모든 vision-language 모델의 이미지 인코더
- **자기지도 학습:** DINO, MAE, BEiT 등 SSL 기법이 ViT 위에서 발전 → 라벨 없이 강력한 표현 학습
- **세그멘테이션·디텍션:** SAM(Segment Anything), Mask2Former 등 다용도 vision 모델의 backbone
- **엣지·모바일:** MobileViT, EfficientFormer 등 경량 변형으로 온디바이스 추론에도 진출

---


## Paper 3 (Recent): DINOv3
- **Authors:** Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (외 Meta AI Research)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2508.10104
- **PDF:** [./dinov3-vision-foundation-model-simeoni-2025.pdf](./dinov3-vision-foundation-model-simeoni-2025.pdf)
- **Citation Count:** 빠르게 증가 중 (공개 후 수개월 만에 수백 회)

### 요약
DINOv3는 Meta가 공개한 차세대 자기지도 vision foundation model로, 백본을 7B 파라미터까지 확장하고 17억 장 이미지에서 라벨 없이 학습한다. 핵심 기여인 Gram anchoring 기법으로 장기 학습 시 dense feature가 무너지는 문제를 해결, 단일 frozen 모델이 분류·디텍션·세그멘테이션·깊이 추정 등 모든 비전 태스크에서 fine-tuning 없이 SOTA를 달성한다.

### 핵심 기여
- **Gram anchoring:** 장기 SSL 학습에서 dense feature map 품질이 저하되는 현상을 막는 새로운 정규화 기법
- **스케일링:** DINOv2(1.1B) → DINOv3(7B), 학습 데이터 142M → 1.7B 이미지로 확장
- **Universal frozen backbone:** 단 하나의 동결 모델로 분류·세그멘테이션·디텍션·뎁스·매칭 태스크 SOTA
- **모델 패밀리 공개:** ViT-S/B/L/H+ 다양한 크기와 ConvNeXt 변형까지 포함, 실무 도입 장벽을 낮춤

### 이 논문이 중요한 이유
2025년 시점 비전 영역의 "GPT 모먼트"에 가장 가까운 모델이다. AI 엔지니어 입장에서 라벨 없이 학습한 frozen feature만으로 다운스트림 SOTA가 가능하다는 것은, 도메인별 데이터 라벨링·모델 학습 비용을 극적으로 줄일 수 있다는 의미다. 의료·위성·산업·로보틱스 등 데이터 라벨이 비싼 도메인에서 RAG처럼 "DINOv3 feature + lightweight head" 패턴이 새로운 표준이 되고 있어, vision 파운데이션 모델 도입 전략을 결정할 때 반드시 검토해야 한다.

### 사전 지식
- ViT 구조와 self-supervised learning 개념
- DINO/DINOv2의 student-teacher 프레임워크와 EMA 업데이트
- Masked Image Modeling (MAE, BEiT)
- Linear probing vs. fine-tuning 평가 방식
- Dense prediction 태스크 (segmentation, depth) 평가 지표


### 관련 논문
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Emerging Properties in Self-Supervised Vision Transformers / DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [Masked Autoencoders Are Scalable Vision Learners / MAE (He et al., 2021)](https://arxiv.org/abs/2111.06377)
- [Segment Anything 2 / SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714)

### 실무 적용
- **데이터 효율적 분류기:** Frozen DINOv3 + linear/MLP head로 적은 라벨로도 산업·의료 분류 태스크 빠르게 구축
- **세그멘테이션 backbone:** Dino U-Net, MedDINOv3, SegDINO 등 lightweight decoder 결합 → 라벨 100~1000장 수준에서 fine-tuning 모델 능가
- **검색·매칭:** 이미지 검색, 중복 탐지, 비주얼 매칭의 임베딩 백본으로 CLIP보다 dense feature 품질 우위
- **로보틱스·자율주행:** 라벨 없는 비디오로 self-supervised 학습한 feature를 perception 스택의 공통 인코더로 활용
- **멀티모달 LLM:** 차세대 VLM의 vision tower 후보 — CLIP 대비 더 풍부한 dense representation 제공

---

## 추천 읽기 순서
1. **VGGNet (2014)** → CNN 깊이의 가치와 단순한 디자인 원칙 이해. 30분이면 핵심 파악 가능.
2. **ViT (2020)** → Transformer가 비전을 어떻게 흡수했는지, 스케일링과 inductive bias 트레이드오프 학습. 멀티모달 시대의 출발점.
3. **DINOv3 (2025)** → 2025년 vision foundation model의 현재. ViT 기반 SSL이 어디까지 왔고 실무에 어떻게 적용되는지 체감.

> 비전의 진화 흐름: **CNN(VGG·ResNet) → ViT → SSL Foundation Model(DINOv3)** — 각 단계가 다음 단계를 어떻게 가능하게 했는지 연결지어 읽으면 학습 효과가 극대화된다.

## 핵심 테이크어웨이
- **단순함의 힘:** VGG의 3×3 반복은 "복잡한 트릭보다 일관된 원칙이 강하다"는 교훈을 남겼고, 이는 Transformer 블록 반복으로도 이어진다.
- **데이터 규모 ≫ Inductive Bias:** ViT는 충분한 데이터가 주어지면 손수 설계한 prior(convolution)가 불필요해질 수 있음을 보여줬다. 모델 설계 시 "데이터가 많은가"를 먼저 물어야 한다.
- **Frozen Foundation Model이 표준:** DINOv3처럼 라벨 없이 학습한 universal backbone + lightweight head 패턴이 비전 실무의 새로운 default. 도메인 데이터 라벨링 전에 SSL 백본으로 baseline을 먼저 만들어야 한다.
- **표현 학습의 중심 이동:** 텍스트(LLM)가 그랬듯 비전도 "task-specific 모델 학습 → frozen foundation model + 가벼운 어댑터"로 패러다임 전환 중.

## 다음 토픽과의 연결
다음 토픽 **RNN, LSTM and Sequence Models**와 자연스럽게 이어진다. 오늘 살펴본 vision 아키텍처 진화(CNN → Transformer → SSL Foundation Model)와 거의 동일한 흐름이 시퀀스 모델에서도 반복된다 — RNN·LSTM의 순차 처리 한계를 Transformer가 해체했고, 최근 Mamba·SSM이 다시 효율성을 가져오는 중이다. 또한 ViT가 NLP의 Transformer를 그대로 차용했듯, 시퀀스 모델 발전을 보면 vision Transformer의 한계와 미래를 거꾸로 추론할 수 있다. 두 도메인의 아키텍처 진화를 평행하게 비교하며 읽기를 권장한다.
