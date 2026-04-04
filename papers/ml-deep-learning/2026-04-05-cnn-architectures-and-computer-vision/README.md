# Daily AI Paper Recommendations

> **Date:** 2026-04-05
> **Module:** Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Deep Residual Learning for Image Recognition
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1512.03385
- **PDF:** [./deep-residual-learning-he-2015.pdf](./deep-residual-learning-he-2015.pdf)
- **Citation Count:** ~220,000+

### 요약
깊은 신경망의 학습을 용이하게 하기 위해 잔차 학습(residual learning) 프레임워크를 제안한 논문이다. Skip connection을 통해 입력을 직접 출력에 더하는 구조로, 기존에는 학습이 어려웠던 152층 이상의 매우 깊은 네트워크를 성공적으로 훈련시켰다. ILSVRC 2015 이미지 분류 대회에서 3.57% top-5 에러율로 1위를 차지했다.

### 핵심 기여
- **잔차 학습 (Residual Learning):** 레이어가 입력 대비 잔차 함수를 학습하도록 재구성하여, 깊은 네트워크에서 발생하는 degradation 문제를 해결
- **Skip Connection:** identity shortcut connection을 통해 그래디언트가 깊은 레이어까지 원활하게 전파되도록 설계
- **극단적 깊이의 네트워크:** 152층까지 성공적으로 학습하면서도 VGG보다 낮은 계산 복잡도 유지

### 이 논문이 중요한 이유
ResNet은 현대 딥러닝 아키텍처의 근간이 되는 논문이다. Skip connection 개념은 이후 거의 모든 딥러닝 아키텍처(Transformer 포함)에 채택되었으며, "깊이를 늘리면 성능이 향상된다"는 패러다임을 실증적으로 입증했다. AI 엔지니어라면 이 논문의 핵심 아이디어를 반드시 이해해야 한다.

### 사전 지식
- CNN의 기본 구조 (convolution, pooling, fully connected layers)
- 역전파(backpropagation)와 그래디언트 소실(vanishing gradient) 문제
- VGGNet, GoogLeNet 등 이전 CNN 아키텍처에 대한 기본 이해
- Batch Normalization 개념

### 관련 논문
- [Very Deep Convolutional Networks for Large-Scale Image Recognition / VGGNet (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [Going Deeper with Convolutions / GoogLeNet (Szegedy et al., 2014)](https://arxiv.org/abs/1409.4842)
- [Identity Mappings in Deep Residual Networks (He et al., 2016)](https://arxiv.org/abs/1603.05027)
- [Densely Connected Convolutional Networks / DenseNet (Huang et al., 2016)](https://arxiv.org/abs/1608.06993)

### 실무 적용
ResNet은 이미지 분류, 객체 탐지(Faster R-CNN의 backbone), 세그멘테이션, 의료 영상 분석 등 거의 모든 컴퓨터 비전 태스크의 backbone으로 널리 사용된다. 특히 transfer learning에서 ImageNet 사전훈련된 ResNet 모델은 산업 현장에서 가장 많이 활용되는 모델 중 하나이며, PyTorch/TensorFlow에서 한 줄로 불러올 수 있을 정도로 생태계가 잘 갖춰져 있다.

---

## Paper 2 (Classic): ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
- **Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Year:** 2012
- **URL:** https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- **PDF:** [./imagenet-classification-krizhevsky-2012.pdf](./imagenet-classification-krizhevsky-2012.pdf)
- **Citation Count:** ~150,000+

### 요약
120만 장의 고해상도 ImageNet 이미지를 1,000개 클래스로 분류하는 대규모 심층 합성곱 신경망을 훈련시킨 논문이다. 5개의 합성곱 레이어와 3개의 완전연결 레이어로 구성된 이 네트워크는 ILSVRC-2012 대회에서 top-5 에러율 15.3%를 달성하며, 기존 최고 성능(26.2%)을 압도적으로 개선했다. 딥러닝 혁명의 시작점으로 평가된다.

### 핵심 기여
- **GPU 기반 대규모 CNN 훈련:** 두 개의 GPU를 사용한 병렬 훈련으로 대규모 CNN의 실용적 학습 가능성을 최초로 입증
- **ReLU 활성화 함수:** 기존의 tanh/sigmoid 대비 훈련 속도를 수배 향상시키는 ReLU의 효과를 실증
- **Dropout 정규화:** 과적합 방지를 위한 dropout 기법을 대규모 네트워크에 최초로 적용
- **Data Augmentation:** 학습 데이터 증강 기법(이미지 변환, PCA 기반 색상 변환)을 체계적으로 활용

### 이 논문이 중요한 이유
AlexNet은 "딥러닝의 빅뱅"이라 불리는 논문으로, 2012년 ImageNet 대회에서의 압도적 승리를 통해 딥러닝이 컴퓨터 비전의 주류 방법론이 되는 계기를 만들었다. 이 논문 이후 GPU 기반 딥러닝 연구가 폭발적으로 증가했으며, 현재 AI 산업의 근본적인 방향을 결정한 역사적 논문이다.

### 사전 지식
- 합성곱 신경망(CNN)의 기본 개념
- ImageNet 데이터셋과 ILSVRC 대회의 의미
- 기본적인 신경망 훈련 과정 (forward/backward pass)
- GPU 컴퓨팅의 기본 개념

### 관련 논문
- [Gradient-Based Learning Applied to Document Recognition / LeNet (LeCun et al., 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [Large Scale Visual Recognition Challenge / ILSVRC (Russakovsky et al., 2015)](https://arxiv.org/abs/1409.0575)
- [Network In Network (Lin et al., 2013)](https://arxiv.org/abs/1312.4400)
- [OverFeat (Sermanet et al., 2013)](https://arxiv.org/abs/1312.6229)

### 실무 적용
AlexNet 자체는 현재 실무에서 직접 사용되지는 않지만, 이 논문이 도입한 핵심 기법들(ReLU, dropout, data augmentation, GPU 훈련)은 현재 모든 딥러닝 파이프라인의 기본 요소로 자리 잡았다. 또한 "대규모 데이터 + 깊은 네트워크 + GPU"라는 공식은 오늘날 LLM과 foundation model 시대까지 이어지는 AI 발전의 핵심 패러다임이다.

---

## Paper 3 (Recent): SAM 2: Segment Anything in Images and Videos
- **Authors:** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2408.00714
- **PDF:** [./sam2-segment-anything-ravi-2024.pdf](./sam2-segment-anything-ravi-2024.pdf)
- **Citation Count:** ~1,500+ (2024년 출판 이후 빠르게 증가 중)

### 요약
Meta AI가 발표한 SAM 2는 이미지와 비디오 모두에서 프롬프트 기반 시각적 세그멘테이션을 수행하는 통합 파운데이션 모델이다. 스트리밍 메모리를 갖춘 트랜스포머 아키텍처를 사용하여 실시간 비디오 처리가 가능하며, 사용자 상호작용을 통해 모델과 데이터를 동시에 개선하는 데이터 엔진을 구축하여 역대 최대 규모의 비디오 세그멘테이션 데이터셋(SA-V, 5.09만 비디오, 3,550만 마스크)을 수집했다.

### 핵심 기여
- **이미지-비디오 통합 세그멘테이션:** 이미지와 비디오를 단일 모델로 처리하는 통합 아키텍처 제안
- **스트리밍 메모리 아키텍처:** 실시간 비디오 처리를 위한 효율적인 메모리 메커니즘으로 긴 비디오에서도 일관된 객체 추적 가능
- **대규모 데이터 엔진:** Human-in-the-loop 방식으로 모델 성능과 데이터 품질을 동시에 개선하는 데이터 수집 파이프라인
- **SA-V 데이터셋:** 기존 비디오 세그멘테이션 데이터셋 대비 53배 더 많은 마스크를 포함하는 대규모 벤치마크

### 이 논문이 중요한 이유
SAM 2는 비전 파운데이션 모델의 최전선을 보여주는 논문이다. 이미지에서 비디오로의 확장, 프롬프트 기반 상호작용, 데이터 플라이휠 구축 등 현대 AI 제품 개발의 핵심 패턴을 모두 담고 있다. 특히 "프롬프트 엔지니어링"이 비전 영역으로 확장되는 트렌드를 이해하는 데 필수적이며, AI 서비스의 UX 설계에도 중요한 시사점을 제공한다.

### 사전 지식
- Transformer 아키텍처와 attention mechanism
- 이미지 세그멘테이션의 기본 개념 (semantic, instance, panoptic)
- 원본 SAM (Segment Anything Model) 논문에 대한 이해
- Vision Transformer (ViT) 기본 구조

### 관련 논문
- [Segment Anything / SAM (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training (Liu et al., 2023)](https://arxiv.org/abs/2303.05499)
- [Track Anything: Segment Anything Meets Videos (Yang et al., 2023)](https://arxiv.org/abs/2304.11968)

### 실무 적용
SAM 2는 비디오 편집, 자율주행, AR/VR, 의료 영상 분석, 로보틱스 등 다양한 산업에서 활용 가능하다. 특히 AI Dubbing이나 AI Avatar 서비스에서 인물 세그멘테이션, 배경 분리, 비디오 객체 추적 등에 직접 적용할 수 있으며, 프롬프트 기반 인터페이스는 비전문가도 쉽게 사용할 수 있는 AI 서비스 UX 설계의 좋은 사례이다.

---

## 추천 읽기 순서

1. **AlexNet (2012)** → 딥러닝 혁명의 시작점. CNN의 기본 구성요소(ReLU, dropout, data augmentation)와 GPU 훈련의 중요성을 이해하는 출발점이다.
2. **ResNet (2015)** → Skip connection과 잔차 학습을 통해 "깊이"의 한계를 극복한 방법을 학습한다. 현대 아키텍처의 기본 빌딩 블록을 이해할 수 있다.
3. **SAM 2 (2024)** → 최신 비전 파운데이션 모델이 어떻게 이미지에서 비디오로, 단일 태스크에서 범용 모델로 진화했는지 확인한다.

## 핵심 테이크어웨이

- **아키텍처 혁신의 힘:** AlexNet→ResNet→SAM 2로 이어지는 흐름은 "더 깊게, 더 크게, 더 범용적으로"라는 딥러닝 발전의 핵심 방향을 보여준다.
- **단순한 아이디어의 강력함:** ResNet의 skip connection은 매우 단순한 아이디어지만, 수백 층의 네트워크 훈련을 가능하게 만든 혁명적 기여였다. 복잡한 문제에 대한 단순하고 우아한 해결책의 가치를 보여준다.
- **프롬프트 기반 상호작용의 확산:** NLP에서 시작된 프롬프트 패러다임이 SAM 2를 통해 비전 영역까지 확장되었다. 이는 AI 제품 설계에서 사용자 인터페이스와 상호작용 방식의 패러다임 전환을 의미한다.
- **데이터 플라이휠:** SAM 2의 데이터 엔진은 모델-데이터 공동 진화의 좋은 사례로, AI 제품의 지속적 성장 전략에 시사점을 제공한다.

## 다음 토픽과의 연결

다음 토픽은 **RNN, LSTM과 Sequence Models**로, 이미지(공간적 데이터)에서 시퀀스(시간적 데이터)로 관심을 확장한다. CNN이 공간적 패턴을 포착하는 데 탁월했다면, RNN/LSTM은 시간적 의존성을 모델링하는 데 설계된 아키텍처이다. 흥미롭게도 SAM 2의 비디오 처리에서 볼 수 있듯이, 현대 모델들은 공간과 시간을 동시에 처리하는 방향으로 진화하고 있다. 다음 논문들에서는 이 시퀀스 모델링의 기초와 최신 발전(특히 Mamba 같은 State Space Models)을 살펴볼 예정이다.
