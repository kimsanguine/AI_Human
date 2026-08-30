# Daily AI Paper Recommendations

> **Date:** 2026-06-25
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Densely Connected Convolutional Networks (DenseNet)
- **Authors:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
- **Year:** 2016 (CVPR 2017)
- **arXiv:** https://arxiv.org/abs/1608.06993
- **PDF:** [./densenet-densely-connected-huang-2017.pdf](./densenet-densely-connected-huang-2017.pdf)
- **Citation Count:** ~45,000+

### 요약
DenseNet은 각 레이어를 이전의 모든 레이어와 직접 연결(dense connection)하는 구조를 제안한다. L개 레이어를 가진 일반 CNN이 L개의 연결을 갖는다면, DenseNet은 L(L+1)/2개의 연결을 가진다. 이를 통해 그래디언트 소실 문제를 완화하고, 특징 재사용(feature reuse)을 극대화하면서도 파라미터 수를 크게 줄인다.

### 핵심 기여
- Dense connectivity 패턴 제안: 모든 레이어가 후속 레이어로 feature map을 직접 전달하여 정보·그래디언트 흐름을 개선
- 특징 재사용을 통해 ResNet보다 적은 파라미터로 더 높은 정확도 달성
- CIFAR-10/100, SVHN, ImageNet 등 주요 벤치마크에서 SOTA 수준 성능을 더 적은 연산으로 달성

### 이 논문이 중요한 이유
DenseNet은 "연결 패턴(connectivity)"이 네트워크 깊이만큼이나 중요하다는 것을 보여준 대표적 아키텍처다. ResNet의 skip connection을 극단까지 밀어붙여, 파라미터 효율성과 그래디언트 흐름이라는 두 마리 토끼를 잡았다. AI 엔지니어에게는 효율적인 백본 설계의 사고방식과 feature reuse라는 개념을 익히는 데 필독이다.

### 사전 지식
- CNN의 기본 구조(convolution, pooling, BatchNorm)
- ResNet의 residual connection 개념과 그래디언트 소실 문제
- 파라미터 수 vs 정확도 vs 연산량(FLOPs) 사이의 트레이드오프

### 관련 논문
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Highway Networks (Srivastava et al., 2015)](https://arxiv.org/abs/1505.00387)

### 실무 적용
DenseNet 백본은 의료 영상 진단(적은 데이터로 높은 정확도가 필요한 영역), 임베디드/모바일 비전처럼 파라미터 예산이 빠듯한 환경에서 자주 쓰인다. 또한 feature reuse 아이디어는 현대 백본(EfficientNet, ConvNeXt) 설계에 영향을 주었다.

---

## Paper 2 (Classic): Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- **Authors:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **Year:** 2015 (NeurIPS 2015)
- **arXiv:** https://arxiv.org/abs/1506.01497
- **PDF:** [./faster-rcnn-ren-2015.pdf](./faster-rcnn-ren-2015.pdf)
- **Citation Count:** ~60,000+

### 요약
Faster R-CNN은 객체 탐지(object detection)에서 병목이던 region proposal 단계를 신경망 안으로 통합했다. Region Proposal Network(RPN)가 detection 네트워크와 convolutional feature를 공유하여 거의 비용 없이 region proposal을 생성하고, end-to-end 학습이 가능하게 만들었다.

### 핵심 기여
- Region Proposal Network(RPN): 객체 경계와 objectness 점수를 각 위치에서 동시에 예측하는 fully-convolutional 네트워크
- Detection 네트워크와 feature 공유로 proposal 생성 비용을 거의 0에 가깝게 축소
- Anchor box 개념 도입으로 다양한 크기·종횡비 객체를 효율적으로 처리, PASCAL VOC·MS COCO에서 SOTA 달성

### 이 논문이 중요한 이유
Faster R-CNN은 2-stage 객체 탐지의 사실상 표준이 된 프레임워크다. RPN과 anchor라는 개념은 이후 거의 모든 탐지 모델(Mask R-CNN, FPN 등)의 기반이 되었다. CNN을 분류를 넘어 "위치까지 예측하는" 태스크로 확장하는 사고를 익히는 데 핵심 논문이다.

### 사전 지식
- R-CNN, Fast R-CNN의 발전 과정과 selective search의 한계
- CNN feature map과 RoI(Region of Interest) pooling 개념
- 분류(classification)와 회귀(bounding box regression)의 multi-task loss

### 관련 논문
- [Fast R-CNN (Girshick, 2015)](https://arxiv.org/abs/1504.08083)
- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)

### 실무 적용
Faster R-CNN 계열은 자율주행, 영상 감시, 산업 결함 검사, 의료 영상의 병변 탐지 등 높은 정확도가 필요한 객체 탐지 서비스에서 여전히 널리 쓰인다. Detectron2, MMDetection 같은 주요 프레임워크의 기본 모델로 제공된다.

---
