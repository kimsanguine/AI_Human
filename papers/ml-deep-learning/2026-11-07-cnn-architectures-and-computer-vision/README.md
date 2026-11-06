# Daily AI Paper Recommendations

> **Date:** 2026-11-07
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Feature Pyramid Networks for Object Detection (FPN)
- **Authors:** Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
- **Year:** 2017 (CVPR 2017, arXiv 2016)
- **arXiv:** https://arxiv.org/abs/1612.03144
- **PDF:** [./feature-pyramid-networks-lin-2017.pdf](./feature-pyramid-networks-lin-2017.pdf)
- **Citation Count:** ~30,000+

### 요약
CNN은 깊은 층으로 갈수록 의미 정보는 풍부해지지만 해상도가 낮아져 작은 물체를 놓치기 쉽습니다. FPN은 top-down 경로와 lateral connection을 추가해 모든 스케일에서 "의미가 풍부한" 피처맵을 저비용으로 만들고, Faster R-CNN에 붙여 COCO에서 단일 모델 SOTA를 달성했습니다.

### 핵심 기여
- 이미지 피라미드(여러 해상도 입력) 없이 CNN 내부의 계층 구조를 그대로 피라미드로 활용하는 설계
- Top-down upsampling + 1x1 lateral connection으로 고해상도·고의미 피처를 결합
- RPN, Fast R-CNN 등 다양한 검출기에 범용 백본 넥(neck)으로 적용 가능함을 실증

### 이 논문이 중요한 이유
현재 거의 모든 검출·분할 모델(Mask R-CNN, RetinaNet, YOLO 계열의 PAN/FPN neck, Mask2Former 등)이 FPN 아이디어를 기본 부품으로 씁니다. "백본-넥-헤드" 구조로 비전 모델을 이해하는 출발점이 되는 논문입니다.

### 사전 지식
- CNN의 receptive field와 다운샘플링(stride, pooling) 개념
- Faster R-CNN의 RPN과 anchor 개념
- 피처맵의 해상도와 의미 정보 사이의 트레이드오프

### 관련 논문
- [Faster R-CNN (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
- [Focal Loss for Dense Object Detection / RetinaNet (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Path Aggregation Network / PANet (Liu et al., 2018)](https://arxiv.org/abs/1803.01534)

### 실무 적용
드론·CCTV·문서 이미지처럼 작은 객체와 큰 객체가 섞인 환경에서 검출 성능을 좌우합니다. Detectron2, MMDetection, Ultralytics YOLO 등 실무 프레임워크에서 FPN/PAN neck은 기본 옵션이며, 작은 객체 검출 성능이 낮을 때 가장 먼저 점검하는 구성 요소입니다.

---

## Paper 2 (Classic): Training data-efficient image transformers & distillation through attention (DeiT)
- **Authors:** Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
- **Year:** 2021 (ICML 2021, arXiv 2020)
- **arXiv:** https://arxiv.org/abs/2012.12877
- **PDF:** [./deit-data-efficient-image-transformers-touvron-2021.pdf](./deit-data-efficient-image-transformers-touvron-2021.pdf)
- **Citation Count:** ~8,000+

### 요약
원조 ViT는 JFT-300M 같은 초대형 데이터가 있어야 CNN을 이겼습니다. DeiT는 강한 데이터 증강·정규화 레시피와 "distillation token"을 통한 CNN 교사 증류를 도입해, ImageNet-1K만으로 단일 8-GPU 노드에서 3일 이내에 학습하면서 CNN과 대등한 성능을 냈습니다.

### 핵심 기여
- ViT를 ImageNet-1K만으로 학습 가능하게 만든 학습 레시피(RandAugment, Mixup, CutMix, Stochastic Depth 등)
- Class token과 별도의 distillation token으로 교사 모델을 따라 배우는 새로운 증류 방식
- CNN(RegNet) 교사가 Transformer 교사보다 더 좋은 학생을 만든다는 발견 — 귀납적 편향(inductive bias)의 전달

### 이 논문이 중요한 이유
"Transformer는 데이터가 엄청 많아야 한다"는 통념을 깨고, 일반 연구실·스타트업도 Vision Transformer를 쓸 수 있게 만들었습니다. 이후 대부분의 ViT 학습 레시피(timm 기본값 포함)의 기반이며, 지식 증류를 모델 아키텍처 수준에서 설계한 대표 사례입니다.

### 사전 지식
- ViT의 패치 임베딩과 class token 구조
- 지식 증류(Knowledge Distillation, soft/hard label)
- 데이터 증강 기법(Mixup, CutMix)과 정규화

### 관련 논문
- [An Image is Worth 16x16 Words / ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [DeiT III: Revenge of the ViT (Touvron et al., 2022)](https://arxiv.org/abs/2204.07118)

### 실무 적용
제한된 사내 데이터로 비전 모델을 학습할 때 DeiT 레시피는 사실상 표준 출발점입니다. 또한 큰 모델(교사)을 작은 모델(학생)로 증류해 모바일·엣지 배포 비용을 줄이는 전략은 비전뿐 아니라 LLM 경량화에도 그대로 이어집니다.

---

## Paper 3 (Recent): Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model (Vim)
- **Authors:** Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang
- **Year:** 2024 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2401.09417
- **PDF:** [./vision-mamba-zhu-2024.pdf](./vision-mamba-zhu-2024.pdf)
- **Citation Count:** ~2,000+

### 요약
Mamba(선택적 상태공간모델)는 시퀀스 길이에 선형 비용이지만 단방향·위치 정보 부재라는 한계가 있어 이미지에 바로 쓰기 어려웠습니다. Vim은 양방향 SSM과 위치 임베딩을 결합한 순수 SSM 비전 백본으로, ImageNet·COCO·ADE20K에서 DeiT보다 높은 성능을 내면서 1248×1248 고해상도에서 2.8배 빠르고 GPU 메모리를 86.8% 절약했습니다.

### 핵심 기여
- Self-attention 없이 양방향 SSM만으로 구성한 범용 비전 백본 제안
- 패치 시퀀스에 위치 임베딩을 더해 공간 정보 인식 문제 해결
- 고해상도 입력에서 Transformer 대비 속도·메모리 효율의 큰 우위 입증

### 이 논문이 중요한 이유
ViT의 이차(quadratic) 어텐션 비용은 고해상도 의료·위성·문서 이미지와 비디오에서 병목입니다. Vim은 "Attention 이후"의 아키텍처 후보로 SSM을 비전에 본격 도입한 대표 논문이며, 오늘의 FPN(다중 스케일)·DeiT(효율적 학습)와 함께 "효율"이라는 축으로 비전 아키텍처 진화를 이해하게 해줍니다.

### 사전 지식
- 상태공간모델(SSM)과 Mamba의 selective scan 개념
- ViT/DeiT의 패치 토큰화 방식
- 연산 복잡도(O(N²) vs O(N))와 GPU 메모리 개념

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [VMamba: Visual State Space Model (Liu et al., 2024)](https://arxiv.org/abs/2401.10166)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone (Hatamizadeh & Kautz, 2024)](https://arxiv.org/abs/2407.08083)

### 실무 적용
병리 슬라이드, 위성 영상, 긴 비디오처럼 토큰 수가 매우 많은 입력에서 추론 비용을 줄이는 백본 후보로 검토할 수 있습니다. 다만 생태계(커널 최적화, 사전학습 가중치)는 아직 ViT 대비 작으므로, 실무에서는 하이브리드(MambaVision 등) 구조와 함께 벤치마크 후 선택하는 것이 현실적입니다.

---

## 추천 읽기 순서
1. **FPN** — 백본-넥-헤드 구조와 다중 스케일 피처 개념을 먼저 잡습니다.
2. **DeiT** — CNN에서 Transformer로 넘어가는 전환점과 효율적 학습·증류 레시피를 이해합니다.
3. **Vision Mamba** — DeiT를 베이스라인으로 비교하므로, DeiT 다음에 읽으면 "Attention의 비용 문제를 어떻게 푸는가"가 명확해집니다.

## 핵심 테이크어웨이
- 좋은 비전 모델은 "무엇을 보는가(백본)"만큼 "여러 스케일을 어떻게 합치는가(넥)"가 중요하다.
- 아키텍처의 한계는 학습 레시피와 증류로 상당 부분 메울 수 있다 — 데이터가 적다고 Transformer를 포기할 필요는 없다.
- 고해상도·긴 시퀀스 시대에는 정확도뿐 아니라 연산 복잡도가 아키텍처 선택의 핵심 기준이 된다.
- 질문해볼 것: 우리 서비스의 이미지 해상도·데이터 규모·지연 시간 요구사항에서 CNN, ViT, SSM 중 무엇이 최적인가?

## 다음 토픽과의 연결
다음 토픽은 **RNN, LSTM과 시퀀스 모델**입니다. 오늘 본 Vision Mamba의 핵심인 상태공간모델(SSM)은 RNN의 "순환 상태" 아이디어를 병렬 학습 가능하게 재해석한 것입니다. LSTM → Seq2Seq → Mamba로 이어지는 흐름을 보면, 비전과 시퀀스 모델링이 결국 같은 효율성 문제를 다른 방식으로 풀고 있음을 확인할 수 있습니다.
