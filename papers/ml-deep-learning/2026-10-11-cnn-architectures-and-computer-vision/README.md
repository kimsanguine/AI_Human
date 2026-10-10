# Daily AI Paper Recommendations

> **Date:** 2026-10-11
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** CNN Architectures and Computer Vision

---

## Paper 1 (Classic): Fully Convolutional Networks for Semantic Segmentation
- **Authors:** Jonathan Long, Evan Shelhamer, Trevor Darrell
- **Year:** 2015 (CVPR 2015 Best Paper Honorable Mention)
- **arXiv:** https://arxiv.org/abs/1411.4038
- **PDF:** [./fully-convolutional-networks-long-2015.pdf](./fully-convolutional-networks-long-2015.pdf)
- **Citation Count:** ~45,000+

### 요약
분류용 CNN(AlexNet, VGG, GoogLeNet)의 fully-connected 층을 1×1 컨볼루션으로 바꿔, 임의 크기 이미지를 입력받아 픽셀 단위 예측 맵을 출력하는 "완전 컨볼루션 네트워크(FCN)"를 제안합니다. 깊은 층의 의미 정보와 얕은 층의 위치 정보를 skip connection으로 합쳐(FCN-32s/16s/8s) end-to-end 학습만으로 당시 시맨틱 세그멘테이션 SOTA를 달성했습니다.

### 핵심 기여
- 분류 네트워크를 dense prediction 네트워크로 바꾸는 "convolutionalization" 개념 정립
- 학습 가능한 업샘플링(deconvolution / transposed convolution)으로 저해상도 특징맵을 원본 해상도로 복원
- 다중 스케일 skip 아키텍처로 "무엇(what)"과 "어디(where)"를 동시에 포착 — 이후 U-Net, DeepLab, Mask R-CNN 헤드의 원형

### 이 논문이 중요한 이유
"이미지 하나 → 라벨 하나"였던 CNN을 "픽셀 하나하나 → 라벨"로 확장한 전환점입니다. 오늘날 세그멘테이션, 깊이 추정, SAM 같은 dense 태스크의 디코더 구조는 모두 FCN이 정의한 문법(인코더 + 업샘플 + skip) 위에 서 있습니다. 또한 ImageNet 사전학습 가중치를 다른 태스크로 옮기는 전이학습의 대표 사례이기도 합니다.

### 사전 지식
- CNN 기본 연산(컨볼루션, 풀링, stride, receptive field)
- VGG/AlexNet 구조와 ImageNet 사전학습
- 픽셀 단위 cross-entropy 손실, IoU(mIoU) 평가 지표

### 관련 논문
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs (Chen et al., 2016)](https://arxiv.org/abs/1606.00915)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition / VGG (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)

### 실무 적용
배경 제거, 인물 누끼(아바타·더빙 영상의 화자 분리), 의료 영상 병변 분할, 자율주행 도로 영역 인식, 문서 레이아웃 분석 등 "영역"을 다루는 거의 모든 비전 기능의 출발점입니다. 실무에서는 FCN 자체보다 그 후손(U-Net, DeepLabv3+, SegFormer)을 쓰지만, 업샘플링 아티팩트·경계 품질 문제를 디버깅할 때 FCN의 skip/upsampling 설계를 이해하고 있으면 원인을 훨씬 빨리 찾을 수 있습니다.

---

## Paper 2 (Classic): A ConvNet for the 2020s (ConvNeXt)
- **Authors:** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
- **Year:** 2022 (CVPR 2022)
- **arXiv:** https://arxiv.org/abs/2201.03545
- **PDF:** [./convnext-convnet-for-2020s-liu-2022.pdf](./convnext-convnet-for-2020s-liu-2022.pdf)
- **Citation Count:** ~9,000+

### 요약
Vision Transformer가 CNN을 대체한다는 분위기 속에서, 표준 ResNet-50을 출발점으로 Swin Transformer의 설계 요소(학습 레시피, stage 비율, patchify stem, depthwise conv, inverted bottleneck, 7×7 대형 커널, LayerNorm, GELU 등)를 하나씩 "현대화"해 순수 ConvNet만으로 Swin과 동등하거나 더 나은 성능을 냈습니다. 결론은 "Transformer의 우위는 attention 자체보다 설계·학습 레시피의 차이에서 상당 부분 왔다"는 것입니다.

### 핵심 기여
- ResNet → ConvNeXt로 가는 단계별 ablation 로드맵: 각 변경이 정확도에 얼마나 기여하는지 정량화
- 순수 컨볼루션 구조로 ImageNet 87.8% top-1, COCO·ADE20K에서 Swin과 동급 이상 성능
- "아키텍처 vs 학습 레시피" 효과를 분리해 보는 실험 방법론 제시

### 이 논문이 중요한 이유
"새 아키텍처가 좋아 보이는 이유가 정말 구조 때문인가?"라는 질문에 답하는 방법을 보여주는 교과서적 논문입니다. AI 엔지니어는 신규 모델을 도입할 때 비교 조건(데이터, 증강, 스케줄, 정규화)을 맞추지 않으면 잘못된 결론을 내리기 쉬운데, ConvNeXt는 공정한 비교의 모범 사례입니다. 또한 ConvNeXt 블록은 F5-TTS, Vocos 등 오디오/음성 모델에도 백본으로 재사용되고 있어 모듈 5(TTS)와도 연결됩니다.

### 사전 지식
- ResNet(residual block, bottleneck) 구조
- ViT / Swin Transformer의 patchify, window attention 개념
- 학습 레시피 용어: AdamW, Mixup/CutMix, Stochastic Depth, Label Smoothing, LayerNorm vs BatchNorm

### 관련 논문
- [Deep Residual Learning for Image Recognition / ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (Woo et al., 2023)](https://arxiv.org/abs/2301.00808)

### 실무 적용
엣지/모바일·온프레미스 환경처럼 attention 커널 최적화가 어렵거나 고해상도 입력에서 메모리가 중요한 경우 ConvNeXt는 ViT보다 배포가 쉽습니다(ONNX/TensorRT 호환성 우수). 이미지 분류·검수, 썸네일 품질 스코어링, 세그멘테이션 백본으로 바로 쓸 수 있고, timm 라이브러리에서 사전학습 가중치를 한 줄로 불러올 수 있습니다. 제품 의사결정 관점에서는 "최신 = Transformer"라는 가정 대신 지연시간·비용·정확도의 트레이드오프로 백본을 고르라는 교훈을 줍니다.

---

## Paper 3 (Recent): Perception Encoder: The best visual embeddings are not at the output of the network
- **Authors:** Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, Christoph Feichtenhofer (Meta FAIR)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2504.13181
- **PDF:** [./perception-encoder-bolya-2025.pdf](./perception-encoder-bolya-2025.pdf)
- **Code:** https://github.com/facebookresearch/perception_models
- **Citation Count:** ~200+ (2025년 공개, 빠르게 증가 중)

### 요약
대규모 이미지-텍스트 대조학습(CLIP 계열)만으로 학습한 비전 인코더(PE)를 이미지 레시피 고도화 + 비디오 데이터 엔진으로 확장해, 제로샷 이미지·비디오 분류/검색에서 SOTA를 달성합니다. 핵심 발견은 "가장 좋은 범용 특징은 네트워크의 최종 출력이 아니라 중간 층에 있다"는 것이며, 이를 끌어내기 위해 언어 정렬(PE-lang, MLLM용)과 공간 정렬(PE-spatial, 검출·추적·깊이 등 dense 태스크용) 두 가지 방법을 제안합니다.

### 핵심 기여
- 대조학습 단일 목적함수로도 OCR·QA·검출·깊이 추정까지 통하는 범용 특징이 학습됨을 실증
- 중간 층 특징이 태스크별로 최적 지점이 다르다는 분석 → language / spatial alignment로 추출
- 모델(PE-Core / PE-Lang / PE-Spatial), 코드, 합성+사람 주석 비디오 데이터셋 공개

### 이 논문이 중요한 이유
"CLIP류는 검색용, DINO류는 dense 태스크용, MLLM은 별도 인코더"로 나뉘던 비전 백본 선택을 하나의 인코더 패밀리로 통합하려는 흐름의 대표 논문입니다. 오늘의 두 고전(FCN의 dense 예측, ConvNeXt의 공정한 아키텍처 비교)이 던진 질문 — "어떤 특징을, 어디서 꺼내 쓸 것인가" — 에 2025년식 답을 줍니다.

### 사전 지식
- CLIP의 대조학습(이미지-텍스트 InfoNCE) 개념
- ViT 구조와 layer별 특징(feature probing, linear probe)
- 멀티모달 LLM에서 비전 인코더 + projector + LLM 결합 방식

### 관련 논문
- [Learning Transferable Visual Models From Natural Language Supervision / CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [SigLIP 2: Multilingual Vision-Language Encoders (Tschannen et al., 2025)](https://arxiv.org/abs/2502.14786)

### 실무 적용
- **멀티모달 에이전트:** 화면 이해·문서 QA가 필요한 에이전트에서 PE-Lang을 비전 인코더로 쓰면 OCR/차트 이해 품질을 높일 수 있습니다.
- **영상 검색·태깅:** 제로샷 비디오 분류/검색 성능이 강해 AI 더빙·아바타 영상의 장면 검색, 자동 태깅에 바로 활용 가능합니다.
- **임베딩 설계 교훈:** 자체 모델에서도 "마지막 층 임베딩"만 쓰지 말고 중간 층을 probing해 태스크별 최적 층을 고르는 것이 저비용 성능 개선 포인트입니다.

---

## 추천 읽기 순서
1. **FCN (2015)** — CNN이 분류를 넘어 픽셀 단위 예측으로 확장되는 원리를 먼저 이해합니다. Section 3(convolutionalization, upsampling, skip)이 핵심.
2. **ConvNeXt (2022)** — ResNet→ConvNeXt 로드맵 그림(Figure 2)을 따라가며 "어떤 설계가 왜 효과가 있었는지"를 단계별로 확인합니다.
3. **Perception Encoder (2025)** — 앞의 두 논문을 바탕으로, 대규모 사전학습 인코더에서 특징을 어디서 어떻게 꺼내는지 봅니다. 분량이 길므로 Introduction + layer analysis + alignment 섹션 위주로 읽기를 권합니다.

## 핵심 테이크어웨이
- **Dense 예측의 문법은 FCN에서 시작됐다:** 인코더 → 업샘플 → skip 결합 구조는 10년이 지나도 세그멘테이션·깊이·SAM 디코더의 기본형입니다.
- **아키텍처보다 레시피가 더 클 때가 많다:** ConvNeXt는 공정한 비교 없이는 "새 구조가 더 좋다"는 결론을 믿지 말라는 교훈을 줍니다.
- **최고의 특징은 출력층에 없을 수 있다:** PE는 임베딩을 꺼낼 위치 자체가 하이퍼파라미터임을 보여줍니다.
- **질문으로 정리:** "우리 서비스의 비전 기능은 이미지 단위 판단인가, 픽셀 단위 판단인가? 그리고 그 판단에 필요한 특징은 백본의 어느 층에 있는가?"

## 다음 토픽과의 연결
다음 토픽은 **RNN, LSTM and Sequence Models**입니다. 오늘 본 CNN/ViT가 "공간" 구조를 다뤘다면, 다음은 "시간·순서" 구조를 다룹니다. 특히 PE가 이미지 인코더를 비디오로 확장한 방식(프레임 시퀀스 처리)은 시퀀스 모델링 문제와 직결되며, ConvNeXt처럼 "attention 없이도 경쟁력 있는 구조"라는 질문은 Mamba·State Space Model 논의로 그대로 이어집니다.
