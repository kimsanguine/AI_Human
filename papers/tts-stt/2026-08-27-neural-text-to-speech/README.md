# Daily AI Paper Recommendations

> **Date:** 2026-08-27
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): Neural Speech Synthesis with Transformer Network
- **Authors:** Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, Ming Liu, Ming Zhou
- **Year:** 2019 (arXiv 2018)
- **arXiv:** https://arxiv.org/abs/1809.08895
- **PDF:** [./transformer-tts-li-2019.pdf](./transformer-tts-li-2019.pdf)
- **Citation Count:** 약 1,100+

### 요약
Tacotron 2의 RNN 기반 인코더/디코더와 location-sensitive attention을 Transformer의 multi-head self-attention으로 완전히 대체한 첫 번째 대표적 TTS 논문입니다. 음소(phoneme) 시퀀스를 입력받아 mel-spectrogram을 생성하고, WaveNet 보코더로 최종 파형을 만듭니다. 학습 속도를 약 4.25배 끌어올리면서도 MOS 4.39로 Tacotron 2와 동등한 자연스러움을 달성했습니다.

### 핵심 기여
- RNN을 제거하고 self-attention으로 인코더/디코더 hidden state를 **병렬 구성** → 학습 시간 대폭 단축
- 임의의 두 시점을 직접 연결하는 self-attention으로 **long-range dependency 문제 해결** (긴 문장 운율 안정화)
- TTS에 맞춘 구조적 적응: Scaled Positional Encoding(학습 가능한 스케일), Encoder/Decoder Pre-net, Mel Linear + Stop Linear + Post-net 설계

### 이 논문이 중요한 이유
오늘날의 거의 모든 TTS 시스템(FastSpeech, VALL-E, CosyVoice, Spark-TTS)은 Transformer 백본 위에 서 있습니다. 이 논문은 "TTS에 Transformer를 어떻게 이식하는가"라는 문제를 처음으로 체계적으로 풀어낸 기준점입니다. AI 엔지니어 입장에서는 *왜 TTS 학습이 느렸는가*, *어떤 구조 변경이 그 병목을 제거했는가*를 이해하는 것이 이후의 비자기회귀(non-autoregressive) TTS와 LLM 기반 TTS 계보를 읽는 전제 조건이 됩니다. 또한 attention alignment가 붕괴하면 발음이 반복·누락되는 TTS 특유의 실패 모드를 이 논문에서 처음 정면으로 다룹니다.

### 사전 지식
- Transformer 구조: multi-head self-attention, positional encoding, residual + layer norm
- Tacotron 2의 seq2seq TTS 파이프라인 (텍스트 → mel-spectrogram → 보코더 → 파형)
- Mel-spectrogram, STFT 등 음성 신호 표현의 기초
- Teacher forcing과 autoregressive 디코딩, 그리고 학습/추론 불일치(exposure bias)
- WaveNet 보코더의 역할과 추론 비용

### 관련 논문
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions / Tacotron 2 (Shen et al., 2018)](https://arxiv.org/abs/1712.05884)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [FastSpeech: Fast, Robust and Controllable Text to Speech (Ren et al., 2019)](https://arxiv.org/abs/1905.09263)
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)

### 실무 적용
상용 TTS 서비스의 학습 파이프라인 설계에 직접적으로 쓰입니다. 오디오북·더빙 서비스처럼 긴 문장을 다루는 도메인에서는 RNN 기반 모델의 alignment 붕괴가 치명적 품질 이슈(단어 반복, 문장 끝 잘림)로 나타나는데, self-attention 백본이 이를 완화합니다. 다만 이 모델은 여전히 autoregressive라 실시간 스트리밍에는 지연이 큽니다 — 실무에서는 이 논문의 백본 구조는 계승하되, duration predictor 기반 병렬 디코딩(FastSpeech 계열)으로 교체하는 것이 표준 선택입니다.

---

## Paper 2 (Classic): NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers
- **Authors:** Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, Jiang Bian
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2304.09116
- **PDF:** [./naturalspeech2-shen-2023.pdf](./naturalspeech2-shen-2023.pdf)
- **Citation Count:** 약 600+

### 요약
VALL-E류의 discrete token + autoregressive 언어모델 방식이 갖는 불안정성(토큰 오류 누적, 발음 누락, 느린 추론)을 정면으로 지적하고, **연속(continuous) latent 벡터 위의 latent diffusion**으로 방향을 튼 논문입니다. Residual Vector Quantizer가 만든 latent를 diffusion 모델이 비자기회귀적으로 예측하며, in-context learning으로 3초 프롬프트만으로 화자를 복제합니다. 44K시간 데이터로 학습해 zero-shot 음성뿐 아니라 **노래(singing) 합성까지 하나의 모델로** 처리합니다.

### 핵심 기여
- Discrete token 대신 **연속 latent + latent diffusion**을 채택해 양자화 오차와 토큰 누적 오류를 제거
- 완전 비자기회귀(NAR) 구조로 **추론 안정성과 강건성(robustness)** 확보 — VALL-E 대비 word error rate를 크게 낮춤
- Speech prompting 메커니즘으로 duration/pitch predictor와 diffusion 모델 모두에 in-context learning 부여
- 음성과 노래를 통합 학습해, 말하기 프롬프트만으로 노래를 합성하는 **zero-shot singing** 능력 시연

### 이 논문이 중요한 이유
2023년 이후 TTS 연구는 크게 두 갈래 — (1) LLM 스타일 discrete token 자기회귀(VALL-E, CosyVoice, Spark-TTS)와 (2) diffusion/flow-matching 기반 비자기회귀(NaturalSpeech 2/3, Voicebox, F5-TTS) — 로 갈라졌습니다. 이 논문은 두 번째 갈래의 출발점이자 가장 명확한 문제 제기입니다. AI 엔지니어에게 필독인 이유는 **"discrete냐 continuous냐"라는 표현 공간 선택이 품질이 아니라 강건성·지연시간·제어성의 트레이드오프**라는 사실을 데이터로 보여주기 때문입니다. 이 트레이드오프를 이해하면 제품 요구사항(실시간성 vs 안정성)에서 아키텍처를 역산할 수 있습니다.

### 사전 지식
- Diffusion 모델의 기본 원리 (forward noising / reverse denoising, DDPM)
- Latent diffusion (Stable Diffusion) 개념 — 픽셀/파형이 아닌 latent 공간에서 denoising
- Neural audio codec과 Residual Vector Quantization (SoundStream, EnCodec)
- VALL-E 방식의 neural codec language model과 그 실패 모드
- Duration/pitch predictor 등 variance adaptor (FastSpeech 2)

### 관련 논문
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models (Ju et al., 2024)](https://arxiv.org/abs/2403.03100)
- [Voicebox: Text-Guided Multilingual Universal Speech Generation (Le et al., 2023)](https://arxiv.org/abs/2306.15687)
- [High-Fidelity Neural Audio Compression / EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438)

### 실무 적용
AI 더빙·오디오북 제작처럼 **한 번 생성해서 오래 쓰는(offline) 고품질 요구** 도메인에 적합한 아키텍처입니다. 자기회귀 모델은 긴 대본에서 간헐적으로 발음 누락이 발생해 QA 비용이 커지는데, NAR diffusion은 이 실패율이 낮아 사람 검수 부담이 줄어듭니다. 반대로 diffusion 스텝 수만큼 지연이 생기므로 실시간 대화형 에이전트에는 부적합하며, 실무에서는 스텝 수 축소(distillation)나 flow matching으로 전환해 대응합니다. 또한 3초 프롬프트로 화자를 복제하는 구조는 **동의·초상권 검증 절차를 제품 플로우에 반드시 넣어야 함**을 의미합니다.

---

## Paper 3 (Recent): Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with One-Stream Decoupled Speech Tokens
- **Authors:** Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.01710
- **PDF:** [./spark-tts-wang-2025.pdf](./spark-tts-wang-2025.pdf)
- **Citation Count:** 약 90+ (2026년 8월 기준, 빠르게 증가 중)

### 요약
기존 코덱 기반 TTS가 acoustic token과 semantic token을 각각 다루느라 여러 모델을 이어붙여야 했던 복잡성을 제거하고, **단일 LLM(Qwen2.5)이 하나의 토큰 스트림만으로 음성을 직접 생성**하도록 만든 논문입니다. 핵심은 BiCodec — 음성을 (1) 언어 내용을 담는 저비트레이트 semantic token과 (2) 화자 특성을 담는 고정 길이 global token으로 **분리(decouple)** 하는 코덱입니다. 여기에 100,000시간 규모의 주석 데이터셋 VoxBox를 함께 공개했습니다.

### 핵심 기여
- **BiCodec**: 음성을 semantic token(내용)과 global token(음색·화자)으로 분해하여, 단일 스트림으로도 zero-shot 복제와 속성 제어를 동시에 지원
- **단일 LLM 아키텍처**: 별도 acoustic 생성 모델·flow matching 모듈 없이 Qwen2.5 하나로 처리 → 파이프라인 단순화와 추론 효율 향상
- **Coarse + Fine 이중 음성 제어**: 성별·발화 속도·피치 같은 속성을 자연어 라벨로 거친 제어 후, 세밀한 수치 값으로 미세 조정 (기존 프롬프트 복제 방식과 달리 참조 음성 없이도 새 화자 창조 가능)
- **VoxBox 공개**: 100K시간, 속성 주석이 포함된 대규모 오픈 데이터셋으로 재현성 확보 (zero-shot UTMOS 4.35 보고)

### 이 논문이 중요한 이유
2024~2025년 TTS의 흐름은 "전용 음성 모델"에서 "LLM에 음성 토큰을 먹이는 방식"으로 이동했고, Spark-TTS는 그 흐름에서 **가장 단순한 형태의 레퍼런스 구현**입니다. AI 엔지니어에게 중요한 지점은 세 가지입니다. 첫째, 텍스트 LLM 생태계(양자화, LoRA, vLLM 서빙, 스트리밍 디코딩)를 그대로 재사용할 수 있어 **엔지니어링 비용이 급감**합니다. 둘째, 토큰을 내용/음색으로 분리한다는 설계는 음성 이외의 멀티모달 생성에도 반복되는 패턴입니다. 셋째, 참조 음성 없이 속성만으로 화자를 만들어낸다는 점은 **음성 복제의 법적·윤리적 리스크를 우회하는 실용적 대안**을 제시합니다.

### 사전 지식
- Neural audio codec과 VQ/RVQ, semantic vs acoustic token의 차이
- Decoder-only LLM의 next-token prediction과 텍스트/오디오 토큰 인터리빙
- VALL-E, CosyVoice 등 코덱 언어모델 TTS 계보
- Zero-shot voice cloning과 speaker embedding의 개념
- UTMOS, WER 등 TTS 평가 지표의 의미와 한계

### 관련 논문
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models (Du et al., 2024)](https://arxiv.org/abs/2412.10117)
- [Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis (Ye et al., 2025)](https://arxiv.org/abs/2502.04128)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)
- [SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312)

### 실무 적용
LLM 서빙 인프라를 이미 운영 중인 팀이라면 TTS를 **별도 스택이 아니라 기존 LLM 서빙 위의 한 모델로 편입**시킬 수 있습니다. 이는 GPU 활용률, 배포 파이프라인, 모니터링 도구를 공유한다는 뜻이고 실질적인 운영 비용 절감으로 이어집니다. 속성 기반 화자 생성은 게임 NPC, 광고 나레이션, 캐릭터 보이스처럼 **"특정 사람의 목소리를 쓰면 안 되지만 다양한 목소리는 필요한"** 제품에 바로 맞습니다. 반면 단일 스트림 저비트레이트 구조는 감정 표현의 미세한 뉘앙스에서 flow-matching 계열보다 불리할 수 있으므로, 감정 연기가 중요한 더빙 제품이라면 A/B로 검증한 뒤 도입하는 것이 안전합니다.

---

## 추천 읽기 순서

1. **Transformer TTS (2019)** — 먼저 읽습니다. TTS가 왜 seq2seq 문제이고, attention alignment가 어떻게 깨지는지에 대한 직관을 얻는 것이 나머지 두 논문의 전제입니다. Tacotron 2와 나란히 놓고 "무엇이 바뀌었나"만 추적하면 빠르게 읽힙니다.
2. **NaturalSpeech 2 (2023)** — 다음으로 읽습니다. 여기서 처음 "표현 공간을 discrete로 갈 것인가 continuous로 갈 것인가"라는 진짜 아키텍처 결정이 등장합니다. 논문의 VALL-E 비판 부분을 특히 꼼꼼히 보세요.
3. **Spark-TTS (2025)** — 마지막으로 읽습니다. 2번에서 비판당한 discrete 진영이 어떻게 진화해서 답했는지를 보는 셈이고, 현재 오픈소스 TTS의 실질적 출발점이기도 합니다.

## 핵심 테이크어웨이

- **TTS 아키텍처의 역사는 "속도 vs 안정성" 트레이드오프의 역사입니다.** RNN → Transformer(학습 병렬화), AR → NAR(추론 안정화), 다단계 파이프라인 → 단일 LLM(운영 단순화). 각 전환은 품질 개선이라기보다 **병목의 이동**이었습니다.
- **표현 공간 선택(discrete token vs continuous latent)이 제품 특성을 결정합니다.** Discrete는 LLM 생태계 재사용과 스트리밍에 유리하고, continuous diffusion은 강건성과 자연스러움에 유리합니다. "어느 쪽이 SOTA인가"가 아니라 "우리 제품이 실시간인가, 오프라인 고품질인가"로 물어야 합니다.
- **Decoupling(분리)은 반복되는 설계 패턴입니다.** Spark-TTS의 semantic/global 토큰 분리, NaturalSpeech 3의 factorized codec 모두 "내용과 스타일을 나눠서 각각 제어한다"는 같은 아이디어입니다. 제어 가능성(controllability)은 대개 이 분리에서 나옵니다.
- **평가 지표를 의심하세요.** MOS/UTMOS는 자연스러움을 재지만 발음 누락은 잡지 못합니다. 실무에서는 WER(또는 CER)을 반드시 함께 측정해야 하며, 긴 문장에서의 실패율이 진짜 제품 품질 지표입니다.
- **화자 복제 기술은 곧 컴플라이언스 요구사항입니다.** 3초 프롬프트 zero-shot 복제가 표준이 된 순간, 동의 확인·워터마킹·오남용 탐지는 선택이 아니라 제품 기능입니다.

## 다음 토픽과의 연결

다음 토픽인 **Voice Cloning and Speech Synthesis**는 오늘 다룬 세 논문이 공통으로 건드린 "3초 프롬프트로 화자를 복제한다"는 능력 자체를 정면으로 파고듭니다. 오늘은 그 능력을 *아키텍처의 부산물*로 봤다면, 다음에는 그것을 *목표 함수*로 놓고 화자 유사도(speaker similarity), 감정 전이, 크로스링구얼 복제, 그리고 워터마킹·오남용 방지까지 다루게 됩니다. 오늘 정리한 discrete/continuous 트레이드오프를 손에 쥐고 가면, VALL-E 계열과 flow-matching 계열이 voice cloning 품질에서 왜 다르게 실패하는지 훨씬 빨리 읽힙니다.
