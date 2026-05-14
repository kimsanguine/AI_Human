# AI Human Weekly Digest — Week 11 (2026-05-11 ~ 05-15)

**현재 진도:** Ch04 자연어 및 음성 데이터 마지막 주 (Day 49~53/100)
**다음 주(Week 12):** Ch05 — TTS·STT 본격 진입 (Tacotron, VITS, Whisper, CTC)
**이번 주 콘텐츠 합계:** 25건 (📰 19 · 💻 4 · 📄 2)

---

## TOP 3 — 이번 주 가장 중요한 신호

### 🥇 1. 임베딩 모델 지각변동: Google Gemini #1, Alibaba Qwen3-Embedding 추격 + 카카오 카나나-v
**왜 중요:** 이번 주 임베딩 진영이 한꺼번에 흔들렸다. 폐쇄 진영(Google Gemini Embedding)이 리더보드 1위를 가져갔고, **Alibaba가 Apache 2.0으로 Qwen3-Embedding을 풀면서 오픈소스 격차가 거의 사라졌다**. 같은 주에 **카카오가 카나나-v-임베딩**으로 "경복궁·붕어빵·오타까지" 잡는 한국 문화 특화 임베딩을 공개. Ch04에서 배운 임베딩이 단순한 학습 개념이 아니라 **국가/기업 단위 경쟁의 무대**라는 걸 보여준 한 주.
**관련 일자:** 5/12, 5/13

### 🥈 2. 음성 AI 자본 시장 폭발: Wispr $2B, Vapi $500M, 그리고 Inworld TTS-2의 closed-loop
**왜 중요:** STT(Wispr 받아쓰기)·음성 에이전트 오케스트레이션(Vapi)이 6개월 만에 2~3배 점프했고, Inworld는 **"텍스트가 아니라 오디오 자체"를 입력으로 받는** 차세대 TTS 아키텍처(TTS-2)를 공개. 다음 주 Ch05 진입을 앞두고 "왜 지금 음성 데이터가 가장 뜨거운가?"의 답을 자본 시장이 직접 알려주고 있다.
**관련 일자:** 5/8, 5/12, 5/13

### 🥉 3. 에이전트 시대의 새로운 BOM: "토큰 = 원가"가 본격적으로 비즈니스 모델을 깨다
**왜 중요:** 금주 마지막 날 Anthropic이 Claude 구독제에서 **에이전트 사용분을 별도 크레딧으로 분리**했다. ServiceNow·Uber가 연간 AI 예산을 이미 소진한 상황과 맞물려, "구독제 무한 사용" 모델이 에이전트 시대에서 깨지고 있다는 신호. **토크나이저 효율·임베딩 차원 압축**이 단순 NLP 디테일이 아니라 product margin 결정자가 됐다.
**관련 일자:** 5/14

---

## 이번 주 핵심 키워드

| 카테고리 | 키워드 | 등장 횟수 |
|---|---|---|
| **임베딩** | Gemini Embedding, Qwen3, 카나나-v, sentence-transformers v5.4 | 6+ |
| **음성 AI** | Wispr, Vapi, Inworld TTS-2, sherpa-onnx, OpenAI Realtime | 5+ |
| **에이전트 비즈니스** | Cloudflare 정리해고, Anthropic 크레딧 분리, Codex 모바일, JPMorgan AI 인프라화 | 5+ |
| **LLM·인프라** | Gemini 3.1 Flash-Lite, Karpathy LLM Wiki, DeepSeek $45B, AI 데이터센터 IPO | 4+ |
| **오픈소스/툴** | huggingface/transformers v5.8.1, sentence-transformers v5.4, sherpa-onnx, Qwen3-Embedding, tokenizers | 5+ |

---

## 다음 주(Week 12) 프리뷰 — Ch05 TTS/STT 진입

다음 주 학습은 **음성 합성(Tacotron, VITS)**과 **음성 인식(Whisper, CTC)**에 들어간다. 미리 머리에 넣어두면 좋은 산업 컨텍스트:

1. **closed-loop 음성 모델 등장**: Inworld TTS-2처럼 텍스트가 아닌 "오디오 → 오디오" 파이프라인이 새 표준 후보. Tacotron2의 mel-spectrogram 기반 파이프라인과 비교해서 볼 것.
2. **온디바이스 음성 풀스택**: sherpa-onnx 같은 툴킷이 Whisper·Cohere Transcribe까지 묶어 모바일에서 STT를 돌린다. CTC loss 학습 후 양자화(INT8) → 모바일 배포 흐름이 이번 분기 표준화될 가능성.
3. **음성 STT의 자본 시장 검증**: Wispr $2B, Deepgram $1.3B(연초). 우리가 배울 Whisper가 어떻게 상용 STT 서비스의 베이스라인이 됐는지 보면서 "오픈소스 → 서비스화" 패턴을 익히자.
4. **음성 에이전트 인프라**: Vapi의 "1B 통화 처리"는 단순 STT·TTS가 아니라 **STT → LLM → TTS** 풀스택 오케스트레이션. Ch05를 마치고 Ch08(LangChain)로 가는 다리 역할.

---

## Friday Question
> "임베딩이 오픈소스로 평준화되고, 음성 모델은 closed-loop으로 진화하고, 에이전트는 토큰을 폭발적으로 태운다 — 이 세 흐름이 동시에 일어난 이번 주를 한 문장으로 요약한다면, 우리 제품의 다음 분기 최우선 의사결정은 무엇이어야 할까?"

---
**김생근** | AI Human 튜터
