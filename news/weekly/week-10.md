# AI Human Weekly Digest — Week 10 (2026-05-04 ~ 2026-05-08)

**커리큘럼 구간:** Day 44~48 / 100 · Ch04 자연어 및 음성 데이터 (Day 27~53)
**핵심 학습 주제:** Transformer 구조 · Attention 메커니즘 · 임베딩(텍스트·멀티모달) · 토크나이저 · 음성 데이터 입출력
**총 다룬 콘텐츠:** 25건 (📰 21 · 💻 2 · 📄 2)

---

## TOP 3 — 이번 주 가장 중요한 시그널

### 1. 음성 AI의 패러다임이 한 주 안에 두 번 바뀌었다
- **OpenAI GPT-Realtime-2/Translate/Whisper 동시 출시 (5/7)** — 음성을 듣고-추론하고-번역하고-기록하는 단일 파이프라인
- **Inworld Realtime TTS-2 (5/5)** — 텍스트가 아닌 "이전 턴의 오디오 자체"를 입력으로 받는 closed-loop 모델
- 2주 뒤 Ch05 TTS·STT 학습 진입 직전, 시장이 이미 "STT → LLM → TTS의 모듈 분리" 자체를 해체하고 있음. 학습 전제부터 다시 짜야 한다.

### 2. 멀티모달 임베딩이 인프라 표준으로 굳어진 한 주
- **Google Gemini Embedding 2 정식 GA (5/5, 5/6)** — 텍스트·이미지·오디오·영상·PDF를 단일 임베딩 공간에, MTEB 1위
- **BigQuery + EmbeddingGemma-300m 통합** — 데이터 웨어하우스 안에서 임베딩이 발생
- **EPIC 논문 (5/6)** — LLM을 텍스트 인코더로 만드는 in-context prompt 학습
- 임베딩이 "별도 모델 → 데이터 인프라 내장 컴포넌트"로 이동. RAG 구간(Ch09, Day 91~)에 도달할 때쯤 임베딩은 이미 코모디티가 되어 있을 것.

### 3. 한국 AI 산업 지형이 한 주 만에 재편됐다
- **삼성전자 1Q 사상 최대 실적 — 매출 133.9조·영업익 57.2조 (5/5)**, AI 반도체 견인
- **AI EXPO KOREA 2026 코엑스 개막 (5/6~5/8)** — 350개사·600부스
- **모레(MOREH) × 텐스토렌트 갤럭시 웜홀 추론 검증 (5/7)** — 국산 스택의 NVIDIA 외 옵션 확보
- **업스테이지, 다음(AXZ) 인수 최종 확정 (5/7)** — Solar LLM × 다음 검색DB로 '한국형 Perplexity'
- 칩(삼성·모레)–모델(업스테이지 Solar)–서비스(다음 검색)가 한 주에 모두 움직였다. 한국 AI는 더 이상 "추격자" 포지션이 아니라 자체 스택을 갖춘 플레이어로 재정의되고 있다.

---

## 핵심 키워드 (이번 주 5회 이상 등장)

| 키워드 | 빈도 | 의미 |
|---|---|---|
| **임베딩 / Embedding** | 7 | Gemini 2, EmbeddingGemma, EPIC, MTEB — 멀티모달 표준 경쟁 |
| **에이전트 / Agent** | 6 | Microsoft Agent 365, Sierra $950M, Anthropic Wall Street, Writer 자율 에이전트 |
| **음성 / TTS / STT** | 5 | OpenAI Realtime, Inworld TTS-2 |
| **Transformer / Attention** | 5 | Subquadratic SubQ, Attention Sinks 논문, HF transformers, Mistral Medium 3.5 |
| **자본 / Funding** | 5 | Sierra $950M, Anthropic·OpenAI JV, Nebius-Eigen $643M, NVIDIA-IREN $2.1B |
| **한국 AI** | 5 | 삼성 1Q, AI EXPO KOREA, 모레-텐스토렌트, 업스테이지-다음 |

---

## 학습 연결 — 이번 주 배운 것이 어떻게 뉴스에 박혀 있나

| 학습 주제 (Day 44~48) | 이번 주 시그널 | 다음 주 학습에 어떻게 이어지나 |
|---|---|---|
| Transformer 구조 | Mistral Medium 3.5(128B 덴스), Qwen 3.6-Plus, Subquadratic SubQ | Day 49~53에서 다룰 BERT 계열 vs GPT 계열 비교의 자본·아키텍처 배경 |
| Attention 메커니즘 | Attention Sinks 논문 (cross-attention 83~91% 흡수), Subquadratic 대안 | Day 50 실습: "어텐션 시각화"에서 EOS·구두점이 흡수자 역할을 하는지 직접 확인 |
| 임베딩 | Gemini Embedding 2 / EmbeddingGemma-300m / EPIC | Ch08 Langchain(Day 79~) 진입 시 vector DB 선택 기준의 상수가 됨 |
| 토크나이저 | HF tokenizers 라이브러리, 업스테이지 Solar 한국어 토크나이저 | Day 49 실습: Solar vs Llama vs BERT 토크나이저로 같은 한국어 문장을 인코딩해 어휘 압축률 비교 |
| 음성 데이터 | OpenAI Realtime, Inworld TTS-2 | Day 54~63 Ch05 TTS·STT의 실전 레퍼런스. "텍스트 vs 오디오 토큰" 통합 트렌드를 미리 학습 |

---

## 다음 주(Week 11) 프리뷰 — Day 49~53

**학습 구간:** Ch04 자연어 및 음성 데이터 (마지막 주)
**핵심 주제:** BERT 계열 모델 실습, 한국어 토크나이저 비교, 음성 데이터 전처리(MFCC, Mel-spectrogram)

**관전 포인트 3가지**

1. **OpenAI Realtime-2 vs Inworld TTS-2 직접 비교** — 동일한 한국어 문장(예: 사용자 클레임 응대 시나리오)을 두 API에 넣어보면 "추론 결합형 음성"과 "감정 인식형 음성"이 각자 어디에서 강한지 체감할 수 있다. 학습 + 실무 도입 검토를 같이 가져갈 수 있는 구간.

2. **Solar LLM의 다음 검색 통합 진척도** — 업스테이지 인수가 5/7 클로징된 만큼, 다음 주 중 첫 번째 통합 데모나 Solar 신버전 발표 가능성이 있다. 한국어 토크나이저 학습과 직결.

3. **Mythos·GPT-5.5 Cyber 후속 행정명령 동향** — 백악관의 신규 AI 모델 사전 검토 행정명령(5/6 보도) 관련 후속 발표가 다음 주 중 나올 가능성. 미국 AI 거버넌스가 "행정 명령 → 사전 검토"로 이동하면 글로벌 AI 회사의 학습·배포 속도에 직접 영향.

**준비 액션**
- HF `tokenizers` 라이브러리 환경 세팅 (Day 49 실습 직전)
- OpenAI Realtime API 크레딧 확인 (Day 54 진입 시 곧바로 실습 가능하게)

---

**김생근** | AI Human 튜터
이번 주는 "음성 AI 패러다임 전환 + 한국 AI 자체 스택 가시화"가 동시에 일어난, 100일 과정 중에서도 손에 꼽힐 1주였습니다. 다음 주 학습은 이 흐름 위에 BERT 실습을 얹어, "임베딩 모델 직접 학습 → 음성 인터페이스 진입"으로 자연스럽게 이어집니다.
