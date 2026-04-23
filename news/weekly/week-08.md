# AI Human Weekly Digest — Week 08 (2026-04-20 ~ 2026-04-24)

**이번 주 범위:** Day 34 ~ Day 38
**현재 챕터:** Ch04 자연어 및 음성 데이터 (Transformer·임베딩·음성 데이터)

---

## THIS WEEK'S TOP 3

### 1. 'AI 2강'의 축이 바뀐 한 주 — Claude Opus 4.7 등판 + Anthropic 세컨더리 $1T, OpenAI 추월
월 화요일 공개된 Claude Opus 4.7은 에이전틱 코딩·도구 사용 벤치마크에서 GPT-5.4·Gemini 3.1 Pro를 모두 앞질렀고, 금요일에는 세컨더리 시장에서 Anthropic 가치가 $1T에 도달해 OpenAI($880B)를 추월했다. Claude Code 기반 연환산 매출이 $9B → $30B로 급증한 것이 핵심 동력이다. Sam Altman의 "공포 마케팅" 공개 비판까지 겹치며 두 회사의 경쟁이 가장 극적으로 드러난 주였다.

### 2. 음성 AI가 'API 생태계'로 본격 진입 — xAI Grok STT/TTS, Gemini 3.1 Flash TTS, MAI-Voice, 그리고 보안 이슈
xAI가 Grok STT/TTS API(통화 인식 에러율 5%)를 단독 출시하고, Google은 Gemini 3.1 Flash TTS(200+ 오디오 태그)로 '연출 가능한 TTS' 시대를 열었으며, Microsoft MAI-Voice/Transcribe-1과 Google 온디바이스 'AI Edge Eloquent'까지 음성 스택이 다층화됐다. 동시에 음성 API 확산이 화자 인증(voice biometrics)의 새 공격 표면을 만든다는 경고(Biometric Update)와, Indic 언어 코덱 딥페이크 탐지(Indic-CodecFake/SATYAM, ACL 2026) 연구가 같은 주에 나오며 보안·탐지 이슈가 동반 부상했다.

### 3. 한국/한국어 AI의 존재감 — Stanford AI Index 8개 모델 정정 + 한국어 최적화 연구 러시
Stanford HAI가 AI Index 2026 한국 모델 수를 5 → 8로 정정해 EXAONE 4.0/Path 2.0/Deep, Solar Open 100B, VARCO, A.X K1, HyperCLOVA X SEED 32B Sync가 공식 반영됐다. 한편 arXiv에서는 한국어 중심 LLM을 위한 Token Pruning 연구(Qwen3·Gemma-3·Llama-3·Aya 벤치마크), KR-ELECTRA-Small-KD(지식 증류+저랭크 분해로 교사 성능 97.4% 유지), 한국어-영어 Code-switching Whisper 확장 등이 연달아 발표되며 "한국어 = 작고 똑똑한 모델"의 서사가 굳어지고 있다.

---

## 이번 주 핵심 키워드

- **에이전틱 코딩·Agent Platform** — Claude Opus 4.7, Gemini Enterprise Agent Platform, Claude Code, OpenAI Codex 강화
- **음성 스택의 모듈화** — STT/TTS API 분리 출시, 온디바이스 ASR, 화자 인증 보안
- **한국어 특화 NLP** — Token Pruning, 지식 증류, Code-switching, 30B급 소버린 모델
- **추론 전용 인프라** — Google Ironwood TPU / TPU 8i, Amazon-Anthropic $25B+50B, Nvidia 경쟁 구도
- **World Model의 부상** — Tencent HY-World 2.0, Alibaba Happy Oyster, NVIDIA Lyra 2.0, Bloomberg의 "ChatGPT가 못 잡는 것"
- **로보틱스·피지컬 AI** — Physical Intelligence π0.7
- **투자·자금** — Anthropic $1T, Sequoia $7B 신규펀드, Loop $95M Series C

---

## 다음 주 프리뷰 (Week 09: 2026-04-27 ~ 05-01, Day 39~43)

- **Ch04 후반부 진입:** 이번 주까지 임베딩·Transformer를 다뤘다면, 다음 주는 음성 데이터(MFCC·멜스펙트로그램)와 한국어 형태소·품사 태깅 실습 중심으로 넘어간다.
- **지켜볼 릴리스 포인트:**
  - 5월 초 I/O 2026 예고 기조에서 Google이 어떤 음성/멀티모달 기능을 공개할지
  - Anthropic의 '엔터프라이즈 Routines' 정식 전환 여부와 요금제 발표
  - NeurIPS 2026 CFP 마감(5월 초) 전후 음성·NLP arXiv 투고량 급증 구간
- **학습 체크포인트:** Ch04 중간 과제로 (1) 뉴스 1건을 한국어 Transformer·토크나이저로 전처리 → (2) 화자 임베딩 또는 문장 임베딩으로 유사 기사 3건을 찾고 → (3) 그 결과의 편향을 서술하는 미니 프로젝트를 권장한다.

---
**김생근** | AI Human 튜터
AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저. AI Dubbing·Avatar·Agentic AI 제품을 리딩하며 AI 네이티브 사고를 실무에 적용하고 있습니다.
