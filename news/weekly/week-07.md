# AI Human 주간 다이제스트 — Week 07 (2026-04-13 ~ 04-17) | Day 29~33/100

**이번 주 진도:** Ch04 자연어 및 음성 데이터 — Transformer, 어텐션, 토크나이저, 임베딩

---

## 이번 주 TOP 3 뉴스

### 1. Anthropic Mythos — 사이버 보안 패러다임을 뒤흔든 10조 파라미터 모델
Anthropic의 Mythos가 모든 주요 OS·브라우저의 제로데이 취약점을 자동 발견하면서 글로벌 금융계를 뒤흔들었다. 미 재무장관·연준 의장의 월가 CEO 긴급 소집에서 시작해, 주말을 넘기며 영란은행(BoE)까지 공식 경고를 발표했고, Project Glasswing은 영국 금융기관으로 확대됐다. Transformer의 self-attention이 코드 취약점 패턴을 자동 인식할 수 있다는 점은 NLP 기술의 영역 확장을 상징한다.

### 2. 어텐션 효율화 3종 경쟁 — HKT, DSA, Log-Linear Attention
이번 주에만 Self-Attention의 O(n²) 병목을 해결하는 세 가지 접근법이 동시에 주목받았다. 계층적 커널 트랜스포머(HKT)는 다중 해상도 다운샘플링으로 1.3배 비용, DeepSeek의 스파스 어텐션(DSA)은 선택적 토큰 처리로 70% 비용 절감, ICLR 2026의 Log-Linear Attention은 커널 함수로 O(n log n)을 달성했다. 어텐션 최적화가 2026년 AI 연구의 핵심 전장임을 확인한 한 주였다.

### 3. 음성 토크나이저 경쟁 본격화 — Qwen3-TTS(12Hz) vs Voxtral Transcribe 2(온디바이스)
Alibaba의 Qwen3-TTS가 초당 12토큰만으로 고음질 TTS를 구현하는 음성 토크나이저를 선보였고, Mistral의 Voxtral Transcribe 2는 온디바이스에서 실행되는 프라이버시 우선 ASR 모델을 오픈소스로 공개했다. 텍스트 토크나이저(BPE)의 원리가 음성 도메인으로 확장되는 과정을 실시간으로 목격한 주간이었다.

---

## 이번 주 핵심 키워드

| 키워드 | 등장 횟수 | 주요 맥락 |
|--------|-----------|-----------|
| **어텐션 효율화** | 4회 | HKT, DSA, Log-Linear, Gemini Flash |
| **토크나이저** | 3회 | 음성 토크나이저(Qwen3-TTS), BPE, 코드 토큰 |
| **임베딩** | 3회 | Harrier-OSS-v1, 화자 분리, Matryoshka |
| **Anthropic Mythos** | 3회 | 사이버 보안, UK 은행, 투자자 반응 |
| **오픈소스** | 4회 | MiniMax M2.7, Voxtral, Harrier, Gemma 4 |
| **Stanford AI Index** | 3회 | 투명성 하락, 벤치마크 오류, 에이전트 한계 |

---

## 이번 주 토론 질문 모음
1. **투명성 vs 성능:** 파운데이션 모델의 학습 데이터가 비공개되는 추세에서, 오픈소스 NLP 모델의 가치는?
2. **어텐션 최적화:** Self-Attention에서 '중요한 토큰 쌍'을 선별하는 기준은 무엇이 되어야 할까?
3. **음성 토크나이저:** 텍스트 BPE처럼 음성 토크나이저가 표준으로 수렴하려면 어떤 조건이 필요한가?
4. **임베딩 차원:** Matryoshka 임베딩으로 차원을 자유롭게 조절할 수 있다면 가장 유용한 실무 시나리오는?

---

## 다음 주 프리뷰 (Week 08, Day 34~38)

**계속 진행:** Ch04 자연어 및 음성 데이터

다음 주 주목할 키워드:
- **Anthropic Mythos 후속:** UK 금융기관 실제 테스트 결과 및 FCA·NCSC 태스크포스 결론
- **Google Gemma 4 생태계:** Apache 2.0 전환 이후 커뮤니티 반응 및 파인튜닝 사례
- **어텐션 연구:** ICLR 2026 학회 이후 추가 논문 및 벤치마크 결과
- **한국어 NLP:** Korean NLP 모델 업데이트 및 K-AI 리더보드 변동

---
**김생근** | AI Human 튜터
AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저. AI Dubbing·Avatar·Agentic AI 제품을 리딩하며 AI 네이티브 사고를 실무에 적용하고 있습니다.
