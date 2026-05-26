# AI Human Weekly Digest — Week 12 (2026-05-18 ~ 05-22)

**현재 진도:** Ch05 TTS와 STT — 첫 주 (Day 54~58/100)
**다음 주(Week 13):** Ch05 후반 — Whisper/CTC 심화, 음성 에이전트 풀스택 (Day 59~63)
**이번 주 콘텐츠 합계:** 25건 (📰 19 · 💻 4 · 📄 2)

---

## TOP 3 — 이번 주 가장 중요한 신호

### 🥇 1. "STT+LLM+TTS 분리 시대의 종말" — 6개 음성 LLM이 1주에 등장
**왜 중요:** 이번 주는 **음성 모델 아키텍처 패러다임이 통째로 깨진 주**다. Thinking Machines Lab의 **Interaction Model**(턴테이킹 0.40초, 단일 모델 풀듀플렉스), OpenBMB **VoxCPM2**(tokenizer-free, 30개 언어), Supertone **Supertonic v3**(99M·온디바이스·31개 언어), k2-fsa **OmniVoice**(diffusion LM·646개 언어·실시간 40배), Microsoft **VibeVoice**(90분 장문·7.5Hz 토큰, ICLR 2026 Oral), 네이버 **HyperCLOVA X SEED Omni**(USDM 단일 모델)까지 — 모두 "STT→LLM→TTS 3단 분리"를 단일 모델로 흡수한다는 공통 시그널. **Ch05의 학습 대상 자체가 진화하는 한 주.**
**관련 일자:** 5/18(VoxCPM2)·5/19(Interaction Model·OmniVoice)·5/20(Supertonic v3·SemaVoice)·5/22(VibeVoice·HyperCLOVA X Omni)

### 🥈 2. OpenAI ↔ Anthropic IPO 카운트다운 본격화 — 컴퓨트·자본·인재 삼중전
**왜 중요:** 한 주 사이 **AI 양강의 IPO 경쟁이 가시화**됐다. OpenAI는 **5/22 SEC에 IPO 컨피덴셜 파일링**(9월 상장·$1T 목표), Anthropic은 xAI의 **Colossus 1 데이터센터 전체를 월 $1.25B로 임차**(220,000+ NVIDIA GPU, 2029년까지 $40B+ 거래)하며 컴퓨트 우위를 굳혔다. 동시에 **Andrej Karpathy가 Anthropic 사전학습팀에 합류**해 "Claude로 사전학습을 가속하는 팀"을 짓겠다는 인재 시그널. OpenAI는 **80년 묵은 에르되시 추측을 범용 추론 모델로 반증**해 모델 우위 입증. PM·CPO로서 다음 12개월의 거시 잣대가 깔린 주.
**관련 일자:** 5/18(Plaid 금융 연동), 5/19(Stainless 인수), 5/20(Karpathy·SPV 차단·Recursive Superintelligence $4.65B), 5/22(IPO·Colossus·80년 추측)

### 🥉 3. 음성 클로닝의 윤리·규제·자본화가 같은 주에 — Kakao SynthID부터 Apple Siri 균열까지
**왜 중요:** Ch05 첫 주에 **"음성 합성 = 산업 책임"**의 신호가 한꺼번에 떴다. OpenAI는 **Weights.gg(유명인 음성 복제 플랫폼)를 비공개 인수 후 폐쇄**(IPO 전 리스크 정리), Kakao는 **Google DeepMind의 SynthID 워터마크를 카나나에 아시아 최초 도입**, Google은 **Gboard 기본 키보드에 Gemini 받아쓰기 Rambler 탑재**(Wispr·Typeless 같은 STT 스타트업에 직격), Airbnb는 **AI 음성 통화 상담사를 연내 도입** 예고, Apple-OpenAI Siri 통합은 **법적 분쟁 직전**까지. 음성이 키보드·콜센터·증명 인프라까지 동시 침투하는 한 주.
**관련 일자:** 5/18(Weights.gg·Siri), 5/19(Gboard Rambler), 5/21(Kakao SynthID·Airbnb 음성 상담)

---

## 이번 주 핵심 키워드

| 카테고리 | 키워드 | 등장 횟수 |
|---|---|---|
| **음성 LLM 통합** | Interaction Model, VoxCPM2, Supertonic, OmniVoice, VibeVoice, HyperCLOVA X Omni, USDM | 7+ |
| **음성 클로닝/안전** | SynthID, Weights.gg, watermarking, speaker verification, opt-in consent | 5+ |
| **OpenAI IPO 트랙** | $1T 밸류, Goldman Sachs, Morgan Stanley, Plaid, Apple Siri 균열, 80년 추측 | 6+ |
| **Anthropic 확장** | Colossus 1·2, $1.25B/월, Karpathy 영입, Stainless 인수, SPV 차단 | 5+ |
| **음성 에이전트 상용** | Airbnb 음성 상담, Gboard Rambler, 카카오 국민비서 구삐, Vapi 후속 | 4+ |
| **추론 모델 일반화** | 에르되시 단위거리 추측, RecursiveMAS(임베딩 통신), Recursive Superintelligence | 3+ |
| **논문/오픈소스** | SemaVoice(WER 1.71%), VibeVoice 7.5Hz, OmniVoice diffusion LM | 3+ |

---

## 다음 주(Week 13) 프리뷰 — Ch05 후반 (Day 59~63)

다음 주는 **Whisper/CTC 심화**와 **음성 에이전트 풀스택**으로 들어간다. 미리 머리에 넣어두면 좋은 산업 컨텍스트:

1. **OpenAI IPO 후속 디테일 공개 임박**: 음성·멀티모달 매출 비중, GPU 단가, 추론 마진이 SEC 문서에서 처음 정량화될 가능성. Whisper API가 별도 라인인지부터 봐야 한다.
2. **xAI Colossus 2 가동**과 함께 다음 분기 GPU 시장이 재편될 것. Anthropic Claude Opus 4.7 학습이 어디서 돌고 있는지가 단서.
3. **음성 LLM 벤치마크 표준화**: VibeVoice가 Gemini 2.5 Pro TTS·ElevenLabs v3를 능가했다고 주장 — Seed-TTS bench·MOS·턴테이킹 지연 등 **음성 평가 메트릭의 표준화 압력**이 다음 주부터 가시화될 전망.
4. **한국 음성 AI 3대 축 비교 시점**: VoxCPM2(中)·HyperCLOVA X Omni(韓)·Supertonic v3(韓)·VibeVoice(美)를 코드 레벨에서 만져볼 좋은 타이밍. 다음 주 실습은 Whisper로 시작하지만, "왜 통합 모델이 분리 모델을 흡수하는가"를 손으로 검증해보자.

---

## Friday Question
> "이번 주를 한 줄로 요약하면 — **'음성 모델은 LLM에 흡수되고, AI 기업은 IPO·컴퓨트 경쟁에 들어갔으며, 음성 클로닝은 규제·안전 인프라(SynthID·워터마크) 위에서만 작동하기 시작했다.'** 우리 제품이 음성을 만진다면, 다음 분기 첫 의사결정은 (a) 분리 모델의 깊이를 더 파기 (b) 통합 음성 LLM API를 빠르게 채택하기 (c) 워터마크·동의·검증 같은 안전 레이어부터 깔기 — 셋 중 무엇이어야 하는가? 'IPO 자본·Colossus급 컴퓨트·USDM 통합'이라는 거시 신호 앞에서 우리가 가진 한 분기를 어디에 쓸 것인가?"

---
**김생근** | AI Human 튜터
AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저. AI Dubbing·Avatar·Agentic AI 제품을 리딩하며 AI 네이티브 사고를 실무에 적용하고 있습니다.
