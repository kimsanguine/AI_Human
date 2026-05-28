# AI Human Weekly Digest — Week 13 (2026-05-25 ~ 2026-05-29)

**진도:** Ch05 TTS와 STT — Day 59~63 (이번 주로 Ch05 완료, 다음 주부터 Ch06 거대 언어 모델)
**주간 키워드:** 한국어 TTS의 SOTA 경쟁 · 오픈소스 음성 모델 폭증 · AI 1위 자리 교체 · 풀스택 음성 에이전트

---

## 이번 주 TOP 3

### TOP 1. 🇰🇷 한국 TTS 진영의 'SOTA 분기점' — Kakao·DeepBrain·Raon이 같은 주에 새 모델 공개
지난 한 주 동안 한국 음성 AI에서 **3개의 큰 신호**가 동시에 잡혔다. (1) Kakao는 if(kakao)25에서 KakaoTalk '일상 AI'에 음성을 통합하고 Kanana 2.0을 MLA+MoE 구조로 진화시켰다. (2) DeepBrain AI는 '문맥 인식형(Context-Aware) 표현 TTS'를 1,000+ 보이스로 정식 출시했고, (3) Raon-OpenTTS는 **데이터·가중치·학습 파이프라인을 전 과정 오픈**한 DiT 기반 TTS로 Seed-TTS-Eval SIM에서 1위를 차지했다. 한국어 자연성·감정 표현·오픈성 3축이 동시에 흔들린 한 주.

### TOP 2. 💰 Anthropic, $965B로 OpenAI($730B) 추월 — '클로징 임박' → '실제 클로즈' 5일 만에
5/26 브리프(Day 60)에서 다룬 '$30B 라운드 클로징 임박'이 5/28(Day 62) **$65B 실제 클로즈, 포스트머니 $965B**로 두 배 가까이 커진 채 마감됐다. run-rate 매출은 $47B(5월 초 기준), 자금은 안전성·해석성 연구 + 컴퓨트 확장에 투입. 같은 주에 Q2 매출 $10.9B 흑자 진입 전망(5/27)까지 발표되며, 비용·매출·밸류 3대 축이 한 주 안에 모두 갱신됐다. 다음 주 Ch06 LLM 진입 시 **'프런티어 모델 학습 = 자본 게임'** 의 현재 좌표.

### TOP 3. 🔓 오픈 음성 모델 폭증 — Stability·Raon·VoxCPM2·X-Voice가 1주일에 몰아치다
이번 주 공개된 오픈 음성 모델: (1) Stability AI의 라이선스 데이터 기반 6분 곡 오디오 모델(5/26), (2) Raon-OpenTTS(5/28), (3) ICLR 2026 Oral인 microsoft/VibeVoice(지난 주), (4) GitHub Trending #1 OpenBMB/VoxCPM2(이번 주), (5) arXiv X-Voice 제로샷 크로스링구얼 클로닝(5/7 게재, 이번 주 부상). '클로즈드 SOTA(ElevenLabs Music v2) ↔ 오픈 SOTA' 격차가 빠르게 좁혀지는 구간에 진입했다.

---

## 주간 핵심 키워드

- **Tokenizer-Free / Continuous Acoustic Latent** — 멜-스펙트로그램·이산 토큰 단계를 건너뛰는 새 흐름 (VoxCPM2, Raon-OpenTTS)
- **Context-Aware TTS** — 텍스트 의미·감정을 모델이 직접 추론해 발화 스타일을 조절 (DeepBrain, Kakao)
- **Zero-shot Cross-lingual Voice Cloning** — 화자 임베딩 + 언어 ID의 이중 컨디셔닝 (X-Voice, F5-TTS 계열)
- **풀스택 음성 에이전트** — STT → 의도 분류 → 도구 호출 → 음성 응답을 한 파이프라인에 통합 (SKT A.Biz Cowork, Perplexity Computer)
- **AI 1위 교체 + IPO 시나리오** — Anthropic $965B, Q3~Q4 IPO 가시화

---

## 다음 주 프리뷰 (Week 14, Day 64~68 → Ch06 LLM 진입)

**커리큘럼:** Ch06 거대 언어 모델 — GPT/Claude/Gemini/LLaMA, 오픈소스 LLM, 파인튜닝, Ollama
**예상 화두:**
1. **Anthropic Claude 다음 버전** — $965B 펀딩 직후 모델 로드맵 공개 가능성
2. **OpenAI 응전** — GPT-5.5 후속/실시간 API 업데이트
3. **오픈 LLM 진영** — LLaMA·Mistral·Qwen 최신 릴리스, Ollama 통합 사례
4. **한국 LLM 동향** — HyperCLOVA X SEED, Upstage Solar Pro, Kanana 2.0 후속
5. **추론 인프라** — Fireworks AI($15B), Together AI 등의 추론 가격 경쟁

이번 주 학습한 **TTS·STT가 LLM의 '입력/출력 인터페이스'로 어떻게 묶이는지** 다음 주에 확인하게 된다. 음성 → 텍스트 → 모델 → 텍스트 → 음성 풀스택을 직접 그려보고 다음 주에 진입하자.

---
**김생근** | AI Human 튜터
