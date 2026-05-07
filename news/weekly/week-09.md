# AI Human 주간 다이제스트 — Week 09 (2026-04-27 ~ 05-01) | Day 39~43

**이번 주 진도:** Ch04 자연어 및 음성 데이터 — 핵심 구간(어텐션·BERT/Transformer·토크나이저·Speech-aware LLM)을 가장 집중적으로 통과한 한 주

---

## 🏆 이번 주 TOP 3

### 1. 프론티어 LLM의 "공개 → 한정 배포" 전환이 표준이 되다 (OpenAI · Anthropic · Meta)
주 초 OpenAI는 GPT-5.5 일반 공개로 Terminal-Bench 2.0에서 Claude Mythos를 추월했고, DeepSeek V4(1.6T MoE·1M 토큰)가 1/6 비용을 내세우며 추격했다. 그러나 주 후반에는 OpenAI가 GPT-5.5 Cyber를 "핵심 사이버 방어자"에게만 한정 공개(4/30) — 한 달 전 Anthropic의 Mythos 접근 제한을 비판하던 입장에서 정확히 같은 정책으로 수렴했다. Meta LlamaCon은 같은 주 Llama Guard 4·LlamaFirewall·Prompt Guard 2를 묶은 'Llama Defenders Program'을 공개했다. 빅3가 같은 주에 같은 결론에 도달한 것은 우연이 아니다 — 도메인 특화 LLM 배포 정책이 산업 표준으로 굳어지는 변곡점이다.

### 2. 한국어·다국어 NLP 평가 인프라가 한 주에 폭발적으로 쌓였다
KoALa-Bench(한국어 오디오 LLM 이해·충실도 — 4/27), Korean-Centric LLM Token Pruning(한국어 어휘 압축 벤치마크 — 4/29), MINT-Bench(10개 언어 instruction-following TTS — 4/28), PSP(Indic TTS 악센트 6축 분해 — 4/30), Do LLM Decoders Listen Fairly?(LLM prior가 만드는 ASR 편향 — 4/29), In-Sync(Speech-aware LLM 타임스탬프 — 4/30)까지 — 한 주에 6편의 평가 벤치마크/평가가능성 연구가 쏟아졌다. NLP·음성 도메인이 "성능 경쟁"에서 "정렬·편향·해석가능성 경쟁"으로 무게중심을 옮기고 있다는 분명한 시그널이다.

### 3. AI 자본 사이클 정점론 vs 현실 통계 — 같은 주에 부딪히다
주중 Anthropic 가치 평가가 $900B+ 추가 라운드(4/30) 검토로 알려지며 OpenAI 추월 가시권에 들어왔고, 동시에 Bezos의 Project Prometheus(100억 달러 — 4/28)와 Sierra의 Fragment 인수(4/29) 등 실물·에이전트 영역으로 자본이 분산됐다. 주말 직전에는 Bloomberg·Axios가 잇따라 "OpenAI 버블 ≠ AI 시장 버블" 논지를 제시했고, 미국 1분기 GDP 2% 성장과 3월 캐피탈 굿즈 주문 폭증(2020년 중반 이후 최대)이 거시 데이터로 그 주장을 뒷받침했다. 반대편에서는 Ares Management가 10억 달러 규모 SaaS 투자에 'AI 중간 이상 위험'을 표시했다 — 자본 시장이 두 갈래로 분기하는 중이다.

---

## 📊 이번 주 핵심 키워드

| 키워드 | 등장 빈도 | 핵심 맥락 |
|--------|-----------|-----------|
| Domain-specific LLM 배포 정책 | 매우 높음 | OpenAI Cyber 한정 공개, Anthropic Mythos, Meta Llama Guard 4 — 빅3 동시 수렴 |
| 한국어·다국어 NLP 평가 | 매우 높음 | KoALa-Bench, Korean Token Pruning, MINT-Bench, PSP, In-Sync, Do LLM Decoders |
| Speech-aware LLM | 높음 | NIM4-ASR, In-Sync, UniSonate, DriftSE — 음성-텍스트 정합성 연구 가속 |
| Llama 생태계 확장 | 높음 | LlamaCon Llama API, Llama Stack, NVIDIA NeMo·IBM·Red Hat 통합 |
| 프론티어 모델 가격·효율 | 높음 | DeepSeek V4 1/6 비용, GPT-5.5, Claude Mythos 추격 구도 |
| AI 자본 분기 | 중간 | Anthropic $900B+, Project Prometheus $10B, Ares 'AI 위험' 분류 |

---

## 🇰🇷 한국 학습자가 주목할 포인트

- **Korean-Centric LLM Token Pruning(4/29)**은 한국어 NLP 실무자에게 가장 즉시적인 무기다. 다국어 LLM의 한국어 어휘 압축이 모델 효율과 한국어 품질을 동시에 끌어올린다는 정량 증거가 처음 정리됐다.
- **KoALa-Bench(4/27)**는 한국어 오디오 LLM의 충실도(faithfulness)까지 평가하는 첫 종합 벤치마크 — 한국어 음성 어시스턴트 만드는 팀이라면 다음 분기 평가 표준으로 곧장 채택할 만하다.
- **OpenAI Cyber 한정 배포(4/30) + Llama Guard 4(4/29)** 조합은 한국 보안·금융 산업에 LLM을 도입하려는 팀이 어떤 게이트키핑 설계를 디폴트로 가져가야 하는지를 보여준다.

---

## 🔮 다음 주 프리뷰 (Week 10, Day 44~48)

- **Ch04 막바지 → Ch05 TTS·STT 진입(Day 54~)**까지 약 2주 남았다. 음성 토크나이저·음성 표상·SpeechLLM 결합 흐름이 이번 주에 이미 강하게 드러났으므로, 다음 주에는 **TTS 모델 구조(NaturalSpeech 계열·Diffusion-TTS·Flow-Matching)**와 **STT-LLM 결합 평가** 쪽 큐레이션이 늘어날 가능성이 높다.
- **Meta LlamaCon 후속 발표**(API 정식 공개·Llama Defenders 파트너 명단)가 주중 풀려나올 것으로 예상되며, 같은 주 OpenAI는 Cyber 모델 일반 공개 일정 또는 GPT-5.5 응용 카테고리(헬스·교육)를 공개할 가능성이 있다.
- **자본 시장 분기점**으로는 Anthropic $900B+ 라운드의 공식 클로징과, Ares가 제기한 'AI 중간 위험' SaaS 카테고리 리스트가 다른 사모 신용 운용사로 확산되는지가 다음 한 주의 핵심 관찰 포인트다.

---
**김생근** | AI Human 튜터
AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저. AI Dubbing·Avatar·Agentic AI 제품을 리딩하며 AI 네이티브 사고를 실무에 적용하고 있습니다.
