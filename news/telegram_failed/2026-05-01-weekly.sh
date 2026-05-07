#!/bin/bash
# AI Human 주간 다이제스트 — Week 09 (2026-04-27 ~ 05-01)

MESSAGE=$(cat <<'MSG'
🗓️ *AI Human 주간 다이제스트 — Week 09 (2026-04-27 ~ 05-01) | Day 39~43*

*이번 주 진도:* Ch04 NLP/음성 핵심 구간(어텐션·BERT·토크나이저·Speech-aware LLM)을 가장 집중적으로 통과한 한 주

*🏆 TOP 3*

*1. 프론티어 LLM "공개 → 한정 배포" 표준화 (OpenAI · Anthropic · Meta)*
GPT-5.5 일반 공개로 출발했지만, 주말까지 OpenAI는 GPT-5.5 Cyber 한정 공개로, Meta는 LlamaCon에서 Llama Guard 4·Defenders Program으로 정확히 같은 결론에 도달. 도메인 특화 LLM 배포 정책이 산업 표준이 되는 변곡점.

*2. 한국어·다국어 NLP 평가 인프라가 한 주에 6편 폭발*
KoALa-Bench, Korean Token Pruning, MINT-Bench, PSP, Do LLM Decoders Listen Fairly?, In-Sync — "성능 경쟁 → 정렬·편향·해석가능성 경쟁"으로 무게중심 이동.

*3. AI 자본 사이클 — 분기 시작*
Anthropic $900B+ 라운드, Bezos Project Prometheus $10B vs Ares Management 'AI 중간 위험' SaaS 분류. Bloomberg·Axios 모두 "OpenAI 버블 ≠ AI 시장 버블" 결론. 미국 1Q GDP 2% 성장의 핵심 동인이 AI 설비투자.

*🇰🇷 한국 학습자 주목 포인트*
- Korean-Centric LLM Token Pruning — 다국어 LLM 한국어 어휘 압축의 첫 정량 증거
- KoALa-Bench — 한국어 오디오 LLM 충실도 첫 종합 평가
- OpenAI Cyber + Llama Guard 4 — 한국 보안·금융 LLM 도입의 디폴트 게이트키핑

*🔮 다음 주 (Week 10, Day 44~48)*
TTS·STT(Ch05) 진입 임박 — NaturalSpeech·Diffusion-TTS·Flow-Matching 큐레이션 증가 예상. LlamaCon 후속 발표, Anthropic 라운드 클로징, Ares 'AI 중간 위험' 리스트의 사모 신용 시장 확산이 핵심 관찰 포인트.

🔗 https://github.com/kimsanguine/AI_Human/blob/main/news/weekly/week-09.md
MSG
)

curl -s -X POST "https://api.telegram.org/bot8236282258:AAENnWVQ2wgtJWMzaEuA4tge9qpDL-oXYhg/sendMessage" \
  -d chat_id=8595911950 \
  -d parse_mode=Markdown \
  --data-urlencode "text=$MESSAGE"
