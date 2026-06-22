# AI Human 7기 — 새 세션 킥오프 프롬프트

아래 블록 전체를 복사해 **새 채팅 세션**에 붙여넣으세요. 토큰 한 줄만 본인 것으로 바꾸면 됩니다.

---

## 📋 복사해서 붙여넣을 프롬프트

```
너는 내 AI Human 7기 일일 브리프 자동화를 셋업하는 역할이야. 아래 3단계를 순서대로 실행해줘.

[전제] 작업 폴더에 ~/Documents/AI_Human 레포가 연결되어 있어야 함.

1) GitHub 자동배포 인증 설정 (1회성)
- 아래 PAT로 remote URL을 교체해줘 (oauth2 형식):
  GITHUB_PAT = <여기에_새_fine-grained_PAT_붙여넣기>
- 실행: cd ~/Documents/AI_Human && git remote set-url origin "https://oauth2:${GITHUB_PAT}@github.com/kimsanguine/AI_Human.git"
- 인증 확인: GIT_TERMINAL_PROMPT=0 git ls-remote origin -h 가 성공하는지 확인하고 결과만 보고해줘 (토큰은 출력 마스킹).
- 그동안 로컬에 쌓인 커밋이 있으면 git push origin main 으로 한 번 올려줘.

2) 스케줄 등록
- 평일(월~금) 오전 7시(한국시간, KST)에 자동 실행되는 스케줄 태스크를 만들어줘.
- 스케줄 태스크의 지시문은 다음으로 해줘:
  "~/Documents/AI_Human/cohort7/SKILL.md 를 Read하고 그 지침을 정확히 따라 오늘자 AI Human 7기 Daily Brief 5건을 생성한 뒤, cohort7/news/daily/ 에 저장하고 GitHub(kimsanguine/AI_Human)에 커밋·푸시하라. 텔레그램·슬랙 발송은 하지 않는다. 주말에는 실행하지 않는다."

3) 첫 실행(오늘자) 테스트
- 지금 ~/Documents/AI_Human/cohort7/SKILL.md 를 Read해서 그대로 한 번 실행해줘 (오늘이 평일이면 Day 계산해서 브리프 생성 → GitHub 배포까지).
- 마지막에 Day 번호, 선별 5건 구성, GitHub 배포 성공 여부를 요약해줘.
```

---

## 🔑 사전 준비: GitHub PAT 발급 (중요)

기존 레포에 박혀 있던 토큰은 **만료/무효** 상태라 push가 실패합니다. 새 토큰이 꼭 필요합니다.

1. GitHub → Settings → Developer settings → **Fine-grained personal access tokens** → Generate new token
2. Repository access: **Only select repositories → `kimsanguine/AI_Human`**
3. Permissions → Repository permissions → **Contents: Read and write**
4. 만료일은 길게(예: 1년) 설정
5. 생성된 `github_pat_...` 토큰을 위 프롬프트의 `<여기에_새_fine-grained_PAT_붙여넣기>` 자리에 교체

> 토큰은 1회만 노출되니 바로 복사하세요. 새 세션에서 한 번 설정하면 이후 스케줄 실행은 자동으로 push됩니다.

## ✅ 이미 준비된 것 (이 세션에서 생성 완료)
- `cohort7/SKILL.md` — 7기 일일 브리프 지침 (텔레그램·슬랙 제외, GitHub 전용)
- `cohort7/curriculum/mapping.json` — 시작일 2026-06-23, 100일/9챕터
- `cohort7/news/daily/`, `cohort7/news/weekly/` — 산출물 폴더
- remote URL은 oauth2 형식으로 이미 교정됨 (토큰만 유효한 것으로 교체하면 됨)

## 📅 참고: 7기 일정
- Day 1 = 2026-06-23(화), Day 100 ≈ 평일 100일 후
- Ch01 Python(1~6) → Ch02 전처리(7~13) → Ch03 ML/DL(14~26) → Ch04 NLP/음성(27~53) → Ch05 TTS/STT(54~63) → Ch06 LLM(64~70) → Ch07 프롬프트(71~78) → Ch08 LangChain(79~90) → Ch09 RAG(91~100)
