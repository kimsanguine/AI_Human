# AI Human Daily Brief — 셋업 가이드

## 현재 준비 완료된 것

| 항목 | 상태 | 위치 |
|------|------|------|
| GitHub 레포 구조 | ✅ 완료 | `~/Documents/AI_Human/` |
| 커리큘럼 매핑 JSON | ✅ 완료 | `~/Documents/AI_Human/curriculum/mapping.json` |
| Scheduled Task 프롬프트 | ✅ 완료 | `~/Documents/Claude/Scheduled/ai-human-daily-brief/SKILL.md` |
| 제안서 | ✅ 완료 | 별도 outputs 폴더 |

## 남은 셋업 단계 (5분)

### 1단계: GitHub 레포 초기화 + Push

터미널에서 아래 명령 실행:

```bash
cd ~/Documents/AI_Human
git init
git remote add origin https://github.com/kimsanguine/AI_Human.git
git add .
git commit -m "init: AI Human Daily Brief 자동화 구조"
git branch -M main
git push -u origin main
```

> GitHub 인증이 필요하면 `gh auth login` 또는 SSH key 설정

### 2단계: Slack 연결

Cowork에서 Slack 커넥터가 현재 미연결 상태입니다.
1. Cowork 설정에서 Slack 커넥터를 다시 연결
2. `#ai-human-news` 채널이 존재하는지 확인 (없으면 생성)

### 3단계: Scheduled Task 활성화

Cowork에서 새 세션을 열고:
```
"ai-human-daily-brief 스케줄 태스크를 활성화해줘"
```
또는 Claude Code에서:
```bash
claude schedule list
```
으로 태스크가 등록되었는지 확인

> 만약 자동 인식이 안 되면, 새 세션에서 아래 명령 실행:
> "매일 오전 8시에 ai-human-daily-brief 스케줄 태스크를 만들어줘. 프롬프트는 ~/Documents/Claude/Scheduled/ai-human-daily-brief/SKILL.md 파일을 참고해"

### 4단계: 수동 테스트 실행

Cowork에서:
```
"ai-human-daily-brief를 지금 한번 실행해줘"
```
→ 결과 확인 후 포맷/톤 조정

## 자동화 스케줄

- **매일 오전 8:00 (월~금)**: Daily Brief 자동 생성 + 배포
- **매주 금요일**: Weekly Digest 추가 생성
- **주말**: 미실행

## 커스터마이징

### 과정 시작일 변경
`curriculum/mapping.json`의 `start_date`를 수정하거나,
`SKILL.md`의 "과정 시작일: 2026-02-24" 부분을 변경

### 챕터 일정 조정
`curriculum/mapping.json`의 각 챕터 `day_start`, `day_end` 수정

### Telegram 채널 변경
`SKILL.md`의 Chat ID (`8595911950`) 변경

### 뉴스 건수/톤 조정
`SKILL.md`의 Phase별 뉴스 톤 섹션 수정
