#!/usr/bin/env python3
"""
fetch_signals.py — AI Human Daily Brief 임팩트 시그널 사전 수집기 (v2)

매일 아침 cron/launchd로 실행:
- HackerNews Algolia API → 지난 48시간 점수 50+ 스토리 (AI 키워드 클라이언트 필터)
- GitHub Search API → 토픽별 개별 호출 (OR 결합 금지 — 422 회피)

결과를 ~/Documents/AI_Human/news/signals/YYYY-MM-DD.json 에 저장.

권장 cron:
    0 6 * * 1-5  /usr/bin/python3 ~/Documents/AI_Human/scripts/fetch_signals.py >> ~/Documents/AI_Human/scripts/fetch_signals.log 2>&1
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path.home() / "Documents" / "AI_Human"
OUT_DIR = REPO_ROOT / "news" / "signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OUT_FILE = OUT_DIR / f"{TODAY}.json"

# AI 관련 키워드 — word boundary 매칭용 (\b로 감싸서 false positive 방지)
# 짧은 단어("ai", "ml")는 word boundary 필수, 긴 단어는 substring 허용
AI_KEYWORDS_WORD = [  # 단어 경계 매칭 (양쪽 \b)
    "ai", "ml", "llm", "llms", "gpt", "claude", "gemini", "anthropic", "openai",
    "transformer", "transformers", "agent", "agents", "rag", "whisper", "tts",
    "stt", "asr", "embedding", "embeddings", "diffusion", "neural", "lora",
    "vllm", "tokenizer", "tokenization", "mistral", "llama", "deepseek", "qwen",
    "copilot", "agentic", "mcp", "ollama", "huggingface", "rlhf", "dpo",
    "fine-tuning", "fine-tune", "finetune", "midjourney",
]
AI_KEYWORDS_SUBSTRING = [  # 단순 substring 매칭 (긴 표현, 한국어)
    "deep learning", "machine learning", "stable diffusion", "language model",
    "speech recognition", "text-to-speech", "speech-to-text", "neural network",
    "vector database", "vector db", "ai model", "ai agent", "ai coding",
    "코드", "모델", "ai모델", "인공지능", "한국어", "음성", "언어 모델",
    "거대 언어", "프롬프트", "에이전트",
]

# 한 번에 컴파일 (성능)
_AI_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in AI_KEYWORDS_WORD) + r")\b",
    re.IGNORECASE,
)

# GitHub 토픽 — 우선순위 순서 (rate limit 시 앞쪽 우선 보장)
GITHUB_TOPICS = [
    # Tier 1: 가장 핵심 (Phase 4 LLM/Agent/RAG 전 영역)
    "llm", "ai-agent", "rag", "agentic-ai", "large-language-models",
    # Tier 2: 음성·NLP
    "speech-recognition", "tts", "nlp", "transformer",
    # Tier 3: 일반 AI / 모델 생태계
    "ollama", "claude", "vector-database", "ai", "machine-learning",
    # Tier 4: 보조
    "deep-learning", "agent", "retrieval-augmented-generation", "text-to-speech",
]

# GitHub 노이즈 컷: 너무 큰 레포는 "trending" 신호 약함 (이미 baseline)
GITHUB_MAX_STARS = 30000
GITHUB_MIN_STARS = 10
# rate limit 회피: 호출 간 sleep (초)
GITHUB_SLEEP = 1.5

UA = "AI-Human-Daily-Brief/1.1 (+https://github.com/kimsanguine/AI_Human)"


def fetch_json(url: str, headers: dict | None = None, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def is_ai_related(title: str) -> bool:
    """word boundary 매칭 + 긴 표현 substring 매칭. False positive(train→ai, said→ai) 방지."""
    if not title:
        return False
    if _AI_WORD_RE.search(title):
        return True
    t = title.lower()
    return any(kw in t for kw in AI_KEYWORDS_SUBSTRING)


def fetch_hackernews() -> list[dict]:
    """지난 48시간 점수 50+ 스토리를 가져온 뒤 AI 키워드 클라이언트 필터링"""
    epoch_48h_ago = int((datetime.now(timezone.utc) - timedelta(hours=48)).timestamp())

    # query는 비우고 numericFilters만 사용 — OR 폭주 회피
    qs = urllib.parse.urlencode({
        "tags": "story",
        "numericFilters": f"created_at_i>{epoch_48h_ago},points>50",
        "hitsPerPage": 100,
    })
    url = f"https://hn.algolia.com/api/v1/search?{qs}"

    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"[HN] 실패: {e}", file=sys.stderr)
        return []

    hits = data.get("hits", [])
    out = []
    for h in hits:
        title = h.get("title") or ""
        if not is_ai_related(title):
            continue
        # 매칭된 키워드 기록 (디버깅용)
        match = _AI_WORD_RE.search(title)
        matched_kw = match.group(1).lower() if match else next(
            (kw for kw in AI_KEYWORDS_SUBSTRING if kw in title.lower()), ""
        )
        out.append({
            "title": title,
            "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            "hn_url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            "points": h.get("points", 0),
            "comments": h.get("num_comments", 0),
            "created_at": h.get("created_at"),
            "author": h.get("author"),
            "matched_keyword": matched_kw,
        })
    out.sort(key=lambda x: (x["points"], x["comments"]), reverse=True)
    return out


def fetch_github_trending() -> list[dict]:
    """토픽별 개별 쿼리. created>last_week (신규 trending 우선) + pushed>last_week (활발 유지).
    rate limit 회피: 호출 간 sleep, 403 만나면 retry once with longer wait."""
    last_week = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    seen: set[str] = set()
    out: list[dict] = []
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    consecutive_403 = 0
    rate_limited = False

    for topic in GITHUB_TOPICS:
        if rate_limited:
            print(f"[GH] rate limit 진입 — 남은 토픽 스킵: {topic}부터", file=sys.stderr)
            break

        # 신규 trending 우선 (created), 그 다음 활발 유지 (pushed)
        for date_qual in (f"created:>{last_week}", f"pushed:>{last_week}"):
            q = f"topic:{topic} {date_qual}"
            qs = urllib.parse.urlencode({
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": 10,
            })
            url = f"https://api.github.com/search/repositories?{qs}"

            data = None
            for attempt in (1, 2):
                try:
                    data = fetch_json(url, headers=headers)
                    consecutive_403 = 0
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 403 and attempt == 1:
                        # rate limit 가능성 — 한 번 더 wait 후 재시도
                        time.sleep(10)
                        continue
                    print(f"[GH] '{q}' HTTP {e.code}: {e.reason}", file=sys.stderr)
                    if e.code == 403:
                        consecutive_403 += 1
                    break
                except Exception as e:
                    print(f"[GH] '{q}' 실패: {e}", file=sys.stderr)
                    break

            if data is None:
                if consecutive_403 >= 3:
                    rate_limited = True
                    break  # 안쪽 for 빠져나감
                time.sleep(GITHUB_SLEEP)
                continue

            for r in data.get("items", []):
                full = r.get("full_name")
                if not full or full in seen:
                    continue
                stars = r.get("stargazers_count", 0)
                # 노이즈 컷: 너무 작거나 너무 큰 레포 제외 (trending 신호가 약함)
                if stars < GITHUB_MIN_STARS or stars > GITHUB_MAX_STARS:
                    continue
                seen.add(full)
                out.append({
                    "full_name": full,
                    "description": r.get("description") or "",
                    "url": r.get("html_url"),
                    "stars": stars,
                    "language": r.get("language"),
                    "topics": r.get("topics", []),
                    "created_at": r.get("created_at"),
                    "pushed_at": r.get("pushed_at"),
                    "match_topic": topic,
                    "match_qualifier": date_qual,
                })

            # 호출 간 sleep
            time.sleep(GITHUB_SLEEP)

    out.sort(key=lambda x: x["stars"], reverse=True)
    return out[:30]


def main():
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "date": TODAY,
        "hackernews": fetch_hackernews(),
        "github_trending": fetch_github_trending(),
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved {OUT_FILE}")
    print(f"  HN: {len(payload['hackernews'])} items")
    print(f"  GitHub: {len(payload['github_trending'])} items")
    if payload["hackernews"]:
        for top in payload["hackernews"][:3]:
            kw = top.get("matched_keyword", "")
            print(f"  HN: [{top['points']}점·{top['comments']}댓글] ({kw}) {top['title'][:70]}")
    if payload["github_trending"]:
        for top in payload["github_trending"][:3]:
            print(f"  GH: ⭐{top['stars']:>5} {top['full_name'][:40]:40} ({top['match_topic']})")

    # 30일 이상 지난 시그널 정리
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    for old in OUT_DIR.glob("*.json"):
        try:
            d = datetime.strptime(old.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if d < cutoff:
                old.unlink()
                print(f"  pruned: {old.name}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
