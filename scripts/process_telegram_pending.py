#!/usr/bin/env python3
"""
process_telegram_pending.py — pending Telegram 브리프 자동 발송기

매 10분마다 cron으로 실행:
- ~/Documents/AI_Human/news/telegram_pending/*.sh 를 순회
- 각 .sh를 그대로 실행 (이미 curl 명령이 들어있음)
- Telegram API가 {"ok":true} 반환하면 ~/Documents/AI_Human/news/telegram_processed/ 로 이동
- 실패하면 pending에 남겨두고 로그 기록

환경변수 (선택):
    TELEGRAM_BOT_TOKEN  — 새 토큰. 설정되어 있으면 .sh 안의 옛 토큰을 자동 치환해서 실행
    TELEGRAM_CHAT_ID    — 채팅 ID (기본: .sh 안의 값 그대로)

권장 crontab:
    */10 * * * * /usr/bin/python3 ~/Documents/AI_Human/scripts/process_telegram_pending.py >> ~/Documents/AI_Human/scripts/process_telegram_pending.log 2>&1
"""

from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "Documents" / "AI_Human"
PENDING = ROOT / "news" / "telegram_pending"
PROCESSED = ROOT / "news" / "telegram_processed"
FAILED = ROOT / "news" / "telegram_failed"
LOG_PATH = ROOT / "scripts" / "process_telegram_pending.log"

PROCESSED.mkdir(parents=True, exist_ok=True)
FAILED.mkdir(parents=True, exist_ok=True)

# 옛 토큰 패턴 — 새 토큰으로 치환할 때 매칭용
OLD_TOKEN_PATTERN = re.compile(r"bot\d{9,12}:[A-Za-z0-9_\-]{30,}")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def _run_bash_file(path: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute a bash script file directly (NOT via stdin) to keep heredocs intact."""
    try:
        result = subprocess.run(
            ["/bin/bash", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

    output = (result.stdout or "") + (result.stderr or "")
    return ('"ok":true' in output), output.strip()


def process_one(script: Path) -> bool:
    """
    Run one pending .sh. Optionally rewrite the bot token from env.
    Executes as a file (never via stdin) so heredocs aren't broken.
    Returns True on success.
    """
    new_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    body = script.read_text(encoding="utf-8")

    target_path = str(script)
    temp_path: str | None = None

    # If env token is set and the file holds an old-format bot token, rewrite to a temp file.
    if new_token:
        rewritten = OLD_TOKEN_PATTERN.sub(f"bot{new_token}", body)
        if rewritten != body:
            log(f"  token rewritten from env for {script.name}")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(rewritten)
                temp_path = tf.name
            os.chmod(temp_path, 0o700)
            target_path = temp_path

    try:
        ok, output = _run_bash_file(target_path)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    output_short = output[:200] if output else "(empty)"
    if ok:
        log(f"  ✓ sent {script.name} — Telegram ok")
        return True

    log(f"  ✗ FAILED {script.name} — response: {output_short}")
    return False


def main() -> int:
    log(f"=== run start, pending dir: {PENDING} ===")

    if not PENDING.exists():
        log("pending dir not found — nothing to do")
        return 0

    scripts = sorted(PENDING.glob("*.sh"))
    if not scripts:
        log("no pending scripts")
        return 0

    log(f"{len(scripts)} pending script(s)")
    sent, failed = 0, 0
    for s in scripts:
        log(f"processing {s.name}")
        if process_one(s):
            shutil.move(str(s), PROCESSED / s.name)
            sent += 1
        else:
            # Leave in pending so next run retries; copy to failed/ for inspection
            shutil.copy2(str(s), FAILED / s.name)
            failed += 1

    log(f"=== run done — sent={sent} failed={failed} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
