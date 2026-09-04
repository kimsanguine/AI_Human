"""공격 테스트 러너.

    python run_attacks.py                    # 모든 방어 켜고 실행 (전부 BLOCKED 여야 정상)
    python run_attacks.py --disable network  # 네트워크 차단만 해제 → A01/A02 가 뚫린다
    python run_attacks.py --disable readonly --disable caps
    python run_attacks.py --only A05

--disable 로 방어를 하나씩 끄면서 "이 플래그가 없으면 무엇이 가능해지는가"를
직접 보는 것이 이 실습의 핵심이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from attacks import ATTACKS, Attack
from sandbox import run_python

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / ".out"

DISABLE_CHOICES = {
    "network": "network",       # --network none 해제
    "readonly": "read_only",    # --read-only 해제
    "caps": "drop_caps",        # --cap-drop ALL 해제
    "user": "run_as_nobody",    # --user 65534 해제 (root 로 실행)
    "pids": "pids_limit",       # --pids-limit 해제
    "memory": "memory",         # --memory 해제
}


def build_kwargs(attack: Attack, disabled: set[str]) -> dict:
    kwargs = dict(attack.kwargs)
    if "network" in disabled:
        kwargs["network"] = True
    if "readonly" in disabled:
        kwargs["read_only"] = False
    if "caps" in disabled:
        kwargs["drop_caps"] = False
    if "user" in disabled:
        kwargs["run_as_nobody"] = False
    if "pids" in disabled:
        kwargs["pids_limit"] = 4096
    if "memory" in disabled:
        kwargs["memory"] = "4g"
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable", action="append", default=[],
                        choices=sorted(DISABLE_CHOICES), help="방어 장치를 끈다")
    parser.add_argument("--only", action="append", default=[], help="특정 케이스만 실행")
    args = parser.parse_args()

    disabled = set(args.disable)
    cases = [a for a in ATTACKS if not args.only or a.id in args.only]

    if disabled:
        print(f"⚠️  방어 해제: {', '.join(sorted(disabled))}\n")

    rows, failures = [], 0
    for atk in cases:
        result = run_python(
            atk.code,
            data_dir=DATA,
            out_dir=OUT / atk.id,
            **build_kwargs(atk, disabled),
        )
        leaked = atk.marker in result.stdout

        if atk.expect_success:
            passed = result.ok and not leaked
            verdict = "OK" if passed else "BROKEN"
        else:
            passed = not leaked
            verdict = "BLOCKED" if passed else "PWNED"

        if not passed:
            failures += 1

        detail = []
        if result.timed_out:
            detail.append("timeout")
        if result.oom_killed:
            detail.append("oom-killed")
        if result.exit_code not in (0, None):
            detail.append(f"exit={result.exit_code}")
        if result.artifacts:
            detail.append(f"artifacts={len(result.artifacts)}")

        rows.append((atk.id, verdict, atk.title, atk.blocked_by,
                     ",".join(detail) or "-", f"{result.duration:.1f}s"))

    width = max(len(r[2]) for r in rows)
    print(f"{'ID':<4} {'결과':<8} {'시나리오':<{width}}  {'방어 장치':<42} {'상세':<22} 시간")
    print("-" * (width + 88))
    for rid, verdict, title, blocked_by, detail, dur in rows:
        icon = {"BLOCKED": "🛡 ", "OK": "✅", "PWNED": "💥", "BROKEN": "❌"}[verdict]
        print(f"{rid:<4} {icon}{verdict:<6} {title:<{width}}  {blocked_by:<42} {detail:<22} {dur}")

    print()
    if failures:
        print(f"❌ {failures}개 케이스가 기대와 다릅니다.")
        if disabled:
            print("   (방어를 해제했으니 뚫리는 것이 정상입니다 — 어떤 것이 뚫렸는지 보세요.)")
            return 0
        return 1
    print(f"✅ {len(rows)}개 케이스 모두 기대대로 동작합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
