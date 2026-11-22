"""합성 문서 → PDF + 정답(JSON) 생성.

    python3 docgen/generate.py            # dataset/ 아래에 전부 생성
    python3 docgen/generate.py --png      # 미리보기 PNG도 함께

Chromium 헤드리스로 HTML을 인쇄한다. 별도 파이썬 PDF 라이브러리가 필요 없고,
템플릿을 HTML/CSS로 다루므로 강사가 난이도를 바꾸기 쉽다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import re  # noqa: E402

import templates  # noqa: E402
from fonts import ensure_fonts  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "dataset"

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
]


def html_to_text(html: str) -> str:
    """태그를 제거해 평문 정답을 만든다. 표 셀은 공백으로 구분한다."""
    text = re.sub(r"<(td|th|p|div|h[1-6]|tr)[^>]*>", " ", html)
    text = re.sub(r"</(tr|p|div|h[1-6])>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if not c:
            continue
        if Path(c).is_file():
            return c
        found = shutil.which(c)
        if found:
            return found
    # 설치 위치가 버전마다 달라지므로 마지막으로 한 번 훑는다
    for p in Path("/opt/pw-browsers").glob("chromium*/chrome-linux/chrome"):
        return str(p)
    raise SystemExit(
        "Chromium 을 찾지 못했습니다. CHROME_PATH 환경변수로 경로를 지정하세요.\n"
        "  예: export CHROME_PATH=/usr/bin/chromium"
    )


def render(chrome: str, html_path: Path, out_path: Path, png: bool = False) -> None:
    common = ["--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars"]
    subprocess.run(
        [chrome, *common, "--no-pdf-header-footer",
         f"--print-to-pdf={out_path}", f"file://{html_path}"],
        capture_output=True, timeout=120, check=False,
    )
    if not out_path.exists():
        raise RuntimeError(f"PDF 생성 실패: {out_path}")
    if png:
        subprocess.run(
            [chrome, *common, "--window-size=1240,1754",
             f"--screenshot={out_path.with_suffix('.png')}", f"file://{html_path}"],
            capture_output=True, timeout=120, check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="미리보기 PNG도 생성")
    args = parser.parse_args()

    fonts = ensure_fonts()
    chrome = find_chrome()
    OUT.mkdir(parents=True, exist_ok=True)

    css = (templates.CSS
           .replace("__REG__", str(fonts["NotoSansKR-Regular.ttf"]))
           .replace("__BOLD__", str(fonts["NotoSansKR-Bold.ttf"])))

    manifest = []
    for make in templates.ALL:
        doc = make()
        doc_id = doc["doc_id"]
        html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                f"<style>{css}</style></head><body>{doc['html']}</body></html>")
        html_path = OUT / f"{doc_id}.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_path = OUT / f"{doc_id}.pdf"
        render(chrome, html_path, pdf_path, png=args.png)

        truth = {k: v for k, v in doc.items() if k != "html"}
        # CER 채점용 평문. HTML 태그를 걷어내 문서의 '읽히는 글자'만 남긴다.
        truth["text"] = html_to_text(doc["html"])
        (OUT / f"{doc_id}.truth.json").write_text(
            json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest.append({
            "doc_id": doc_id,
            "level": doc["level"],
            "note": doc["note"],
            "pdf": pdf_path.name,
            "truth": f"{doc_id}.truth.json",
            "bytes": pdf_path.stat().st_size,
        })
        print(f"  ✓ {doc_id:<28} {pdf_path.stat().st_size // 1024:>4}KB")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)}개 문서를 {OUT} 에 생성했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
