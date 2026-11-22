"""한글 폰트 확보.

폰트 파일(약 6MB×2)은 저장소에 커밋하지 않고 최초 실행 시 받아 캐시한다.
Noto Sans KR 은 SIL Open Font License 1.1 이라 재배포·임베딩이 가능하다.
사내망이라 외부 접속이 막혀 있으면 FONT_DIR 에 직접 넣어두면 된다.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

FONT_DIR = Path(__file__).parent / "fonts"

# fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700 이 가리키는 실제 파일
FONTS = {
    "NotoSansKR-Regular.ttf":
        "https://fonts.gstatic.com/s/notosanskr/v39/"
        "PbyxFmXiEBPT4ITbgNA5Cgms3VYcOA-vvnIzzuoyeLQ.ttf",
    "NotoSansKR-Bold.ttf":
        "https://fonts.gstatic.com/s/notosanskr/v39/"
        "PbyxFmXiEBPT4ITbgNA5Cgms3VYcOA-vvnIzzg01eLQ.ttf",
}


def ensure_fonts() -> dict[str, Path]:
    """폰트를 확보하고 {이름: 경로} 를 돌려준다."""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FONTS.items():
        dest = FONT_DIR / name
        if not dest.exists() or dest.stat().st_size < 100_000:
            print(f"  폰트 내려받는 중: {name}")
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths


if __name__ == "__main__":
    for n, p in ensure_fonts().items():
        print(f"{n}: {p} ({p.stat().st_size // 1024}KB)")
