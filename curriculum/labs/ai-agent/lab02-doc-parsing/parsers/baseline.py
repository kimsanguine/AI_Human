"""대조군: pdfplumber 로 그냥 텍스트만 뽑기.

'파서에 왜 돈을 내야 하는가'를 보여주는 기준선이다.
디지털 PDF(L1~L3)에서는 의외로 잘 나오고, 스캔본(L4)에서는 0점이 된다.
그 대비 자체가 수업 자료다.

    pip install pdfplumber
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import ParseResult, Parser


def _key(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


class BaselineParser(Parser):
    name = "pdfplumber"
    kind = "대조군"

    def available(self) -> tuple[bool, str]:
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            return False, "pdfplumber 미설치 (pip install pdfplumber)"
        except Exception as exc:   # noqa: BLE001
            # 의존성이 깨진 채로 설치된 경우(예: cryptography 빌드 실패)를
            # 여기서 잡지 않으면 비교 실행 전체가 죽는다.
            return False, f"pdfplumber 임포트 실패: {type(exc).__name__}"
        return True, ""

    def parse(self, pdf_path: Path) -> ParseResult:
        import pdfplumber

        chunks: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                md_rows, seen = [], set()
                for table in page.extract_tables():
                    for row in table:
                        cells = [c or "" for c in row]
                        md_rows.append("| " + " | ".join(cells) + " |")
                        seen.add(_key("".join(cells)))

                # 표로 이미 잡힌 줄을 본문에서 다시 넣으면 같은 내용이
                # 두 번 계산된다. 채점이 왜곡되므로 여기서 걸러낸다.
                for line in (page.extract_text() or "").splitlines():
                    if _key(line) and _key(line) not in seen:
                        chunks.append(line)
                chunks.extend(md_rows)
        return ParseResult(text="\n".join(chunks), cost_usd=0.0)
