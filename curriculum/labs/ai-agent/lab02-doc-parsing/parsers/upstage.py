"""Upstage Document Parse 어댑터 (상용 API).

    export UPSTAGE_API_KEY=up_...

국내 금융권 레퍼런스가 있는 상용 파서. 한글 표·체크박스 인식이 강점으로
알려져 있으나, 이 실습의 목적은 그 주장을 **직접 검증하는 것**이다.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from .base import ParseResult, Parser

ENDPOINT = "https://api.upstage.ai/v1/document-digitization"


def _multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = "----labboundary7f3a9c1e"
    parts = []
    for k, v in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
            .encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="document"; '
        f'filename="{file_path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class UpstageParser(Parser):
    name = "upstage"
    kind = "상용 API"

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("UPSTAGE_API_KEY"):
            return False, "UPSTAGE_API_KEY 미설정"
        return True, ""

    def parse(self, pdf_path: Path) -> ParseResult:
        body, content_type = _multipart(
            {"model": "document-parse", "output_formats": '["markdown"]'}, pdf_path
        )
        req = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {os.environ['UPSTAGE_API_KEY']}",
                "Content-Type": content_type,
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        content = payload.get("content", {})
        text = content.get("markdown") or content.get("text") or ""
        if not text:
            # 페이지 단위로만 오는 응답 형태 대비
            text = "\n".join(
                el.get("content", {}).get("markdown", "")
                for el in payload.get("elements", [])
            )
        return ParseResult(text=text)
