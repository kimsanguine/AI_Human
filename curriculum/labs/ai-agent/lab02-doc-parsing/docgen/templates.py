"""합성 한국어 금융 문서 4종.

핵심 아이디어: **구조화된 데이터에서 문서를 만든다.**
그래서 정답(ground truth)이 공짜로 따라온다. 사람이 라벨링할 필요가 없고,
저작권·개인정보 문제도 없으며, 난이도를 마음대로 올릴 수 있다.

난이도:
  L1 선이 있는 단순 표          — 어떤 파서든 통과해야 하는 하한선
  L2 선 없는 표 + 음수 괄호      — 여기서부터 갈린다
  L3 2단 레이아웃 + 체크박스 + 각주
  L4 스캔본 시뮬레이션 (기울임·회색조·대비)

모든 회사명·성명·수치는 가상이다.
"""

from __future__ import annotations

CSS = """
@font-face{font-family:NK;src:url("file://__REG__");font-weight:400}
@font-face{font-family:NK;src:url("file://__BOLD__");font-weight:700}
@page{size:A4;margin:18mm}
body{font-family:NK,sans-serif;font-size:10.5pt;line-height:1.7;color:#000}
h1{font-size:15pt;margin:0 0 4mm}
h2{font-size:12pt;margin:6mm 0 2mm}
p{margin:0 0 3mm;text-align:justify}
.num{text-align:right;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;margin:3mm 0;font-size:10pt}
.lined td,.lined th{border:1px solid #333;padding:2mm 3mm}
.lined th{background:#eee}
.ruleless td,.ruleless th{border:none;padding:1.6mm 3mm}
.ruleless thead th{border-bottom:1.2pt solid #000}
.ruleless tbody tr:last-child td{border-top:0.8pt solid #000}
.two-col{column-count:2;column-gap:9mm}
.box{border:1px solid #666;padding:3mm;margin:3mm 0}
.chk{font-size:11pt;margin:0 0 1.5mm}
.foot{font-size:8.5pt;color:#333;border-top:0.5pt solid #999;margin-top:5mm;padding-top:2mm}
.scan{filter:grayscale(1) contrast(1.18) brightness(0.97);transform:rotate(-0.7deg)}
"""


def _is_num(cell: str) -> bool:
    s = str(cell).strip().strip("()").replace(",", "").replace("%", "").replace("-", "")
    return s.replace(".", "", 1).isdigit()


def _table(header, rows, ruleless=False):
    cls = "ruleless" if ruleless else "lined"
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="num">{c}</td>' if _is_num(c) else f"<td>{c}</td>" for c in r
        ) + "</tr>"
        for r in rows
    )
    return (f'<table class="{cls}"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def level1():
    header = ["담보명", "가입금액", "보험료", "보장개시일"]
    rows = [
        ["상해사망", "100,000,000", "4,200", "계약일"],
        ["상해후유장해", "100,000,000", "3,150", "계약일"],
        ["질병사망", "50,000,000", "8,900", "계약일로부터 90일 후"],
        ["입원일당(1일)", "30,000", "2,400", "계약일로부터 30일 후"],
        ["수술비", "1,000,000", "1,750", "계약일로부터 30일 후"],
    ]
    html = f"""
    <h1>가온생명 무배당 안심상해보험 가입내역</h1>
    <p>본 문서는 실습을 위해 생성된 가상의 보험 가입내역입니다.
    아래 담보별 가입금액과 보험료는 실제 상품과 무관합니다.</p>
    <h2>1. 담보별 가입내역</h2>
    {_table(header, rows)}
    <p>합계보험료는 월 20,400원이며, 납입기간은 20년, 보험기간은 80세만기입니다.</p>
    """
    return {
        "doc_id": "L1_보험가입내역",
        "level": 1,
        "note": "선이 있는 단순 표. 모든 파서가 통과해야 하는 하한선.",
        "html": html,
        "tables": [{"name": "담보별_가입내역", "header": header, "rows": rows}],
        "fields": [
            {"key": "상해사망 가입금액", "anchor": "상해사망", "value": "100,000,000"},
            {"key": "질병사망 보험료", "anchor": "질병사망", "value": "8,900"},
            {"key": "입원일당 보장개시일", "anchor": "입원일당", "value": "계약일로부터 30일 후"},
            {"key": "합계보험료", "anchor": "합계보험료", "value": "20,400"},
        ],
    }


def level2():
    header = ["과목", "제25기", "제24기", "증감률"]
    rows = [
        ["매출액", "1,842,300", "1,653,120", "11.4%"],
        ["매출원가", "1,398,740", "1,281,050", "9.2%"],
        ["매출총이익", "443,560", "372,070", "19.2%"],
        ["판매비와관리비", "312,880", "298,410", "4.8%"],
        ["영업이익", "130,680", "73,660", "77.4%"],
        ["금융원가", "(41,220)", "(38,905)", "5.9%"],
        ["법인세비용차감전순이익", "89,460", "34,755", "157.4%"],
        ["당기순이익", "68,790", "26,320", "161.4%"],
    ]
    html = f"""
    <h1>주식회사 다온화학 요약 손익계산서</h1>
    <p>제25기 2025.01.01 ~ 2025.12.31 / 제24기 2024.01.01 ~ 2024.12.31 (단위: 백만원)</p>
    {_table(header, rows, ruleless=True)}
    <p class="foot">※ 괄호는 부(負)의 금액을 의미합니다. 본 자료는 실습용 가상 데이터입니다.</p>
    """
    return {
        "doc_id": "L2_요약손익계산서",
        "level": 2,
        "note": "선 없는 표 + 음수 괄호 + 우측정렬. 여기서부터 파서 실력이 갈린다.",
        "html": html,
        "tables": [{"name": "요약손익계산서", "header": header, "rows": rows}],
        "fields": [
            {"key": "제25기 영업이익", "anchor": "영업이익", "value": "130,680"},
            {"key": "제25기 금융원가", "anchor": "금융원가", "value": "(41,220)"},
            {"key": "제24기 당기순이익", "anchor": "당기순이익", "value": "26,320"},
            {"key": "당기순이익 증감률", "anchor": "당기순이익", "value": "161.4%"},
        ],
    }


def level3():
    header = ["구분", "내용", "확인"]
    rows = [
        ["계약자", "김실습 (1985.03.12)", "☑"],
        ["피보험자", "김실습 (계약자와 동일)", "☑"],
        ["수익자", "법정상속인", "☐"],
        ["납입방법", "자동이체 / 매월 25일", "☑"],
    ]
    html = f"""
    <h1>보험계약 청약서 (실습용 가상 서식)</h1>
    <div class="two-col">
      <p>본 청약서는 문서 파싱 실습을 위해 생성된 가상 서식입니다.
      기재된 성명·생년월일은 모두 허구이며 실재하지 않습니다.</p>
      <p>계약자는 청약서 작성 시 아래 사항을 반드시 확인하여야 하며,
      자필서명 또는 전자서명으로 갈음할 수 있습니다. 청약 철회는
      청약일로부터 15일 이내<sup>1)</sup>에 가능합니다.</p>
      <p>보험회사는 계약 체결 시 계약자에게 약관 및 청약서 부본을
      교부하고 그 주요 내용을 설명하여야 합니다.</p>
    </div>
    <h2>1. 계약 기본사항</h2>
    {_table(header, rows)}
    <div class="box">
      <p class="chk">◈ 중요사항 확인</p>
      <p class="chk">☑ 약관 및 청약서 부본을 수령하였습니다.</p>
      <p class="chk">☑ 상품설명서의 주요 내용을 설명받았습니다.</p>
      <p class="chk">☐ 보험료 자동이체 신청에 동의합니다.</p>
      <p class="chk">☑ 개인정보 수집·이용에 동의합니다. (필수)</p>
    </div>
    <p class="foot">1) 진단계약, 보험기간 1년 미만 계약, 전문보험계약자 계약은 청약철회가 제한됩니다.</p>
    """
    return {
        "doc_id": "L3_청약서",
        "level": 3,
        "note": "2단 레이아웃 + 체크박스 상태 + 각주. 읽기 순서와 ☑/☐ 구분이 관건.",
        "html": html,
        "tables": [{"name": "계약_기본사항", "header": header, "rows": rows}],
        "fields": [
            {"key": "계약자", "anchor": "계약자", "value": "김실습"},
            {"key": "수익자", "anchor": "수익자", "value": "법정상속인"},
            {"key": "청약철회 기간", "anchor": "청약 철회", "value": "15일"},
        ],
        # 문서 전체의 ☑ 개수: 표 3개 + 확인란 3개 = 6. 체크박스 상태 인식 능력을
        # 따로 채점하기 위해 필드와 분리한다.
        "checkbox": {"checked": 6, "unchecked": 2},
    }


def level4():
    base = level2()
    return {
        "doc_id": "L4_스캔_요약손익계산서",
        "level": 4,
        "note": "L2 와 내용은 같고 스캔본처럼 기울고 흐리다. 정답은 L2 와 동일.",
        "html": f'<div class="scan">{base["html"]}</div>',
        "tables": base["tables"],
        "fields": base["fields"],
    }


ALL = [level1, level2, level3, level4]
