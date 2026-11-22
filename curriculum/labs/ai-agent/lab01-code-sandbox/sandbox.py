"""에이전트가 생성한 파이썬 코드를 격리 컨테이너에서 실행하는 도구.

설계 원칙
---------
1. 컨테이너는 1회용이다. 요청 하나 = 컨테이너 하나 = 실행 후 삭제.
2. 기본은 전면 차단이다. 네트워크, 쓰기, 권한, 자원 모두 명시적으로 열어줘야 열린다.
3. 입력은 읽기 전용, 출력은 tmpfs. 호스트 디스크에 직접 쓰는 경로를 만들지 않는다.
4. 실패는 조용히 넘어가지 않는다. 타임아웃/OOM/차단은 모두 결과 객체에 남는다.

사용:
    from sandbox import run_python
    result = run_python("print(1 + 1)")
    print(result.stdout)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

IMAGE = "agent-sandbox:py"

# 이 상수들이 이 실습의 핵심이다. 하나씩 빼면서 무엇이 뚫리는지 확인해 볼 것.
DEFAULT_TIMEOUT = 30          # 초. 무한 루프 방어
DEFAULT_MEMORY = "512m"       # 메모리 폭탄 방어
DEFAULT_CPUS = "1.0"          # CPU 독점 방어
DEFAULT_PIDS = 64             # fork bomb 방어
TMP_SIZE = "64m"              # /tmp 채우기 방어
OUT_SIZE = "32m"              # 산출물용 tmpfs 상한 (out_dir 미지정 시)

NOBODY = "65534:65534"


@dataclass
class SandboxResult:
    """실행 결과. 에이전트에게는 이 객체를 요약해서 돌려준다."""

    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    oom_killed: bool = False
    artifacts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, max_chars: int = 4000) -> str:
        """LLM에게 되돌려줄 문자열. 길이를 반드시 잘라서 컨텍스트를 지킨다."""
        if self.timed_out:
            head = f"[TIMEOUT] {DEFAULT_TIMEOUT}초 내에 끝나지 않아 강제 종료했습니다."
        elif self.oom_killed:
            head = f"[OOM] 메모리 한도({DEFAULT_MEMORY})를 초과해 종료됐습니다."
        elif self.ok:
            head = "[OK]"
        else:
            head = f"[EXIT {self.exit_code}]"

        parts = [head]
        if self.stdout.strip():
            parts.append("--- stdout ---\n" + self.stdout.strip())
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + self.stderr.strip())
        if self.artifacts:
            parts.append("--- 생성된 파일 ---\n" + "\n".join(self.artifacts))
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (총 {len(text)}자 중 {max_chars}자만 표시)"
        return text


def _docker(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_python(
    code: str,
    *,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    network: bool = False,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
    pids_limit: int = DEFAULT_PIDS,
    read_only: bool = True,
    drop_caps: bool = True,
    run_as_nobody: bool = True,
    image: str = IMAGE,
) -> SandboxResult:
    """`code`를 격리 컨테이너에서 실행한다.

    Args:
        code: 실행할 파이썬 소스. 에이전트가 생성한 문자열을 그대로 넣는다.
        data_dir: 컨테이너의 /work/in 에 **읽기 전용**으로 붙일 호스트 디렉터리.
        out_dir: 컨테이너의 /work/out 에 쓰기 가능하게 붙일 호스트 디렉터리.
            지정하지 않으면 /work/out 은 크기 제한된 tmpfs 가 되어 산출물이 버려진다.
        network: True면 외부 통신을 허용한다. 기본은 완전 차단.
        read_only/drop_caps/run_as_nobody: 실습에서 하나씩 꺼 보기 위한 스위치.
            운영에서는 절대 끄지 않는다.
    """
    stage = Path(tempfile.mkdtemp(prefix="sbx-"))
    name = f"sbx-{uuid.uuid4().hex[:12]}"
    try:
        (stage / "main.py").write_text(code, encoding="utf-8")
        # 컨테이너 안의 nobody 가 읽을 수 있어야 한다.
        (stage / "main.py").chmod(0o644)
        stage.chmod(0o755)

        create_args = [
            "create",
            "--name", name,
            "--network", "bridge" if network else "none",
            "--memory", memory,
            "--memory-swap", memory,          # 스왑으로 우회하지 못하게 동일값
            "--cpus", cpus,
            "--pids-limit", str(pids_limit),
            "--security-opt", "no-new-privileges",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={TMP_SIZE},mode=1777",
            "-v", f"{stage}:/work/code:ro",
            "-w", "/work",
        ]
        if read_only:
            create_args += ["--read-only"]
        if drop_caps:
            create_args += ["--cap-drop", "ALL"]
        # 이미지에도 USER 가 박혀 있으므로, 실습에서 방어를 끌 때는 root 를
        # 명시적으로 되돌려야 "무엇이 뚫리는지"가 실제로 보인다.
        create_args += ["--user", NOBODY if run_as_nobody else "0:0"]
        if data_dir is not None:
            create_args += ["-v", f"{Path(data_dir).resolve()}:/work/in:ro"]
        if out_dir is None:
            # 산출물을 안 받는 경우에도 /work/out 은 존재해야 코드가 깨지지 않는다.
            create_args += ["--tmpfs", f"/work/out:rw,size={OUT_SIZE},mode=1777"]
        else:
            host_out = Path(out_dir).resolve()
            host_out.mkdir(parents=True, exist_ok=True)
            host_out.chmod(0o777)   # 컨테이너의 nobody 가 써야 한다
            create_args += ["-v", f"{host_out}:/work/out"]
        create_args += [image, "python", "/work/code/main.py"]

        created = _docker(*create_args, timeout=60)
        if created.returncode != 0:
            return SandboxResult(
                exit_code=created.returncode,
                stdout="",
                stderr=f"컨테이너 생성 실패: {created.stderr.strip()}",
                duration=0.0,
            )

        started = time.monotonic()
        timed_out = False
        try:
            _docker("start", "-a", name, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _docker("kill", name, timeout=30)
        duration = time.monotonic() - started

        # 로그는 컨테이너에서 다시 읽는다. start -a 가 중간에 끊겨도 남아 있다.
        logs = _docker("logs", name, timeout=60)
        inspect = _docker(
            "inspect", "-f",
            "{{.State.ExitCode}}|{{.State.OOMKilled}}", name,
            timeout=30,
        )
        exit_code: int | None = None
        oom = False
        if inspect.returncode == 0:
            raw_code, _, raw_oom = inspect.stdout.strip().partition("|")
            exit_code = int(raw_code) if raw_code.lstrip("-").isdigit() else None
            oom = raw_oom.strip().lower() == "true"

        artifacts: list[str] = []
        if out_dir is not None:
            dest = Path(out_dir).resolve()
            artifacts = sorted(
                str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()
            )

        return SandboxResult(
            exit_code=exit_code,
            stdout=logs.stdout,
            stderr=logs.stderr,
            duration=duration,
            timed_out=timed_out,
            oom_killed=oom,
            artifacts=artifacts,
        )
    finally:
        # 컨테이너를 남기지 않는다. 이걸 빼먹으면 수업 한 번에 수백 개가 쌓인다.
        _docker("rm", "-f", name, timeout=60)
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    demo = "import pandas as pd; print(pd.DataFrame({'a':[1,2,3]}).sum().to_json())"
    r = run_python(demo)
    print(json.dumps({"ok": r.ok, "stdout": r.stdout.strip(), "sec": round(r.duration, 2)},
                     ensure_ascii=False))
