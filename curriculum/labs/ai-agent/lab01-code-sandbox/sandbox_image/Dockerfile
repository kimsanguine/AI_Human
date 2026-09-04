# 에이전트가 생성한 코드를 실행할 격리 이미지.
# 원칙: 필요한 것만 넣는다. 셸 유틸리티, 네트워크 도구, 패키지 관리자 흔적을 남기지 않는다.
FROM python:3.11-slim

# 분석용 최소 라이브러리만. 여기에 requests/httpx 를 넣지 않는 것도 방어의 일부다.
RUN pip install --no-cache-dir \
        pandas==2.2.3 \
        numpy==1.26.4 \
        matplotlib==3.9.2 \
 && find /usr/local -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && rm -rf /root/.cache

# 비루트 사용자(nobody). 런타임에서 --user 로 한 번 더 강제한다.
USER 65534:65534

# 헤드리스 환경에서 matplotlib 이 디스플레이를 찾지 않도록.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work
