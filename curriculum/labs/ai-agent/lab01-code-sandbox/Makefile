.PHONY: build test demo clean

IMAGE ?= agent-sandbox:py

build:                ## 샌드박스 이미지 빌드
	docker build -t $(IMAGE) sandbox_image/

test: build           ## 공격 테스트 11종 실행 (모두 막혀야 정상)
	python3 run_attacks.py

breach: build         ## 방어를 끄면 무엇이 뚫리는지 시연
	python3 run_attacks.py --disable readonly --disable user --only A03 --only A04 --only A08

demo: build           ## LLM 에이전트가 샌드박스를 도구로 쓰는 데모
	python3 agent_demo.py "부채비율 상위 3개 회사와 그 수치를 알려줘"

clean:                ## 남은 컨테이너/산출물 정리
	-docker ps -aq --filter "name=sbx-" | xargs -r docker rm -f
	rm -rf .out
