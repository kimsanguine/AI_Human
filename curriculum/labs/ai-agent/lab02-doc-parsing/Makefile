.PHONY: dataset test compare preview clean

dataset:              ## 합성 문서 4종 + 정답 생성
	python3 docgen/generate.py

preview:              ## 미리보기 PNG까지 생성
	python3 docgen/generate.py --png

test:                 ## 채점기 자체 검증 (파서 없이 동작)
	python3 test_score.py

compare:              ## 사용 가능한 파서 전부 비교
	python3 run_compare.py --save

clean:
	rm -rf out dataset/*.png dataset/*.html __pycache__ */__pycache__
