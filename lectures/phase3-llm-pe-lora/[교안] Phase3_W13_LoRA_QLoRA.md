# Phase 3 — W13: LoRA/QLoRA Fine-tuning (Gap ① 핵심 주차)

> KDT AI Human 5기 | Phase 3 (W11-13) "LLM + PE + 음성AI + LoRA"
> 주차: W13 (Day 61-65)
> 작성일: 2026-03-05
> 강사: Ethan (김생근)
> 상태: 초안 v1

---

## 주간 개요

| 항목 | 내용 |
|------|------|
| 주제 | **LoRA/QLoRA를 활용한 sLLM Fine-tuning** |
| 선수지식 | W11-12 — LLM 기초, PE, API 활용, 모델 평가 |
| 목표 | 소형 LLM을 QLoRA로 효율적으로 Fine-tuning하고 배포할 수 있다 |
| 산출물 | 본인이 Fine-tuning한 모델 + HuggingFace Hub 배포 |
| Gap 해소 | **① QLoRA — 이번 주가 핵심** |
| GPU 필요 | ⚠️ **필수** — Colab Pro (T4/A100) 또는 학원 GPU 서버 |

### 왜 이번 주가 중요한가?

W12D4에서 경험한 "PE로 안 되는 것들"을 해결하는 주차. 채용시장 Tier 2 스킬(55%+)인 LoRA/QLoRA를 직접 다루고, "BERT FT는 해봤지만 LLM FT는 처음"인 수강생을 "sLLM FT를 해봤고, 허브에 올려봤다"로 업그레이드.

### 3-Layer 시스템 적용

| Layer | W13 적용 |
|-------|---------|
| L1: AI Pair Coding | FT 코드 작성 시 Claude 활용 (PEFT 코드 디버깅) |
| L2: Product Thinking | "FT가 필요한 비즈니스 시나리오는?" |
| L3: Thought Process | 하이퍼파라미터 선택 근거, 실험 결과 분석 문서화 |

---

## D1 (Day 61): Fine-tuning 이론 — 왜, 언제, 어떻게

### 오전 (09:30-12:30) — FT 개념과 LoRA 이론

**[09:30-10:30] Fine-tuning의 전체 그림**
- Pre-training → Fine-tuning → Inference 파이프라인
- Full Fine-tuning의 문제:
  - 7B 모델 = 약 28GB GPU 메모리 (FP32 기준)
  - 학습 시 optimizer state까지 = 3-4배 → 100GB+ 필요
  - Q. "우리가 이걸 할 수 있나?" → "LoRA가 답이다"
- Fine-tuning 스펙트럼:
  | 방법 | 수정 범위 | 메모리 | 성능 |
  |------|---------|--------|------|
  | Full FT | 전체 파라미터 | 매우 높음 | 최고 |
  | LoRA | 저랭크 어댑터만 | 낮음 | 높음 |
  | QLoRA | LoRA + 4bit 양자화 | 매우 낮음 | 높음 |
  | Prompt Tuning | 프롬프트 임베딩만 | 최소 | 보통 |
  | Prefix Tuning | 접두어 벡터만 | 최소 | 보통 |

**[10:40-11:30] LoRA 깊이 이해**
- Low-Rank Adaptation 핵심 아이디어:
  - 원래 가중치 W (d×d) 를 동결
  - 작은 행렬 A (d×r) × B (r×d) 를 학습 (r << d)
  - 결과: W + ΔW = W + A×B
  - r=8이면? 수십억 파라미터 → 수백만 파라미터만 학습
- 직관적 설명:
  - "큰 그림은 그대로 두고, 작은 스티커를 붙여서 커스터마이징"
  - Phase 2 선형대수 연결: 행렬 분해, 저랭크 근사
- PEFT 라이브러리 소개:
  - HuggingFace의 Parameter-Efficient Fine-Tuning
  - `LoraConfig`, `get_peft_model`, `PeftModel`

**[11:30-12:30] QLoRA — 메모리 효율의 극대화**
- 양자화 기초:
  - FP32 → FP16 → INT8 → INT4
  - 4-bit 양자화: NF4 (Normal Float 4) 데이터 타입
- QLoRA = 4-bit 양자화 + LoRA:
  - 4-bit 양자화로 모델 로드 → 메모리 4배 절약
  - LoRA 어댑터만 FP16으로 학습
  - 이중 양자화 (Double Quantization)
- BitsAndBytes 라이브러리:
  ```python
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
  )
  ```
- Q. "4-bit으로 줄이면 성능이 떨어지지 않나?" → 벤치마크 결과 공유

### 오후 (13:30-17:30) — 환경 셋업 + 첫 실습

**[13:30-14:30] 실습 환경 셋업**
- Google Colab Pro에서 GPU 런타임 설정
- 필수 패키지 설치:
  ```bash
  pip install transformers peft bitsandbytes accelerate datasets trl
  pip install wandb  # 실험 추적용
  ```
- GPU 확인: `torch.cuda.get_device_name(0)`
- WandB 계정 연동

**[14:30-16:00] 첫 QLoRA 실습: 감성분류 FT**
- 목표: Gemma-2B를 한국어 감성분류에 특화
- 데이터셋: NSMC (Naver Sentiment Movie Corpus)
- 단계별 코드:
  1. 모델 로드 (4-bit):
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         "google/gemma-2b",
         quantization_config=bnb_config,
         device_map="auto",
     )
     ```
  2. LoRA 설정:
     ```python
     lora_config = LoraConfig(
         r=8, lora_alpha=16, lora_dropout=0.05,
         target_modules=["q_proj", "v_proj"],
         task_type="CAUSAL_LM",
     )
     model = get_peft_model(model, lora_config)
     model.print_trainable_parameters()
     # → "trainable: 0.5% of total"
     ```
  3. 학습 데이터 준비 (프롬프트 포맷팅)
  4. Trainer 설정 + 학습 시작

**[16:00-17:00] 학습 모니터링**
- WandB 대시보드에서 loss 곡선 관찰
- Q. "loss가 떨어지는데, 이게 잘 되고 있는 건가?"
- Overfitting 징후 관찰: train loss vs eval loss
- 학습이 돌아가는 동안 이론 복습 + Q&A

**[17:00-17:30] D1 정리**
- 오늘의 학습 결과 (loss 곡선) 캡처
- L3: "LoRA의 핵심 아이디어를 비개발자에게 설명한다면?" 1문단 작성
- Git push (코드 + 학습 설정)

### D1 Ethan's Note
> 이론이 무거운 날. 수학적 직관은 "행렬을 잘게 쪼개서 효율적으로 학습"으로 충분. 핵심은 "우리가 가진 GPU로도 LLM을 Fine-tuning할 수 있다!"는 체험. loss가 내려가는 걸 보는 순간 "와" 반응이 나옴. 그 경험에 집중.

---

## D2 (Day 62): QLoRA 실전 — 다양한 태스크 Fine-tuning

### 오전 (09:30-12:30) — 태스크별 FT 전략

**[09:30-10:30] D1 결과 리뷰 + 평가**
- D1에서 학습한 모델 평가:
  - FT 전 vs FT 후 성능 비교
  - 테스트 데이터 10개로 직접 비교
- 실습: 평가 스크립트 작성
  ```python
  # FT 전: base model에 프롬프트로 감성분류
  # FT 후: fine-tuned model로 감성분류
  # 정확도, 응답 형식, 속도 비교
  ```

**[10:40-11:30] 데이터셋 준비의 기술**
- 학습 데이터 포맷:
  - Instruction format: `### Instruction:\n{task}\n### Input:\n{text}\n### Response:\n{answer}`
  - Chat format: `<|user|>\n{query}\n<|assistant|>\n{answer}`
- 데이터 품질이 모든 것:
  - 최소 데이터 양: 100-1000개 (태스크에 따라)
  - 데이터 정제: 중복 제거, 라벨 검증, 균형 조정
- 실습: 커스텀 데이터셋 만들기
  - LLM으로 학습 데이터 생성 (합성 데이터)
  - Q. "LLM으로 데이터를 만들어서 LLM을 학습시킨다고?" → Self-improvement 패러다임

**[11:30-12:30] 하이퍼파라미터 튜닝**
- 핵심 하이퍼파라미터:
  | 파라미터 | 역할 | 권장 범위 |
  |---------|------|---------|
  | r (rank) | LoRA 랭크 | 4, 8, 16, 32 |
  | lora_alpha | 스케일링 | r×2 (보통) |
  | learning_rate | 학습률 | 1e-4 ~ 5e-4 |
  | epochs | 학습 횟수 | 1~3 |
  | batch_size | 배치 크기 | GPU에 따라 |
  | lora_dropout | 드롭아웃 | 0.05~0.1 |
- 실습: r=4 vs r=16 비교 실험
  - WandB에 두 실험 동시 기록

### 오후 (13:30-17:30) — 새로운 태스크로 FT

**[13:30-15:00] 실습: 요약 모델 Fine-tuning**
- 목표: Gemma-2B를 한국어 뉴스 요약에 특화
- 데이터: AI 뉴스 기사 50개 + 사람이 작성한 요약
- 과정:
  1. 데이터셋 준비 (Instruction format)
  2. QLoRA 설정 (D1 코드 재활용, 하이퍼파라미터만 변경)
  3. 학습 실행
  4. 평가: ROUGE 스코어 계산

**[15:10-16:30] 실습: 코드 생성 모델 Fine-tuning (선택)**
- 목표: 코드 생성에 특화된 소형 모델
- 데이터: Python 코딩 문제 + 솔루션 100개
- 또는: 본인이 원하는 태스크로 자유 FT
  - 번역, 분류, 질의응답, 대화 등
- 핵심: "같은 기법, 다른 데이터 → 다른 모델" 체험

**[16:30-17:30] D2 정리**
- 실험 결과 비교 (감성분류 vs 요약 vs 선택태스크)
- WandB 대시보드 캡처 + 분석
- L3: "어떤 태스크가 FT 효과가 가장 컸나? 왜?" 분석 작성
- Git push

### D2 Ethan's Note
> 이 날의 핵심은 "재현 가능성". D1 코드를 약간만 바꿔서 다른 태스크에 적용하는 경험이 중요. "아, 이 패턴을 알면 어떤 태스크든 FT할 수 있구나"라는 자신감 형성. 합성 데이터 생성은 현실적인 실무 팁.

---

## D3 (Day 63): Unsloth로 고속 Fine-tuning

### 오전 (09:30-12:30) — Unsloth 소개 + 실습

**[09:30-10:30] Unsloth — 왜 빠른가?**
- Unsloth 핵심 특징:
  - LoRA 학습 속도 2-5배 향상
  - 메모리 사용량 60-80% 절감
  - HuggingFace Transformers 호환
- 어떻게?
  - 커스텀 CUDA 커널
  - 최적화된 메모리 관리
  - 불필요한 연산 제거
- D1-D2 (기본 PEFT) vs D3 (Unsloth) 비교 예고

**[10:40-12:30] Unsloth 실습: Gemma-2B FT**
- Unsloth 설치 + 모델 로드:
  ```python
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/gemma-2b-bnb-4bit",
      max_seq_length=2048,
      load_in_4bit=True,
  )
  model = FastLanguageModel.get_peft_model(
      model, r=16, lora_alpha=16,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0,
  )
  ```
- D1과 동일한 감성분류 태스크로 학습
- 비교 측정:
  | 측정 항목 | 기본 PEFT | Unsloth |
  |---------|----------|---------|
  | 학습 시간 | ? min | ? min |
  | GPU 메모리 | ? GB | ? GB |
  | 최종 loss | ? | ? |
  | 평가 정확도 | ? % | ? % |

### 오후 (13:30-17:30) — 다양한 모델 + 고급 기법

**[13:30-15:00] 다른 모델로 FT**
- Gemma-2B 외에 시도:
  | 모델 | 크기 | 특징 |
  |------|------|------|
  | Qwen2.5-1.5B | 1.5B | 중국어/영어 강점 |
  | Phi-3-mini | 3.8B | 추론 강점 |
  | LLaMA-3.2-3B | 3B | Meta, 범용 |
- 실습: 모델 1개 선택 → Unsloth로 FT → Gemma-2B와 비교
- Q. "어떤 베이스 모델을 선택할 것인가?" 의사결정 프레임워크

**[15:10-16:30] FT 고급 기법 소개**
- DPO (Direct Preference Optimization):
  - RLHF 대안: 선호 데이터로 직접 최적화
  - 개념만 소개 (실습은 캡스톤에서 선택)
- LoRA 병합과 모델 저장:
  ```python
  model = model.merge_and_unload()
  model.save_pretrained("my-finetuned-model")
  tokenizer.save_pretrained("my-finetuned-model")
  ```
- GGUF 변환 (Ollama에서 사용):
  - "내가 만든 모델을 로컬에서 돌린다!"

**[16:30-17:30] D3 정리**
- 속도/메모리 비교 결과 정리
- 모델 저장 + GGUF 변환 시도
- L3: "Unsloth 같은 최적화 도구가 왜 중요한가?" 비용 관점 분석
- Git push

### D3 Ethan's Note
> Unsloth의 "체감 속도"가 핵심 포인트. D1에서 30분 걸리던 학습이 10분에 끝나는 경험. GGUF 변환 → Ollama 실행은 "내가 만든 모델을 내 컴퓨터에서 돌린다"는 완결감. DPO는 개념만 — 실습은 무리.

---

## D4 (Day 64): HuggingFace Hub 배포 + 모델 공유

### 오전 (09:30-12:30) — HuggingFace Hub 배포

**[09:30-10:30] HuggingFace Hub 이해**
- Hub의 역할: 모델 저장소 + 버전 관리 + 커뮤니티
- Model Card 작성법:
  - 모델 설명, 사용법, 학습 데이터, 성능, 제한사항
  - 책임 있는 AI: 편향, 제한사항 명시
- 실습: 좋은 Model Card 분석 (인기 모델 3개)

**[10:40-11:30] 모델 업로드 실습**
- HuggingFace Hub에 모델 Push:
  ```python
  from huggingface_hub import HfApi
  model.push_to_hub("username/my-korean-sentiment-model")
  tokenizer.push_to_hub("username/my-korean-sentiment-model")
  ```
- Model Card 작성:
  ```yaml
  ---
  language: ko
  tags:
    - sentiment-analysis
    - qlora
    - gemma
  datasets:
    - nsmc
  ---
  # My Korean Sentiment Model
  ...
  ```
- 실습: 본인 모델 업로드 + Model Card 작성

**[11:30-12:30] 배포된 모델 사용하기**
- 다른 사람의 모델 다운로드 + 추론:
  ```python
  model = AutoModelForCausalLM.from_pretrained("username/my-model")
  ```
- HuggingFace Inference API:
  - 업로드하면 자동으로 API 엔드포인트 생성
  - 무료 체험 가능
- Q. "포트폴리오에 HuggingFace 링크를 넣으면?" → 채용 담당자의 시선

### 오후 (13:30-17:30) — 종합 실습 + 최적화

**[13:30-15:00] 자유 Fine-tuning 프로젝트**
- 본인이 원하는 태스크 선택:
  - 옵션 A: 한국어 대화 모델 (캐릭터 챗봇)
  - 옵션 B: 특정 도메인 QA 모델 (의료/법률/기술)
  - 옵션 C: 코드 생성 특화 모델
  - 옵션 D: 자유 주제
- 전체 파이프라인: 데이터 준비 → FT → 평가 → Hub 배포

**[15:10-16:30] 프로젝트 개발 계속**
- Ethan 순회 코칭
- 직원 호출: GPU 이슈, CUDA 에러 등
- WandB로 실험 추적

**[16:30-17:30] D4 정리**
- 모델 Hub 배포 완료 확인
- Model Card 상호 리뷰 (짝꿍 피드백)
- L3: "내 모델의 강점과 한계" 분석
- Git push

### D4 Ethan's Note
> Hub 배포는 "나만의 모델이 세상에 공개되었다"는 성취감. Model Card 작성은 PM의 "제품 문서화" 역량과 직결. 채용 면접에서 "HuggingFace에 올린 모델이 있다"고 말할 수 있다는 것의 의미를 강조.

---

## D5 (Day 65): W13 종합 + Phase 3 마무리

### 오전 (09:30-12:30) — 미니 프로젝트 마무리 + 발표

**[09:30-10:30] 프로젝트 최종 정리**
- D4 자유 프로젝트 마무리
- 발표 자료 준비:
  1. 문제 정의 (왜 이 태스크?)
  2. 데이터 준비 과정
  3. FT 설정 (하이퍼파라미터 선택 근거)
  4. 실험 결과 (WandB 대시보드)
  5. FT 전 vs FT 후 비교 데모
  6. Hub 링크

**[10:30-12:30] 발표 (1인 7분)**
- 기술 데모 + 의사결정 근거 설명
- 동료 피드백 (각 발표 후 3분)
- Ethan 코멘트: 실무 관점 피드백

### 오후 (13:30-17:30) — Phase 3 종합 회고 + Phase 4 프리뷰

**[13:30-14:30] Phase 3 종합 회고**
- Phase 3 (W11-W13) 전체 KPT:
  - Keep: Phase 3에서 가장 잘한 것
  - Problem: 가장 어려웠던 것
  - Try: Phase 4에서 시도할 것
- 기술 역량 자가 평가:
  | 기술 | W11 이전 | W13 이후 |
  |------|---------|---------|
  | LLM 이해 | ☆☆☆ | ★★★ |
  | Prompt Engineering | ☆☆☆ | ★★★ |
  | 음성 AI | ☆☆☆ | ★★★ |
  | LoRA/QLoRA | ☆☆☆ | ★★★ |
  | 모델 배포 | ☆☆☆ | ★★★ |

**[14:30-15:30] Phase 4 프리뷰**
- W14-W17 로드맵 소개:
  | 주차 | 주제 | 핵심 |
  |------|------|------|
  | W14 | LangChain | "LLM을 시스템으로 연결" |
  | W15 | RAG | "외부 지식을 LLM에 연결" |
  | W16 | Agent | "LLM이 스스로 행동" |
  | W17 | 서비스 프로토타이핑 | "만든 것을 세상에 보여주기" |
- Q. "Phase 3에서 배운 것이 Phase 4에 어떻게 연결되는가?"
  - PE → LangChain Prompt Templates
  - LLM API → LangChain LLM 래퍼
  - FT 모델 → RAG + Agent의 백본
  - 음성 → 멀티모달 Agent 입력

**[15:30-16:30] 포트폴리오 중간 정리**
- Phase 1-3 결과물 GitHub 정리:
  - README 업데이트 (프로젝트 목록)
  - 폴더 구조 정돈
  - 각 프로젝트별 간단한 설명 추가
- HuggingFace 프로필 정리:
  - 모델 카드 최종 점검
  - 프로필 bio 업데이트

**[16:30-17:30] 1:1 면담 + 캡스톤 아이디어 씨앗**
- 개인별 5분 면담:
  - 진도 상담
  - 캡스톤 프로젝트 방향 브레인스토밍
  - "Phase 4에서 특히 집중하고 싶은 것?"

### D5 Ethan's Note
> Phase 3 마무리는 "연결의 날". Phase 3에서 배운 것들이 Phase 4의 재료가 된다는 것을 명확히 보여줘야 함. 캡스톤 아이디어를 이 시점에서 심어두면, Phase 4를 "캡스톤 준비"로 목적의식 있게 진행할 수 있음.

---

## W13 준비물 체크리스트

| 항목 | 상세 | 담당 |
|------|------|------|
| GPU 환경 | ⚠️ Colab Pro (T4 이상) 또는 학원 GPU 서버 | Ethan + 직원 (사전 테스트 필수) |
| HuggingFace | 계정 + Access Token (write 권한) | 수강생 (W12D5 사전과제) |
| WandB | 계정 생성 + API 키 | 수강생 (D1 전) |
| 데이터셋 | NSMC, 뉴스 요약 데이터, 코딩 데이터 | Ethan (사전 준비) |
| Python 패키지 | `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `trl`, `unsloth`, `wandb` | Ethan (requirements.txt + Colab 노트북) |

## W13 숙제/사전과제

| 시점 | 과제 |
|------|------|
| W12→W13 | Colab Pro 확인, HuggingFace 토큰 발급, WandB 계정 |
| D1→D2 | D1 학습 결과 분석 + 하이퍼파라미터 변경 아이디어 |
| D2→D3 | 자유 FT 태스크 구상 + 데이터셋 조사 |
| D3→D4 | 자유 FT 데이터셋 최소 50개 준비 |
| D5→W14 | Phase 3 포트폴리오 최종 정리, LangChain 공식문서 훑어보기 |

---

## Phase 3 전체 흐름 요약

```
W11: LLM 이해 + PE 기초/심화 + API 활용
  ↓ "LLM을 잘 쓰려면 프롬프트가 핵심"
W12: 음성 AI(STT/TTS) + LLM 결합 + 모델 평가
  ↓ "PE로 안 되는 것도 있다 → Fine-tuning 필요"
W13: LoRA/QLoRA + Unsloth + Hub 배포
  ↓ "이제 나만의 모델이 있다 → Phase 4에서 시스템으로 연결"
```

| 주차 | 핵심 성취 | 산출물 |
|------|---------|--------|
| W11 | LLM+PE 마스터 | PE 포트폴리오 5개+ |
| W12 | 음성AI+모델평가 | 음성 AI 어시스턴트 |
| W13 | QLoRA FT+배포 | HuggingFace 모델 배포 |
