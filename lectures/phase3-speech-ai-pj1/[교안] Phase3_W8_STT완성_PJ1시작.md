# Phase 3 — W8: STT 완성 + PJ1 시작

> KDT AI Human 5기 | Phase 3 (W7-10) "음성AI + PJ1 음성모델"
> 주차: W8 (Day 36-40)
> 작성일: 2026-03-24
> 강사: Ethan (김생근)
> 상태: 초안 v1 (A1/A2 개선 적용)

---

## 주간 개요

| 항목 | 내용 |
|------|------|
| 주제 | STT 고급 기능(화자인식, 감정분석) + PJ1 음성모델 프로젝트 시작 |
| 선수지식 | W7 — Whisper 기초, 음성 스펙트로그램, TTS/STT 파이프라인 이해 |
| 목표 | Whisper의 고급 기능을 활용하고, 프로젝트 기획→설계까지 완료할 수 있다 |
| 산출물 | ① STT 모델 평가 리포트 ② PJ1 기획서 ③ 아키텍처 설계 문서 ④ 데이터 전처리 파이프라인 |
| RD 매핑 | 과목5 "음성 데이터 활용 TTS와 STT 모델 개발" (8h) + PJ1 "나만의 음성 모델 만들기" (32h) |
| Gap 요소 | ③ Git 실전 — PJ1 리포 초기화, .gitignore 설정 |

### 🎯 채용시장 연결

> **음성AI 엔지니어가 반드시 가져야 할 경험:**
> - Whisper 같은 오픈소스 모델을 **프로덕션에 어떻게 적용**하는가?
> - 모델 평가(WER, CER) 경험 — "내 모델이 정말 좋은지 어떻게 판단?"
> - **기획→설계→구현 경험** — "개발자는 무엇을 갖춰야 하나?" 채용 체크리스트 #1
> - **PERSO AI 같은 실제 제품 SDK와 통합**한 경험 = "실무 경험"으로 강력 어필

### 3-Layer 시스템 적용

| Layer | W8 적용 |
|-------|--------|
| L1: AI Pair Coding | ChatGPT로 "Whisper 화자 분리 코드 작성해줘"로 시작, 개선 |
| L2: Product Thinking | "내가 만든 STT 모델이 PERSO AI에서 어떤 문제를 푸는가?" |
| L3: Thought Process | 기획서 작성 (사용자 시나리오 → 기술 선택 이유 정리) |

---

## D1 (Day 36): STT 고급 기능 — 화자인식 + 감정분석

> **오늘의 핵심 질문**: "기계가 '누가' 말했는지, '어떤 감정으로' 말했는지 알 수 있을까?"

### 오전 (09:30-12:30) — 이론 + 개념 실습

**[09:30-10:30] 음성 감정 인식 (Speech Emotion Recognition, SER)**

- **Q. 왜 감정 분석이 필요한가?**
  - 콜센터 자동화: 고객 불만 수위 자동 감지 → 우선순위 라우팅
  - AI 더빙/아바타: "텍스트의 감정"과 "음성의 감정"을 맞춰야 자연스러움
  - Phase 2에서 배운 감정분석 = **텍스트 기반** → 이제 **음성 신호**로 확장

- **감정 분류의 어려움**:
  - 텍스트: "화났어"라고 써있으면 명확
  - **음성**: 같은 문장도 톤/속도/음높이에 따라 달라짐
  - 따라서 **음성의 특성(음에너지, 피치, MFCC 등)을 특징**으로 사용

- **음성 감정 인식 아키텍처**:
  ```
  음성 신호
    ↓
  특징 추출 (Mel-Spectrogram, MFCC, ProsodicFeature)
    ↓
  신경망 분류기 (CNN, RNN, Transformer)
    ↓
  감정 레이블 (행복, 슬픔, 분노, 중립 등)
  ```
  - Phase 2: Mel-Spectrogram = 2D 이미지 형태 → CNN으로 분류
  - 실제 SER에서: Mel-Spectrogram + MFCC 조합 → 더 강력

- **실습**: 공개 감정 데이터셋(RAVDESS) 로드 + 시청
  ```python
  import librosa
  import numpy as np

  # 공개 감정 데이터셋 라벨: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgusted, 08=surprised
  audio_path = "RAVDESS_data/03-01-05-01-01-01-01.wav"
  y, sr = librosa.load(audio_path)

  # 감정을 듣고 확인
  # 파일명: 03 = "happy", 01 = "first sentence"
  print(f"감정: Happy (Emotion code 03)")
  ```
  - "같은 문장인데도 감정을 구별할 수 있다" — 깨달음 순간

**[10:40-11:30] 화자 분리 (Speaker Diarization)**

- **Q. "회의 음성에서 누가 언제 말했는지 어떻게 알까?"**
  - STT 결과: "회의 기록"
  - STT + Diarization: **"김생근: 안녕, 이든: 반갑습니다" 형태**
  - Phase 2 연결: CNN으로 이미지 분류 → 이제 **음성 임베딩으로 사람 구별**

- **화자 분리 2가지 방식**:
  1. **Non-intrusive (더 간단)**: pyannote-audio 사용 — 미리 학습된 모델
  2. **Fine-tuning (심화)**: 내 데이터로 커스터마이징 — 현재는 1번만

- **pyannote-audio란?**
  - 프랑스 CNRS 연구진이 만든 최강 화자 분리 라이브러리
  - 사용 방법: "설치 → 모델 로드 → 오디오 넣기 → 화자별 타임스탬프 반환"

- **실습 준비**: HuggingFace 토큰 발급
  ```
  1. https://huggingface.co/ 가입
  2. Settings > Access Tokens > New token 생성
  3. 토큰을 .env에 저장: HUGGINGFACE_TOKEN=...
  ```

**[11:30-12:30] 멀티태스크 STT (Emotion + Speaker 동시 처리)**

- **왜 한 번에 여러 작업을 할까?**
  - "음성 한 번의 input으로 STT(텍스트) + 감정 + 화자" 동시 추출
  - 계산 효율: 3번 모델 돌리는 것보다 1번 멀티태스크 모델 더 빠름
  - Phase 2 Transformer: 하나의 encoder → 여러 decoder(task-specific) 아키텍처 상기

- **실무 적용 예시**:
  - 고객 센터: "김생근이 화났다"고 감지 → 우선 처리
  - 자동 회의록: 화자/감정/텍스트 한번에 정리

- **🔧 Ethan's Note**:
  > AI 더빙 서비스에서 가장 많이 실수하는 부분: 음성 감정과 텍스트 감정 불일치.
  > 예: "고마워요"(감정: 행복)를 슬픈 음성으로 생성하면 어색함.
  > 따라서 음성 감정을 미리 분석해서 TTS 파라미터(style, pitch, rate)를 조정해야 함.

### 오후 (13:30-17:30) — 실습 워크숍

**[13:30-15:00] 음성 감정 분류 실습**

**실습 목표**: RAVDESS 공개 데이터셋으로 감정 분류 모델 만들기

**코드 스켈레톤** (학생은 `# TODO` 부분만 작성):

```python
import librosa
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===================== PART 1: 데이터 로드 =====================
def load_audio_feature(file_path, sr=22050):
    """음성 파일에서 MFCC 특징 추출 (난이도 낮추기)"""
    y, sr = librosa.load(file_path, sr=sr)

    # MFCC (Mel-Frequency Cepstral Coefficients)
    # = 음성의 "음색"을 나타내는 특징 (13개 계수)
    # W7에서 배운 Mel-Spectrogram과 유사하지만, 더 간단한 형태
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # (13, time_steps)

    # 시간축으로 평균내기 (13개 특징만 남음)
    mfcc_mean = np.mean(mfcc, axis=1)  # (13,)

    return mfcc_mean

# 파일명 규칙: Emotion03 = Happy(03), 파일01 = Actor01
def parse_emotion_label(filename):
    """
    RAVDESS 파일명에서 감정 추출
    예: 03-01-05-01-01-01-01.wav → 03(Happy)
    """
    # TODO: 파일명 split, emotion code 추출
    # HINT: filename.split("-")[2]를 보면 emotion code
    # HINT2: emotion_map = {1: "neutral", 2: "calm", 3: "happy", ...}로 매핑
    emotion_code = int(filename.split("-")[2])
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgusted", 8: "surprised"
    }
    return emotion_map.get(emotion_code, "unknown")

# ===================== PART 2: 모델 정의 =====================
class EmotionClassifier(nn.Module):
    """간단한 감정 분류 신경망 (2-layer MLP)"""
    def __init__(self, input_size=13, num_classes=8):
        super().__init__()
        # TODO: nn.Linear(input_size, 64) 레이어 1 정의
        # TODO: nn.ReLU() 활성화 함수
        # TODO: nn.Linear(64, 32) 레이어 2 정의
        # TODO: nn.ReLU() 활성화 함수
        # TODO: nn.Linear(32, num_classes) 출력 레이어 정의
        # HINT: self.fc1, self.relu1, self.fc2, self.relu2, self.fc3로 이름 붙일 것
        pass

    def forward(self, x):
        """전 방향 패스"""
        # TODO: x를 fc1 → relu1 → fc2 → relu2 → fc3을 거쳐 출력
        return x

# ===================== PART 3: 데이터 준비 =====================
# 실제 RAVDESS 다운로드가 필요하므로 Toy 데이터로 대체
np.random.seed(42)
X_toy = np.random.randn(100, 13)  # 100 샘플, 13 특징
y_toy = np.random.randint(0, 8, 100)  # 8개 감정 클래스

X_train, X_test, y_train, y_test = train_test_split(X_toy, y_toy, test_size=0.2)

# 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# ===================== PART 4: 학습 =====================
model = EmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO: 5 에포크 학습 루프 작성
# HINT: for epoch in range(5):
# HINT2: forward pass → loss 계산 → backward → optimizer.step()
# HINT3: 매 에포크마다 loss 출력

# ===================== PART 5: 평가 =====================
# TODO: 테스트 데이터로 정확도(accuracy) 계산
# HINT: torch.argmax(model(X_test), dim=1)로 예측값 얻기
# HINT2: accuracy = (pred == y_test).float().mean()
```

**실습 과제**:
1. `parse_emotion_label` 함수 완성 (파일명 → 감정 추출)
2. `EmotionClassifier` 신경망 정의 완성
3. 학습 루프 구현
4. 테스트 정확도 계산

**학생이 구현해야 할 부분 (총 30줄 정도, 난이도 2.5/5)**:
- 함수형 코딩 (parse_emotion_label)
- PyTorch MLP 모델 정의 (Phase 2 복습)
- 표준 학습 루프 (Phase 2 복습)

**왜 이렇게 설계했나?**
- W7은 Whisper 모델 **사용** → 이번엔 **간단한 분류 모델 구축**
- 코드 스켈레톤 제공 = 딱 구현해야 할 부분만 집중
- TODO 주석과 HINT 제공 = "어디를 어떻게 채워야 하나" 명확히

**[15:10-16:30] 화자 분리(Diarization) 실습**

**실습 목표**: pyannote-audio로 회의 음성에서 화자 구분하기

**기본 개념 설명** (15분):
- **왜 화자 분리가 어려운가?**
  - 입력: 여러 사람이 섞인 음성
  - 출력: 시간대별로 "누가 말했는가" (Speaker 1, Speaker 2, ...)
  - 알고리즘은 **오디오 신호 자체**로만 사람을 구별해야 함 (얼굴/지문 없이)

- **방법**: 음성 임베딩 (음성의 "사람 지문" 추출)
  - Phase 2의 word2vec = 단어 → 벡터
  - 이제: 음성 세그먼트 → 임베딩 벡터
  - "사람 A와 사람 B의 임베딩이 다르다" → 화자 분리 가능

**코드 스켈레톤**:

```python
# ===================== pyannote-audio 설치 + 로드 =====================
# 터미널: pip install pyannote-audio==3.0.0

from pyannote.audio import Pipeline
import torch

# 사전학습 모델 다운로드 (처음 1회만, ~500MB)
# HuggingFace 토큰 필요 (처음 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="YOUR_HF_TOKEN"  # .env에서 읽을 것
).to(device)

# ===================== 회의 음성 분석 =====================
audio_file = "meeting_recording.wav"

# TODO: pipeline(audio_file)을 호출해서 diarization 결과 받기
# diarization = ???
# diarization 결과 형태:
# [
#   (Segment(start=0.5, end=5.2), None, "Speaker 1"),
#   (Segment(start=5.3, end=10.1), None, "Speaker 2"),
#   (Segment(start=10.2, end=15.0), None, "Speaker 1"),
#   ...
# ]

# ===================== 결과 시각화 =====================
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f}s ~ {turn.end:.2f}s: {speaker}")
    # 출력 예:
    # 0.50s ~ 5.20s: Speaker 1
    # 5.30s ~ 10.10s: Speaker 2
    # 10.20s ~ 15.00s: Speaker 1

# ===================== STT와 결합 =====================
# TODO: Whisper로 전체 오디오 STT
# whisper_result = ???

# TODO: 시간대별로 화자 라벨 붙이기
# for segment, _, speaker in diarization.itertracks(yield_label=True):
#     # whisper_result에서 segment.start ~ segment.end 구간의 텍스트 추출
#     # "{speaker}: {text}" 형태로 출력
```

**실습 과제**:
1. pyannote-audio 모델 로드
2. 샘플 회의 음성(또는 제공된 테스트 오디오)에서 화자 분리 실행
3. "Speaker 1: 00:30-01:15", "Speaker 2: 01:16-02:00" 형태로 타임스탐프 출력

**왜 이렇게 간단한가?**
- pyannote-audio는 **이미 학습된 모델** → 사용만 하면 됨
- 학생은 "**모델 사용법**" + "**결과 해석**"에 집중
- 내부 동작은 W9에서 (LoRA fine-tuning할 때) 깊게 학습

**🔧 Ethan's Note**:
> Whisper STT는 "무엇을 말했는가"만 알려줌.
> 실제 서비스(고객 센터, 회의록)에서는 "**누가** 언제 말했는가"가 중요.
> pyannote-audio와 Whisper 조합 = STT 완성형.
> 단, 2개 모델을 파이프라인으로 연결할 때 **지연시간(latency) 관리**가 핵심.

**[16:40-17:30] 정리 + Git 커밋**

- **오늘 배운 3줄 요약** (학생이 직접 작성):
  ```
  1. 감정 분석: 음성 신호 → MFCC → 신경망 분류
  2. 화자 분리: 사전학습 pyannote 모델로 "누가" 구별
  3. 멀티태스크: 감정 + 화자 + STT 조합 = 강력한 음성 이해
  ```

- **✅ Git 체크포인트**:
  ```bash
  git add day36_emotion_classification.py day36_speaker_diarization.py
  git commit -m "W8D1: STT 고급 - 감정분석 + 화자분리 실습"
  ```

- **Q&A 시간**: 어려운 부분 질문

- **🏠 과제**:
  - 내일을 위해 "내가 만들고 싶은 음성 모델은 뭔가?" 생각해오기
  - "그 모델로 풀 수 있는 문제는 뭔가?" 1문단 작성

---

## D2 (Day 37): 음성 모델 평가 + PJ1 오리엔테이션

> **오늘의 핵심 질문**: "내 모델이 정말 좋은지 어떻게 판단할까? PJ1은 뭔가?"

### 오전 (09:30-12:30) — 모델 평가 이론 + PJ1 소개

**[09:30-10:30] 음성 모델 평가 지표 종합**

- **Q. 왜 여러 지표가 필요한가?**
  - 정확도만으로는 부족: "모델 A가 92%, B가 91%인데, A가 더 좋다?"
  - 음성 모델은 용도에 따라 중요한 지표가 다름:
    - STT: **WER (Word Error Rate)** 가장 중요
    - TTS: **MOS (Mean Opinion Score)** 사람이 직접 평가
    - 실시간 시스템: **추론 시간(latency)** 중요

- **평가 지표 4가지**:

| 지표 | 뜻 | 범위 | 해석 | 용도 |
|------|-----|------|------|------|
| **WER** | Word Error Rate | 0-100% | 낮을수록 좋음 | STT 정확도 |
| **CER** | Character Error Rate | 0-100% | 낮을수록 좋음 | STT(한국어는 CER이 중요) |
| **MOS** | Mean Opinion Score | 1-5 | 높을수록 좋음 | TTS 음질 |
| **PESQ** | Perceptual Evaluation of Speech Quality | -0.5-4.5 | 높을수록 좋음 | 자동 음성 품질 평가 |

- **WER 계산 (가장 중요)**:
  ```
  정답: "안녕하세요, 저는 김생근입니다"
  인식: "안녕하세요, 저는 김승근입니다"  (오류: 1개 글자)

  WER = (대체 1 + 삭제 0 + 삽입 0) / (정답 총 글자 수 + 공백) × 100%
      = 1 / 12 × 100% = 8.3%
  ```

- **Phase 2 연결**: 분류 정확도 vs STT의 WER
  - Phase 2: "모델이 90% 정확도" = 좋음
  - STT: "WER 10%" = 좋음 (= 정확도 90%)
  - **같은 개념, 다른 표현**

- **실습**: Whisper 모델 평가
  ```python
  import whisper
  from jiwer import wer

  model = whisper.load_model("base")

  # 테스트 오디오
  audio_file = "test_audio.wav"
  result = model.transcribe(audio_file, language="ko")
  predicted = result["text"]

  # 정답 (수동 작성)
  ground_truth = "안녕하세요"

  # WER 계산
  error_rate = wer(ground_truth, predicted)
  print(f"WER: {error_rate:.2%}")  # 0.25 = 25%
  ```

**[10:40-11:30] PJ1 프로젝트 소개 + 기획 아이디어**

- **Q. PJ1은 뭔가?**
  - 지금까지 배운 TTS/STT = 도구/기술
  - **PJ1**: "이 도구로 실제로 **뭔가 만들기**"
  - 목표: 3주 후 "내가 만든 음성 모델" 발표

- **PJ1 범위**:
  ```
  ✅ 자유도가 높음
  - TTS 모델 커스터마이징 (한국어 가성 모델, 방언, 특수 목소리)
  - STT 도메인 모델 (의료용 STT, 법정용 STT)
  - 하이브리드 (TTS + STT 조합)

  ✅ 최소 요구사항
  - 기획서 (왜 이 모델을 만드나? 누가 쓸 건가?)
  - 구현 코드 (모델 학습 + 평가)
  - 문서 (README + 설계 문서)
  - Git 관리 + 최종 발표
  ```

- **PJ1 평가 기준**:
  | 항목 | 배점 | 체크 |
  |------|------|------|
  | 기획서 (문제정의) | 20점 | "누가 이 모델을 왜 필요로 하나?" 명확한가 |
  | 기술 구현 | 40점 | 모델이 정말 작동하고, 평가 지표는 합리적인가 |
  | 문서화 | 20점 | README, 설계, 실험 로그가 명확한가 |
  | 프레젠테이션 | 20점 | 발표/데모가 설득력 있는가 |

- **PERSO AI SDK 소개**:
  - "PERSO AI"란? = 이스트소프트 AI 음성 서비스 플랫폼
  - PJ1의 최종 목표: "내 모델을 PERSO AI SDK로 탑재"
  - SDK = 내 모델을 제품에 쉽게 끼워넣을 수 있는 인터페이스
  - W9에서 구체적으로 다룰 예정

- **🔧 Ethan's Note**:
  > PJ1에서 가장 중요한 건 "기획".
  > 코드 잘하는 것보다 **"내가 뭘 만드는지 명확히 설명할 수 있는가"가** 채용 면접에서 훨씬 중요.
  > 5000줄짜리 복잡한 코드보다 500줄이지만 **명확한 목표와 문서**가 포트폴리오로 훨씬 강함.

**[11:30-12:30] PJ1 기획 워크숍**

- **기획서 템플릿** (학생이 채울 부분):
  ```markdown
  # PJ1 기획서 (Day 37)

  ## 1. 프로젝트 이름
  예: "방언 보존 TTS" / "의료 상담용 STT" / "AI 더빙 자동화"

  ## 2. 문제 정의
  Q. 누가 어떤 문제를 갖고 있나?
  예: 한국 지역 방언 콘텐츠 제작자들은 자신의 목소리로
      자동 음성 변환할 도구가 없음.

  ## 3. 해결책 (내 모델)
  - 기술: "한국어 방언 특화 TTS 모델"
  - 방식: VITS 베이스라인 + 방언 데이터로 fine-tuning
  - 예상 결과: "사투리로 자연스럽게 말하는 음성 생성"

  ## 4. 성공 기준
  - MOS 점수 4.0 이상 (사람이 평가)
  - 생성 속도 < 2초 (실시간성)
  - 데이터셋: 300개 이상 문장

  ## 5. 기술 스택
  - 베이스라인: VITS
  - 프레임워크: PyTorch
  - 데이터: 자체 수집 또는 공개 데이터셋
  - 배포: PERSO AI SDK

  ## 6. 일정 (W8-W10)
  - W8D3-D4: 데이터 수집 + 전처리
  - W9D1-D2: 모델 학습 + 튜닝
  - W9D3-D5: 평가 + SDK 탑재
  - W10D1: 최종 검증 + 발표 준비
  ```

**[11:45-12:30] 학생 기획 공유 (5분 × 3명)**

- 각 학생이 기획한 PJ1 프로젝트 1분 피치
- 강사 피드백: "이 기획이 3주간 실행 가능한가?"

### 오후 (13:30-17:30) — 기획서 작성 + Git 초기화

**[13:30-15:00] 개별 기획서 작성**

- 학생은 위 템플릿을 자신의 프로젝트에 맞춰 작성
- 강사는 매 15분마다 개인 피드백

**기획서 체크리스트**:
- [ ] 문제 정의가 **구체적**인가? (막연하지 않음)
- [ ] 기술 선택에 **이유**가 있는가?
- [ ] 성공 기준이 **측정 가능**한가?
- [ ] 일정이 **현실적**인가? (3주 / W8-W10)

**[15:10-16:30] PJ1 Git 리포 초기화 + 첫 커밋**

**Git 초기화** (학생이 직접 함):

```bash
# GitHub에 새 리포 생성 (pj1-speech-model)

cd ~/pj1-speech-model
git init
git config user.email "student@example.com"
git config user.name "Student Name"

# 초기 파일 구조 생성
mkdir -p docs data/raw data/processed src notebooks

# README.md 작성
cat > README.md << 'EOF'
# PJ1: [프로젝트 이름]

## 개요
[프로젝트 한 문장 설명]

## 기획서
- [docs/planning.md](docs/planning.md) 참고

## 구조
- `docs/`: 설계 문서
- `src/`: 코드
- `data/`: 데이터 (처리전/후)
- `notebooks/`: 실험용 주피터 노트북

## 진행 상황
- W8D2: 기획서 완성
- W9D1-D5: 개발
- W10D1: 최종 검증

EOF

# .gitignore 작성 (중요!)
cat > .gitignore << 'EOF'
# 데이터 (크기 때문에 제외)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# 모델 체크포인트
*.pt
*.pth
*.bin

# 환경 설정
.env
.venv/

# IDE
.vscode/
.idea/

# 시스템
.DS_Store
__pycache__/
*.pyc
EOF

# 기획서를 docs에 옮기기
mv planning.md docs/planning.md

# 첫 커밋
git add .
git commit -m "W8D2: PJ1 init - 기획서 + 리포 구조"

# 원격 리포에 연결
git remote add origin https://github.com/student-account/pj1-speech-model.git
git branch -M main
git push -u origin main
```

**왜 .gitignore가 중요한가?**
- `data/` 디렉토리: 음성 파일은 보통 수GB → Git에 못 올림
- `*.pt`, `*.bin`: 모델 파일도 수백MB → Git에 못 올림
- `.env`: API 키 같은 민감한 정보는 **절대** Git에 올리면 안 됨
- **→ .gitignore에 명시해야 실수로 올리는 일 방지**

**📦 프로젝트 연결**:
> "기획서에서 정한 기술 스택이 W9에서 실제 구현할 내용"
> 예: "VITS 베이스라인"으로 했으면 → W9D1에서 VITS 학습 코드 작성

- **✅ Git 체크포인트**:
  ```bash
  git add docs/planning.md .gitignore README.md
  git commit -m "W8D2: PJ1 기획서 + Git 초기화"
  git push origin main
  ```

**[16:40-17:30] 정리 + Q&A**

- **체크리스트**:
  - [ ] 기획서 작성 완료
  - [ ] Git 리포 초기화
  - [ ] 첫 커밋 완료

- **내일을 위해**:
  - 기획서에 따라 "수집할 데이터" 리스트 만들어오기
  - 데이터 소스 찾기 (공개 데이터셋 또는 수집 계획)

---

## D3 (Day 38): PJ1 아키텍처 설계 + 데이터 전처리

> **오늘의 핵심 질문**: "어떤 구조로 모델을 만들고, 데이터를 어떻게 준비할까?"

### 오전 (09:30-12:30) — 아키텍처 설계

**[09:30-10:30] 모델 아키텍처 선택 로직**

- **Q. VITS vs Tacotron2, 어떤 걸 고를까?**
  - W7에서 배운 2가지 TTS 모델 비교
  - 선택 기준:
    | 기준 | VITS | Tacotron2 |
    |------|------|---------|
    | 속도 | 빠름 (⚡⚡) | 느림 (⚡) |
    | 음질 | 자연스러움 (⭐⭐⭐⭐⭐) | 좋음 (⭐⭐⭐⭐) |
    | Fine-tuning 용이 | 쉬움 | 어려움 |
    | 데이터 필요 | 중간 (300~500h) | 많음 (1000h) |
    | 학습 시간 | 짧음 | 김 |

  - **PJ1은 보통 VITS 선택** (3주 일정에 적합)

- **STT 아키텍처**:
  - Whisper 사용 가능? (오픈소스, 이미 학습됨)
  - 또는 Wav2Vec2 Fine-tuning? (도메인 특화)
  - 선택: "기획서에 따라 다름"

- **Phase 2 연결**: 신경망 아키텍처
  - Phase 2: "CNN/RNN/Transformer 각각 언제 쓰나?"
  - 이제: "TTS/STT 특성에 맞는 모델 선택"

- **🔧 Ethan's Note**:
  > 아키텍처 선택은 "성능 vs 비용 vs 시간"의 트레이드오프.
  > 완벽한 모델보다 **"실제 3주 안에 끝낼 수 있는 현실적인 선택"이 중요**.
  > PM으로서 가장 자주 하는 질문: "이거 3주 안에 가능해?" → Yes/No 명확히.

**[10:40-11:30] 데이터 파이프라인 설계**

- **음성 데이터 파이프라인**:
  ```
  원본 데이터 (wav, mp3)
    ↓ [로드]
  음성 신호 (waveform)
    ↓ [정규화]
  0~1 범위로 맞춤
    ↓ [노이즈 제거 (선택)]
  배경음 감소
    ↓ [분할]
  청크로 나누기 (보통 2-3초)
    ↓ [특징 추출]
  Mel-Spectrogram / MFCC
    ↓ [저장]
  학습 데이터 준비 완료
  ```

- **코드 스켈레톤** (librosa 사용):
  ```python
  import librosa
  import numpy as np

  def preprocess_audio(file_path, sr=22050):
      """음성 전처리"""
      # 1. 로드
      y, sr = librosa.load(file_path, sr=sr)

      # 2. 정규화 (진폭을 -1.0 ~ 1.0 범위로)
      y = y / np.max(np.abs(y)) + 1e-8

      # 3. 특징 추출
      mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
      mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

      return mel_spec_db
  ```

**[11:30-12:30] 설계 문서 작성**

- **아키텍처 다이어그램** (Markdown + ASCII Art):
  ```
  입력 (한국어 텍스트)
    ↓
  [Tokenizer] → 토큰 시퀀스
    ↓
  [Encoder] → 음성 특징 (VITS 경우)
    ↓
  [Decoder] → 스펙트로그램
    ↓
  [Vocoder (WaveGlow)] → 음성 신호 (waveform)
    ↓
  출력 (음성 파일)
  ```

- **설계 문서 템플릿**:
  ```markdown
  # PJ1 아키텍처 설계

  ## 1. 모델 선택
  - TTS: VITS + WaveGlow Vocoder
  - STT: Whisper base
  - 이유: 성능과 학습 시간 트레이드오프

  ## 2. 데이터 파이프라인
  [위의 파이프라인 다이어그램]

  ## 3. 학습 설정
  - GPU: NVIDIA T4 (Google Colab)
  - Batch size: 16
  - Learning rate: 1e-3 (Adam optimizer)
  - Epochs: 10 (또는 수렴할 때까지)

  ## 4. 평가 지표
  - TTS: MOS (Mean Opinion Score)
  - STT: WER (Word Error Rate)

  ## 5. 일정
  - D3: 데이터 수집 + 전처리 파이프라인
  - D4-D5: 모델 학습
  ```

### 오후 (13:30-17:30) — 데이터 수집 + 전처리

**[13:30-15:00] 데이터 수집**

**데이터 소스 3가지**:

1. **공개 데이터셋** (가장 쉬움)
   - LibriSpeech (영어 STT용)
   - KSS (Korean Single Speaker) — 한국어 TTS
   - RAVDESS (감정 음성)
   - → 이미 정제되어 있음

2. **자체 수집** (충정도 높음)
   - 스마트폰 녹음
   - 유튜브 클립 (자신의 음성)
   - 친구들 녹음 (동의 필수)

3. **합성 데이터** (빠름)
   - Google Text-to-Speech로 생성
   - 또는 기존 모델(Tacotron2) 활용

**실습 가이드**:

```python
# KSS 데이터셋 다운로드 (한국어 여성 목소리)
# https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset

import os
import librosa
import pandas as pd

# KSS 데이터셋 로드 (이미 다운로드했다고 가정)
kss_path = "KSS/1/*/*.wav"
metadata_path = "KSS/1/metadata.csv"

# 메타데이터 읽기 (파일명, 텍스트, 감정, 성별)
df = pd.read_csv(metadata_path, sep="|", header=None)
df.columns = ["filename", "text", "emotion", "gender"]

# 데이터 탐색
print(f"전체 샘플: {len(df)}")
print(f"텍스트 길이 분포:")
print(df["text"].str.len().describe())

# 이상치 제거
df = df[df["text"].str.len() < 100]  # 100자 이상 제외
df = df[df["text"].str.len() > 5]    # 5자 미만 제외

print(f"필터링 후: {len(df)}")

# 훈련/테스트 분할
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"훈련 데이터: {len(train_df)}, 테스트 데이터: {len(test_df)}")
```

**[15:10-16:30] 데이터 전처리 파이프라인**

**목표**: "원본 음성" → "모델 학습용 데이터"

**코드 스켈레톤**:

```python
import librosa
import numpy as np
import os
from tqdm import tqdm

def preprocess_dataset(input_dir, output_dir, sr=22050):
    """
    음성 데이터셋 일괄 전처리

    입력: input_dir — 원본 wav 파일들
    출력: output_dir — 전처리된 mel-spectrogram (.npy)
    """

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(input_dir, filename)

        # TODO: librosa.load(file_path, sr=sr)로 음성 로드
        # TODO: 정규화 (y = y / np.max(np.abs(y)) + 1e-8)
        # TODO: Mel-spectrogram 추출 (librosa.feature.melspectrogram)
        # TODO: log scale 변환 (librosa.power_to_db)
        # TODO: .npy 파일로 저장
        # HINT: output_path = os.path.join(output_dir, filename.replace(".wav", ".npy"))

        pass

# 사용 예
preprocess_dataset("data/raw", "data/processed")
```

**데이터 검사** (눈으로 확인):

```python
import matplotlib.pyplot as plt

# 샘플 데이터 시각화
sample_mel = np.load("data/processed/sample_audio.npy")

plt.figure(figsize=(12, 4))
plt.imshow(sample_mel, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Mel-Spectrogram Example")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

# 데이터 통계
print(f"Shape: {sample_mel.shape}")  # (128, time_steps)
print(f"Min: {sample_mel.min():.2f}, Max: {sample_mel.max():.2f}")
```

**📦 프로젝트 연결**:
> 오늘 만든 전처리 파이프라인이 W9에서 실제 학습에 쓰임.
> "깨끗한 데이터 = 좋은 모델" — 데이터 전처리의 중요성!

**[16:40-17:30] 정리 + 커밋**

**체크리스트**:
- [ ] 데이터 소스 확정
- [ ] 데이터셋 다운로드 + 로드 확인
- [ ] 전처리 파이프라인 동작 확인
- [ ] 샘플 시각화 + 통계 검토

**✅ Git 체크포인트**:
```bash
git add docs/architecture.md src/preprocess.py
git commit -m "W8D3: PJ1 설계 + 데이터 전처리 파이프라인"
git push origin main
```

---

## D4 (Day 39): PJ1 모델 구현 시작 — 베이스라인 학습

> **오늘의 핵심 질문**: "실제로 모델이 학습되나? 첫 결과가 어떻게 나올까?"

### 오전 (09:30-12:30) — 모델 로드 + 학습 설정

**[09:30-10:30] 사전학습 모델 로드**

- **왜 처음부터 학습하지 않나?**
  - VITS 모델은 보통 **영어 또는 중국어로 사전학습됨**
  - 0부터 학습하려면 수천 시간의 고음질 데이터 + GPU 필요
  - **더 효율적**: 사전학습 모델 → Fine-tuning (소수 데이터로 조정)

- **Phase 2 연결**: Transfer Learning
  - Phase 2: BERT 사전학습 모델 → 자신의 텍스트 분류 데이터로 Fine-tuning
  - 이제: VITS 사전학습 모델 → 자신의 음성 데이터로 Fine-tuning

- **HuggingFace에서 모델 로드**:
  ```python
  from transformers import AutoModel

  # 사전학습된 한국어 VITS 모델 다운로드
  model_name = "kresnik/vits-transformer-tts"  # 한국어 가능
  model = AutoModel.from_pretrained(model_name)

  print(f"모델 파라미터: {model.num_parameters():,}")  # 총 가중치 수
  # 출력 예: 모델 파라미터: 14,923,456
  ```

**[10:40-11:30] GPU 환경 설정**

- **로컬 vs Google Colab**:
  - 로컬: GPU 없으면 학습이 매우 느림 (또는 불가능)
  - **Colab (권장)**: 무료 NVIDIA T4 제공, VITS 학습에 충분

- **Colab 셀 구성**:
  ```python
  # Colab 첫 셀
  !pip install transformers torch torchaudio librosa

  # GPU 확인
  import torch
  print(torch.cuda.is_available())  # True이면 OK
  print(torch.cuda.get_device_name(0))  # GPU 이름 출력
  ```

**[11:30-12:30] 학습 파라미터 설정**

- **하이퍼파라미터 정의**:
  ```python
  config = {
      "learning_rate": 1e-3,      # Adam optimizer 학습률
      "batch_size": 16,             # GPU 메모리에 따라 조정
      "num_epochs": 10,             # 데이터셋 크기에 따라 조정
      "save_interval": 500,         # 500 스텝마다 체크포인트 저장
      "device": "cuda" if torch.cuda.is_available() else "cpu"
  }
  ```

- **왜 이 값들인가?**
  - `learning_rate 1e-3`: Fine-tuning이므로 낮게 (0.1 수준이면 기존 학습 망침)
  - `batch_size 16`: T4 GPU에 맞는 안전한 크기
  - `num_epochs 10`: 보통 3-5 에포크면 수렴, 여유 있게 10으로

### 오후 (13:30-17:30) — 모델 학습

**[13:30-15:00] 학습 루프 구현**

**목표**: VITS 모델을 자신의 데이터로 Fine-tuning

**코드 스켈레톤** (간단한 버전):

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# ===================== 데이터셋 클래스 =====================
class AudioDataset(Dataset):
    """음성 데이터셋"""
    def __init__(self, audio_dir, text_list):
        self.audio_dir = audio_dir
        self.texts = text_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # TODO: audio_dir에서 idx번째 음성 로드 (.npy 파일)
        # TODO: 해당 텍스트 반환
        # HINT: 음성은 mel-spectrogram (128, T) 형태
        # HINT2: 텍스트는 이미 전처리되어야 함 (토큰 시퀀스)
        pass

# ===================== 학습 함수 =====================
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """1 에포크 학습"""
    model.train()  # 모델을 훈련 모드로

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, (audio, text) in enumerate(progress_bar):
        audio = audio.to(device)  # GPU로 이동
        text = text.to(device)

        # TODO: Forward pass
        # output = model(text, audio)

        # TODO: Loss 계산
        # loss = output.loss (또는 custom loss)

        # TODO: Backward
        # loss.backward()

        # TODO: Optimizer step
        # optimizer.step()
        # optimizer.zero_grad()

        total_loss += loss.item()

        # 진행률 바 업데이트
        progress_bar.set_postfix({"loss": loss.item()})

        # 중간 체크포인트 저장
        if (batch_idx + 1) % 500 == 0:
            checkpoint_path = f"checkpoints/model_epoch{epoch}_step{batch_idx}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ 체크포인트 저장: {checkpoint_path}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ===================== 메인 학습 루프 =====================
def main():
    # 설정
    config = {
        "learning_rate": 1e-3,
        "batch_size": 16,
        "num_epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    device = torch.device(config["device"])

    # 모델 로드
    from transformers import AutoModel
    model = AutoModel.from_pretrained("kresnik/vits-transformer-tts")
    model = model.to(device)

    # 데이터 로드 (예시)
    # 실제로는 자신의 데이터셋 경로로 변경
    dataset = AudioDataset("data/processed", text_list=[...])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # 학습 루프
    for epoch in range(config["num_epochs"]):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch+1} — 평균 Loss: {avg_loss:.4f}")

        # 매 에포크마다 모델 저장
        checkpoint_path = f"checkpoints/model_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    main()
```

**학습 과정 모니터링**:

```python
# Colab에서 실시간 loss 그래프 그리기
import matplotlib.pyplot as plt

losses = []

for epoch in range(num_epochs):
    avg_loss = train_one_epoch(...)
    losses.append(avg_loss)

    # 매 에포크마다 그래프 업데이트
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()
```

**🔧 Ethan's Note**:
> 첫 학습은 항상 "에러, 에러, 에러"의 연속.
> 중요한 건 **"작은 것부터 시작"** (tiny 모델, 작은 데이터셋).
> 먼저 "10개 샘플로 1에포크 돌려보기" → "100개로 1에포크" → "전체 데이터"
> 이렇게 점진적으로 늘려가면서 버그를 잡아야 함.

**[15:10-16:30] 학습 결과 분석**

- **Loss 그래프 읽는 법**:
  - Loss가 계속 감소? → ✅ 학습이 잘 됨
  - Loss가 높게 유지? → ⚠️ 학습률 문제 또는 데이터 문제
  - Loss가 증가? → ❌ 버그 또는 학습률이 너무 높음

- **체크포인트 선택**:
  ```
  Epoch 1: Loss 5.23 ← 너무 높음
  Epoch 2: Loss 3.45
  Epoch 3: Loss 2.89 ← 여기부터 수렴 시작
  Epoch 4: Loss 2.45
  Epoch 5: Loss 2.41 ← 거의 변화 없음 (수렴)

  → Epoch 5 체크포인트 사용
  ```

**📦 프로젝트 연결**:
> "이 체크포인트가 W9D2에서 STT 모델과 합쳐져서 완전한 음성 시스템이 됨"

**[16:40-17:30] 정리 + 커밋**

**결과 저장**:
```python
# 학습 로그 저장
with open("experiment_log.txt", "w") as f:
    f.write("W8D4 학습 결과\n")
    f.write(f"최종 Loss: {avg_loss:.4f}\n")
    f.write(f"에포크: {num_epochs}\n")
    f.write(f"배치 크기: {batch_size}\n")

# 최적 체크포인트 이름 변경
import shutil
shutil.copy("checkpoints/model_epoch5.pt", "checkpoints/model_best.pt")
```

**✅ Git 체크포인트**:
```bash
git add src/train.py checkpoints/model_best.pt experiment_log.txt
git commit -m "W8D4: PJ1 모델 v0.1 - 베이스라인 학습 완료"
git push origin main
```

---

## D5 (Day 40): W8 스프린트 리뷰 + W9 계획

> **오늘의 핵심 질문**: "지난 주 뭘 배웠고, 다음 주는 뭘 할까?"

### 오전 (09:30-12:30) — 스프린트 리뷰

**[09:30-10:30] 개별 모델 평가**

**각 학생의 PJ1 현황**:
- 기획서: ✅ 완성
- 데이터: ✅ 수집 + 전처리
- 모델: ✅ 학습 시작
- 평가: 🔄 진행 중

**학생 A**: "방언 보존 TTS"
- 데이터: KSS 300개 + 자체 수집 100개 (보령 방언)
- Loss: 2.45 (수렴됨)
- 다음 단계: MOS 평가 (사람이 음질 판정)

**학생 B**: "의료 상담 STT"
- 데이터: 의료 용어 사전 + Whisper 파인튜닝
- WER: 15% (목표 10%)
- 다음 단계: 후처리(의료 용어 자동 교정) 추가

**[10:40-11:30] 실패 사례 공유 + 해결**

- **"데이터가 부족해서 Loss가 안 내려가요"**
  - 해결: 데이터 증강(Data Augmentation) — 음성에 노이즈/피치 변형 추가
  - Phase 2의 이미지 data augmentation과 같은 개념

- **"메모리 부족 에러가 자꾸 나요"**
  - 원인: batch_size가 너무 크거나 모델이 너무 큼
  - 해결: batch_size 줄이기 (16 → 8)

- **"모델이 학습하는 것 같지 않아요 (Loss 변화 없음)"**
  - 원인: 데이터 로더 버그 또는 학습률 문제
  - 디버깅: 1 배치만 돌려서 loss가 나오는지 확인

- **🔧 Ethan's Note**:
  > 개발 90%는 "버그 잡기".
  > 모델이 안 맞으면 먼저 **작은 데이터로 테스트** (10개 샘플).
  > "내 코드가 돌아가나?" 확인 후 → "모델이 배우나?" 확인.
  > 순서를 뒤집으면 시간 낭비함.

**[11:30-12:30] W8 회고 (Retrospective)**

**이번 주 배운 것**:
- STT 고급 (감정분석, 화자분리)
- 모델 평가 지표 (WER, MOS)
- **PJ1 기획→설계→구현 경험** (가장 중요)
- Git 워크플로우

**체크리스트**:
- [ ] STT 고급 기능 실습 완료
- [ ] PJ1 기획서 작성
- [ ] 데이터셋 구축
- [ ] 첫 모델 학습 성공
- [ ] 최소 5회 Git 커밋

**포트폴리오에 기록할 것**:
> "W8에서는 음성 모델의 기획부터 구현까지 경험했다.
> 특히 Whisper + pyannote-audio 조합으로 화자 분리 + STT 통합 구현,
> VITS로 자신의 음성 데이터로 TTS 모델 커스터마이징 시작했다."

### 오후 (13:30-17:30) — W9 개발 계획 + 협력 준비

**[13:30-14:30] W9 일정 설명**

| Day | 주제 | 산출물 |
|-----|------|--------|
| W9D1 | TTS 모델 v0.2 (성능 개선) | 개선된 모델 체크포인트 |
| W9D2 | STT 모델 v0.2 (도메인 특화) | Fine-tuned STT 모델 |
| W9D3 | **PERSO AI SDK 탑재** | SDK 연동 완료 |
| W9D4 | 통합 테스트 + 성능 최적화 | 테스트 결과 리포트 |
| W9D5 | 문서화 + 코드 정리 | 최종 프로젝트 리포 |

**[14:40-15:30] 피어 코드 리뷰 페어링**

- 학생 2명씩 페어를 이뤄 GitHub에서 코드 리뷰
- 리뷰 포인트:
  - [ ] 코드가 실행되는가?
  - [ ] 주석이 명확한가?
  - [ ] 개선 아이디어가 있는가?

**[15:40-16:30] W8 최종 Git Push**

```bash
# W8 전체 작업을 하나의 브랜치로 정리
git checkout -b w8-complete
git add -A
git commit -m "W8 완료: STT 고급 + PJ1 기획→구현 v0.1"

# main에 병합
git checkout main
git merge w8-complete

# GitHub에 push
git push origin main
```

**[16:30-17:30] 다음주 준비**

- **W9D1을 위한 준비**:
  - [ ] TTS 성능 개선 논문 읽기 (선택)
  - [ ] 자신의 모델 loss 그래프 분석
  - [ ] "왜 loss가 이 정도일까?" 가설 세우기

- **질문 리스트 준비**:
  - "W9에서 PERSO AI SDK는 정확히 뭔가?"
  - "모델을 SDK에 어떻게 끼워 넣나?"
  - "배포 후에도 모델을 수정할 수 있나?"

---

## 주간 정리

### ✅ W8 핵심 달성 목표 체크리스트

- [x] STT 고급 기능 이해 (감정분석, 화자분리)
- [x] 음성 모델 평가 지표 학습 (WER, CER, MOS)
- [x] PJ1 기획서 작성
- [x] PJ1 데이터 수집 + 전처리 완료
- [x] PJ1 모델 v0.1 학습 시작
- [x] Git 워크플로우 실전 (최소 5회 커밋)

### W8 → W9 연결

```
W8 달성                          W9 활용
├─ STT 고급 ─────────────────→ STT 모델 v0.2 (도메인 특화)
├─ PJ1 기획 + 설계 ──────────→ 설계에 따른 구현 + 튜닝
├─ 첫 모델 학습 ─────────────→ 하이퍼파라미터 최적화
└─ Git 기초 ─────────────────→ 브랜치 관리 + 코드 리뷰 심화
```

### 🎯 채용시장 체크

> W8을 마친 당신은 이제:
> - **"실제 프로젝트 기획부터 구현까지 경험한" 엔지니어**
> - **모델 평가 지표를 이해하는 데이터 사이언티스트**
> - **Git으로 협업하는 개발자**
>
> 채용 면접에서:
> - "PJ1을 설명해 주세요" → 기획 의도, 기술 선택, 데이터 전략 설명 가능
> - "모델이 좋은지 어떻게 판단했나요?" → WER/MOS/Loss 등 정량 지표로 설명 가능
> - "코드 버전 관리는 어떻게?" → Git 워크플로우(커밋, 브랜치) 설명 가능

---

## 준비물 체크리스트 (강사용)

| 항목 | 상태 | 비고 |
|------|------|------|
| Google Colab Pro 환경 | ☑️ | GPU T4 필수 |
| Whisper 모델 (base) | ☑️ | HuggingFace 사전 다운로드 |
| pyannote-audio 설정 | ☑️ | HuggingFace 토큰 발급 필수 |
| VITS 사전학습 모델 | ☑️ | kresnik/vits-transformer-tts 확인 |
| 샘플 음성 데이터 | ☑️ | RAVDESS, KSS 다운로드 링크 제공 |
| PJ1 템플릿 리포 | ☑️ | GitHub 템플릿 사전 준비 |
| PERSO AI SDK 접근 | ☐ | W9 전 이스트소프트 확인 필요 |

---

## 추가 자료 + 참고

### 추천 학습 자료
- Whisper 공식 문서: https://github.com/openai/whisper
- pyannote-audio 튜토리얼: https://huggingface.co/pyannote/speaker-diarization-3.0
- VITS 논문: https://arxiv.org/abs/2106.06103

### 학생이 자주 하는 질문 (Q&A)

**Q. "모델이 학습 안 돼요, Loss가 계속 NaN이에요"**
A. 보통 3가지:
1. 학습률이 너무 높음 (1e-3 대신 1e-5로 낮춰보세요)
2. 데이터 로더에서 inf/nan 값 (데이터 정규화 확인)
3. 배치 크기가 너무 큼 (메모리 부족)

**Q. "GPU 메모리 초과 에러"**
A. 해결 순서:
1. batch_size 절반으로 (16 → 8)
2. num_workers=0으로 설정
3. 필요시 모델 크기 줄이기 (vits-large → vits-base)

**Q. "W9에서 PERSO AI SDK는 정확히 뭔가요?"**
A. 다음주 W9D3에서 구체적으로 다룰 예정. 미리: "제품에 모델을 끼워넣기 위한 인터페이스"

---

© 2026 김생근 (Sanguine Kim) | CC BY-NC 4.0
최종 수정: 2026-03-24 (A1/A2 개선 적용)
