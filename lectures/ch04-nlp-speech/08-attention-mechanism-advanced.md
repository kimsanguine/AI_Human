# Ch04-8: Attention Mechanism — Cross-Attention & 멀티모달 응용

**학습 목표**: 기본 Self-Attention에서 Cross-Attention으로 확장하여, 텍스트-이미지-음성 동시 처리 구조 이해

**실제 사례**: Meta Muse Spark, Google AI Edge Eloquent, Claude Agents

**기간**: 3-4시간 (강의 1h + 실습 2-3h)

---

## 1️⃣ Self-Attention vs Cross-Attention

### Self-Attention (복습)

```
입력: 같은 시퀀스에서 쿼리, 키, 값 모두 추출
예: "나는 강아지를 본다"
    ↓
[쿼리: "나"]      [쿼리: "강아지"] 
↓                 ↓
자기 자신을 주목  "강아지"를 주목
+ 뒤의 모든 것    + 앞의 맥락 ("나는")
```

**계산**:
```
Attention(Q, K, V) 
where Q, K, V 모두 같은 입력에서 추출
= softmax(Q·K^T / √d) · V
```

---

### Cross-Attention (새로운 개념)

```
입력: 다른 두 시퀀스에서 쿼리/키-값 추출
예: 텍스트와 이미지

텍스트 토큰: "강아지"
    ↓ (쿼리로 사용)
이미지 패치: [패치1, 패치2, ..., 패치196]
    ↓ (키-값으로 사용)
    
"강아지"라는 의미가 이미지의 어느 부분과 매치되는가?
```

**계산**:
```
CrossAttention(Q_text, K_image, V_image)
where Q는 텍스트, K와 V는 이미지
= softmax(Q_text · K_image^T / √d) · V_image
```

---

## 2️⃣ Meta Muse Spark: 네이티브 멀티모달 아키텍처

### 문제 정의: Late Fusion vs Native Integration

#### 방식 1: Late Fusion (기존 방식)

```
입력: 텍스트 + 이미지

[텍스트 토큰]           [이미지]
    ↓                    ↓
Transformer            Vision Transformer
(BERT-style)           (ViT)
    ↓                    ↓
텍스트 표현             이미지 표현
[768차원]               [768차원]
    ↓                    ↓
        [Concatenate]
            ↓
        [1536차원]
            ↓
    [Fusion Decoder]
            ↓
       최종 표현
```

**문제점**:
- 텍스트와 이미지가 **분리되어 처리**됨
- Fusion 레이어가 이들을 억지로 합침
- 깊은 상호작용 불가능

#### 방식 2: Native Integration (Muse Spark)

```
입력: [텍스트 토큰 + 이미지 패치]
       (섞여서 들어옴, 예: "나는 [PAD] 강아지 [IMG1] 를 본다 [IMG2] [IMG3]")
    
    ↓
[동일한 Transformer 레이어들]
    ↓
각 레이어에서:
- 텍스트 토큰이 다른 텍스트를 self-attend
- 텍스트 토큰이 이미지 패치를 cross-attend
- 이미지 패치가 텍스트를 cross-attend
    ↓
[최종 표현: 텍스트+이미지가 완전히 섞여 있음]
```

**장점**:
- **조기 상호작용(Early Interaction)**: 첫 레이어부터 텍스트-이미지 융합
- 모달리티 간 정보 교환 깊음

---

### 구현: Cross-Attention 계산

#### Python 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    텍스트(쿼리)와 이미지(키-값) 간의 Attention
    """
    
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 쿼리, 키, 값 선형 변환
        self.W_q = nn.Linear(d_model, d_model)  # 텍스트용
        self.W_k = nn.Linear(d_model, d_model)  # 이미지용
        self.W_v = nn.Linear(d_model, d_model)  # 이미지용
        
        self.W_out = nn.Linear(d_model, d_model)
    
    def forward(self, text_embedding, image_embedding):
        """
        Args:
            text_embedding: (batch, seq_len_text, d_model)
                           예: (2, 20, 768) - 20개 토큰, 각 768차원
            image_embedding: (batch, n_patches, d_model)
                            예: (2, 196, 768) - 196개 패치 (14x14)
        
        Returns:
            attended_output: (batch, seq_len_text, d_model)
                            텍스트와 이미지가 상호작용한 결과
        """
        batch, seq_len, d_model = text_embedding.shape
        n_image_patches = image_embedding.shape[1]
        
        # 선형 변환
        Q = self.W_q(text_embedding)  # (batch, 20, 768)
        K = self.W_k(image_embedding)  # (batch, 196, 768)
        V = self.W_v(image_embedding)  # (batch, 196, 768)
        
        # Multi-head 분할
        Q = Q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # (batch, 12, 20, 64) - 12개 헤드, 20 토큰, 각 64차원
        
        K = K.view(batch, n_image_patches, self.n_heads, self.d_head).transpose(1, 2)
        # (batch, 12, 196, 64)
        
        V = V.view(batch, n_image_patches, self.n_heads, self.d_head).transpose(1, 2)
        # (batch, 12, 196, 64)
        
        # Attention 스코어
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        # (batch, 12, 20, 196) - 각 토큰이 196개 패치를 보는 관계도
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # 마지막 차원(패치)에 대해 정규화
        # (batch, 12, 20, 196)
        
        # Attention 적용
        attended = torch.matmul(attn_weights, V)
        # (batch, 12, 20, 64)
        
        # Multi-head 재결합
        attended = attended.transpose(1, 2).contiguous()
        # (batch, 20, 12, 64)
        
        attended = attended.view(batch, seq_len, d_model)
        # (batch, 20, 768)
        
        # 출력 선형 변환
        output = self.W_out(attended)  # (batch, 20, 768)
        
        return output, attn_weights  # 가중치도 반환 (시각화용)


# 예시: 실제 사용
batch_size = 2
seq_len_text = 20  # "나는 강아지를 본다"라는 20개 토큰
n_patches = 196     # 14x14 이미지 패치
d_model = 768

text_emb = torch.randn(batch_size, seq_len_text, d_model)
image_emb = torch.randn(batch_size, n_patches, d_model)

cross_attn = CrossAttention(d_model=768, n_heads=12)
output, weights = cross_attn(text_emb, image_emb)

print(f"출력 형태: {output.shape}")  # (2, 20, 768)
print(f"가중치 형태: {weights.shape}")  # (2, 12, 20, 196)

# 가중치 해석
# 예: 첫 번째 배치, 첫 번째 헤드, 토큰 5 ("강아지")
attn_to_patches = weights[0, 0, 5, :]  # (196,)
print(f"'강아지' 토큰이 196개 패치에 할당한 주목도:")
print(f"  최대값: {attn_to_patches.max():.3f}")
print(f"  최대값 인덱스: {attn_to_patches.argmax()}")  # 아마 개가 있는 영역?
```

---

### 시각화: Attention 가중치

```
텍스트: "나는 강아지를 본다"

이미지:      [이미지 14x14 패치]

"강아지"에 대한 Attention 히트맵:
┌─────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░│
│ ░░░░░░░░░░░░░░░░░░░░░░░│
│ ░░░░░░░░░░░░░░░░░░░░░░░│
│ ░░░░░░░░████████░░░░░░░│  ← 강아지가 있는 부분
│ ░░░░░░░░████████░░░░░░░│
│ ░░░░░░░░████████░░░░░░░│
│ ░░░░░░░░░░░░░░░░░░░░░░░│
└─────────────────────────┘

어두운 부분 = 높은 주목도 (Softmax 가중치)
밝은 부분 = 낮은 주목도
```

---

## 3️⃣ 실제 응용: Google AI Edge Eloquent

### 온디바이스 ASR의 멀티모달 구조

```
입력: 음성 신호 (16kHz 샘플링)
      ↓
[1단계: 음성 특성 추출]
스펙트로그램 → 멜 스케일 (80 밴드)
      ↓
    (T, 80)  T = 시간 프레임 수
      ↓
[2단계: 음성 인코더 (Transformer)]
각 프레임이 다른 프레임을 self-attend
      ↓
컨텍스트 표현: (T, 768)
      ↓
[3단계: 디코더 (RNN 또는 Attention-based)]
어느 시간 영역에 집중할지 결정 (Attention)
      ↓
토큰 시퀀스 생성 (subword units)
      ↓
[4단계: 텍스트 후처리]
필터 단어 제거, 스타일 변환
      ↓
최종 텍스트: "오늘 날씨 좋네요"
```

### 온디바이스 vs 클라우드

```
온디바이스 모드 (완전 오프라인):
입력 음성 → [경량 Transformer] → [RNN 디코더] → 텍스트
(모든 처리가 휴대폰에서)
장점: 프라이버시, 지연 시간 짧음
단점: 정확도 ~95%

클라우드 모드 (하이브리드):
입력 음성 → [경량 Transformer] → [Gemini 대형 모델] → 텍스트
(음성 특성 추출은 온디바이스, 최종 의미 이해는 클라우드)
장점: 정확도 ~99%, 실시간 처리
단점: 인터넷 필요, 약간의 지연
```

---

## 4️⃣ Claude Agents: 임베딩으로 도구 선택

### 에이전트가 의도를 이해하는 과정

```
사용자: "파일을 읽고 요약해줘"
    ↓
[Step 1: 임베딩]
"파일을 읽고 요약해줘" → 벡터 (768차원)
[0.2, -0.5, 0.1, ..., 0.3]
    ↓
[Step 2: 도구 임베딩 (사전 계산)]
"read_file" 설명 → 벡터
"summarize" 설명 → 벡터
"write_file" 설명 → 벡터
    ↓
[Step 3: 유사도 계산]
사용자 벡터 · 도구 벡터 (내적)
    ↓
결과:
read_file: 0.841
summarize: 0.892  ← 가장 높음
write_file: 0.234
    ↓
[Step 4: 도구 선택 및 실행]
summarize() 호출
```

#### 구현

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 임베딩 모델
model = SentenceTransformer('all-MiniLM-L6-v2')

# 사용자 입력
user_input = "파일을 읽고 요약해줘"

# 도구 정의
tools = {
    "read_file": "주어진 경로의 파일 내용을 읽는 함수",
    "summarize": "긴 텍스트를 요약하는 함수",
    "write_file": "텍스트를 파일로 저장하는 함수",
    "delete_file": "파일을 삭제하는 함수",
}

# 임베딩
user_emb = model.encode(user_input)
tool_embs = {name: model.encode(desc) for name, desc in tools.items()}

# 유사도
similarities = {}
for tool_name, tool_emb in tool_embs.items():
    # 코사인 유사도
    sim = np.dot(user_emb, tool_emb) / (
        np.linalg.norm(user_emb) * np.linalg.norm(tool_emb)
    )
    similarities[tool_name] = sim

# 출력
print("도구 선택 순위:")
for tool, score in sorted(similarities.items(), key=lambda x: -x[1]):
    print(f"  {tool}: {score:.3f}")

# 선택
best_tool = max(similarities, key=similarities.get)
print(f"\n선택 도구: {best_tool}")
```

---

## 5️⃣ 실습: 멀티모달 Attention 직접 구현

### 실습 1: 간단한 Cross-Attention

```python
import torch
import torch.nn.functional as F

# 간단한 예시
batch = 1
seq_text = 5
n_patches = 9
d = 64

# 입력
text = torch.randn(batch, seq_text, d)      # (1, 5, 64)
image = torch.randn(batch, n_patches, d)    # (1, 9, 64)

# Cross-Attention (간략)
Q = text
K = image
V = image

# Attention 스코어
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
# (1, 5, 9) - 5개 토큰이 9개 패치를 보는 관계도

# Softmax
weights = F.softmax(scores, dim=-1)

# 결과
output = torch.matmul(weights, V)
# (1, 5, 64)

print(f"가중치 형태: {weights.shape}")
print(f"첫 번째 토큰의 주목도:\n{weights[0, 0, :]}")
# 예: [0.15, 0.08, 0.22, 0.05, ...]
# 9개 패치 중 3번째 패치에 가장 집중
```

### 실습 2: 도구 선택 에뮬레이션

```python
import numpy as np

# 사용자 입력들
inputs = [
    "파일을 읽고 요약해줘",
    "이 파일을 삭제해줄래?",
    "텍스트를 파일로 저장해줘",
]

# 도구 정의
tools = {
    "read": [1.0, 0.2],     # 읽기, 삭제와 독립적
    "summarize": [0.9, 0.1],
    "write": [0.1, 0.95],
    "delete": [0.2, 1.0],
}

# 간단한 임베딩 (실제로는 Transformer)
def simple_embed(text):
    if "읽" in text or "요약" in text:
        return np.array([1.0, 0.2])
    elif "삭제" in text:
        return np.array([0.1, 1.0])
    elif "저장" in text:
        return np.array([0.3, 0.8])
    else:
        return np.array([0.5, 0.5])

# 도구 선택
for user_input in inputs:
    user_vec = simple_embed(user_input)
    
    print(f"입력: {user_input}")
    
    similarities = {}
    for tool_name, tool_vec in tools.items():
        sim = np.dot(user_vec, tool_vec) / (
            np.linalg.norm(user_vec) * np.linalg.norm(tool_vec)
        )
        similarities[tool_name] = sim
    
    best_tool = max(similarities, key=similarities.get)
    print(f"선택: {best_tool}\n")
```

---

## 📚 핵심 개념 정리

| 개념 | 정의 | 예시 |
|------|------|------|
| **Self-Attention** | 같은 시퀀스 내에서 주목 | "나는 [강아지]를 본다" → "강아지"가 "나"를 attend |
| **Cross-Attention** | 다른 시퀀스에서 주목 | 텍스트 "강아지"가 이미지 패치들을 attend |
| **Late Fusion** | 모달리티 분리 후 합치기 | 텍스트 모델 + 이미지 모델 + Fusion |
| **Native Integration** | 처음부터 모달리티 섞기 | 하나의 Transformer에 [텍스트 + 이미지 패치] |
| **O(n²) 복잡도** | Attention의 계산량 병목 | 4096 토큰 입력 → 16M 연산 |

---

## 🎯 다음 강의 연결

- **Ch04-9: BERT와 Pre-trained Models**
  → Cross-Attention 아이디어가 BERT의 양방향 처리로 발전
  
- **Ch05: TTS와 STT**
  → Cross-Attention으로 음성-텍스트 상호작용

- **Ch08: LangChain & Agents**
  → 임베딩 기반 도구 선택의 실제 구현

---

**학습 체크리스트**:
- [ ] Self-Attention과 Cross-Attention의 차이 설명 가능?
- [ ] Muse Spark의 native integration 구조 이해?
- [ ] Attention 가중치를 시각화할 수 있나?
- [ ] 도구 선택에 임베딩을 쓰는 이유 알고 있나?
- [ ] O(n²) 복잡도 문제와 해결책 생각해봤나?
