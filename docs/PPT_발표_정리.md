# 온프레미스 스마트 명함·영수증 자동 정리 에이전트
## 딥러닝 기말 프로젝트 — PPT 발표용 정리

> 이 문서는 슬라이드별로 복사·붙여넣기 할 수 있도록 구성했습니다.  
> `---` 구분선 = 슬라이드 1장 권장 분량

---

# Slide 1. 표지

**온프레미스 로컬 딥러닝 기반**  
**스마트 명함·영수증 자동 정리 에이전트**

- 과목: 딥러닝 (기말 프로젝트)
- 팀: 4인
- 핵심 키워드: Vision + NLP · 완전 로컬 · 사무 자동화 · 데이터 보안

---

# Slide 2. 프로젝트 배경 & 문제 정의

### 왜 이 프로젝트인가?

| 문제 | 현실 |
|------|------|
| 비정형 문서 관리 비효율 | 영수증·명함을 사진으로만 보관, 수동 입력 반복 |
| 외부 AI API 의존 | GPT 등 클라우드 API → **개인정보·거래정보 유출 위험** |
| 비용 | API 호출 비용 누적, 오프라인 환경 사용 불가 |

### 우리의 접근

> **노트북 로컬 환경**에서 Vision(이미지) + NLP(의미 파악)를 결합해  
> 명함/영수증을 **자동 구조화 → Excel/DB 저장**하는 온프레미스 에이전트 개발

---

# Slide 3. 프로젝트 목표

1. **완전 로컬 구동** — 외부 서버·API 키 없이 모든 추론 수행
2. **하이브리드 딥러닝 파이프라인** — OCR + 문서 분류 + LLM 구조화 통합
3. **실사용 가능한 데스크톱 앱** — PyQt6 GUI, 날짜별 Excel 누적 저장
4. **안정적 데모** — 단계별 Fallback으로 모델 미설치·오류 시에도 동작

---

# Slide 4. 시스템 아키텍처 (전체 흐름)

```
[사용자] 이미지 업로드 (명함 / 영수증 사진)
    │
    ▼
① OpenCV 전처리 ─── 기울기 보정, CLAHE 대비 향상, 컬러 샤프닝
    │
    ▼
② OCR (EasyOCR) ─── 한국어 텍스트 + 위치(bbox) 추출
    │                  ※ 전처리본·원본 이중 인식 후 고품질 결과 채택
    ▼
③ 문서 분류 ───────── LayoutLM (영수증 / 명함 / 미상)
    │                  ※ 가중치 없으면 키워드 규칙 Fallback
    ▼
④ 정보 구조화 ─────── Ollama + Llama-3 → JSON 필드 추출
    │                  ※ Pydantic 검증 + 정규식·원문 대조 Fallback
    ▼
⑤ 저장 ────────────── SQLite 이력 + 날짜별 Excel 누적 (documents_YYYYMMDD.xlsx)
    │
    ▼
[PyQt6 데스크톱 GUI] 미리보기 · OCR · 추출 결과 · 처리 이력
```

---

# Slide 5. 핵심 기술 스택 (한눈에)

| 단계 | 기술 | 역할 |
|------|------|------|
| **이미지 전처리** | OpenCV | 기울기 보정, LAB-CLAHE, 샤프닝 → OCR 인식률 향상 |
| **텍스트 추출 (OCR)** | EasyOCR (PyTorch) | 한국어 Detection + Recognition, bbox 좌표 출력 |
| **문서 분류** | LayoutLM (Transformer) | OCR 텍스트+bbox로 영수증/명함/미상 분류 |
| **정보 구조화** | Ollama + Llama-3 | 로컬 LLM으로 필드별 JSON 추출 |
| **데이터 검증** | Pydantic v2 | 스키마 검증, 타입·형식 보장 |
| **저장** | SQLite + pandas/openpyxl | 처리 이력 DB + 날짜별 Excel 누적 |
| **UI** | PyQt6 | macOS/Windows 데스크톱 GUI (파일 선택·진행률·결과 탭) |

---

# Slide 6. 핵심 기술 ① — OpenCV 전처리 (Vision)

### 무엇을 하나?
명함·영수증 사진의 **조명 불균일, 기울기, 흐림**을 보정해 OCR 정확도를 높인다.

### 기술 포인트
- **기울기 추정 (minAreaRect)** → 컬러 이미지 회전 보정
- **LAB 색공간 CLAHE** → 밝기 채널만 대비 향상 (한글 획 보존)
- **언샤프 마스크** → 작은 글씨(카드명, 금액) 선명화
- **해상도 상한 1920px** — 과축소로 인한 글자 손실 방지

### 딥러닝 연관
전처리 품질이 downstream **OCR·분류·LLM** 전 단계 성능에 직접 영향

**담당:** CV Engineer  
**코드:** `src/preprocessing/image_preprocess.py`

### 핵심 코드

```python
# src/preprocessing/image_preprocess.py — 컬러 CLAHE + 샤프닝
def _enhance_color(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)

# 기울기 보정 → 컬러 이미지에 회전 적용
angle = _estimate_skew(gray)
if angle != 0.0:
    matrix = cv2.getRotationMatrix2D((ww / 2, wh / 2), angle, 1.0)
    work = cv2.warpAffine(work, matrix, (ww, wh), ...)
processed = _enhance_color(work)
```

---

# Slide 7. 핵심 기술 ② — EasyOCR (딥러닝 OCR)

### 무엇을 하나?
이미지에서 **한국어 텍스트와 위치(bounding box)**를 추출한다.

### 기술 포인트
- **Detection + Recognition** 2단계 딥러닝 모델 (CRAFT + CRNN 계열)
- `canvas_size=2240`, `mag_ratio=1.5` — 작은 글씨 인식 강화
- **이중 OCR 전략**: 전처리 이미지 + 원본 이미지 각각 인식 → **신뢰도 합산 점수가 높은 결과 채택**
- 출력: `[{text, confidence, bbox}]` + `full_text`

### 왜 EasyOCR?
- Mac ARM 환경에서 PaddleOCR CPU 멈춤 이슈 → **EasyOCR로 안정적 로컬 구동**
- 한국어(`ko`) + 영어(`en`) 동시 지원

**담당:** CV Engineer  
**코드:** `src/ocr/easy_ocr_engine.py`, `src/ocr/__init__.py`

### 핵심 코드

```python
# src/ocr/easy_ocr_engine.py — EasyOCR + 작은 글씨 강화 파라미터
_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

raw = reader.readtext(
    path,
    paragraph=False,
    canvas_size=2240,    # 작은 글씨 탐지 영역 확대
    mag_ratio=1.5,       # 확대 비율
    text_threshold=0.6,
    low_text=0.35,
)

# bbox → OCRWord 리스트, 위→아래 정렬 후 full_text 생성
words.append(OCRWord(text=text, confidence=conf, bbox=[x0,y0,x1,y1]))
full_text = "\n".join(w.text for w in words)
```

```python
# src/pipeline/agent.py — 이중 OCR (전처리본 vs 원본)
ocr_proc = run_ocr(pre.processed)
ocr_orig = run_ocr(pre.original)
ocr = ocr_proc if sum(w.confidence for w in ocr_proc.words) \
              >= sum(w.confidence for w in ocr_orig.words) else ocr_orig
```

---

# Slide 8. 핵심 기술 ③ — LayoutLM (문서 이해·분류)

### 무엇을 하나?
OCR 결과(텍스트 + bbox)를 입력받아 **영수증 / 명함 / 미상**으로 분류한다.

### 기술 포인트
- **microsoft/layoutlm-base-uncased** — 문서 레이아웃+텍스트 동시 이해 Transformer
- `LayoutLMForSequenceClassification` — 3-class 분류
- bbox 좌표를 **0~1000 정규화** 후 토큰별 정렬
- **소량 파인튜닝** (`scripts/train_layoutlm.py`, epochs=3)

### Fallback
학습 데이터 부족·가중치 미설치 시 → **키워드 빈도 규칙**  
(예: `합계`, `부가세` → 영수증 / `Tel`, `@`, `대표` → 명함)

**담당:** NLP Engineer (+ PM 통합 관리)  
**코드:** `src/classification/layoutlm_classifier.py`

### 핵심 코드

```python
# src/classification/layoutlm_classifier.py — LayoutLM 추론
words = [w.text for w in ocr.words]
boxes = [_normalize_bbox(w.bbox, max_x, max_y) for w in ocr.words]  # 0~1000

encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", ...)
encoding["bbox"] = torch.tensor([token_boxes])  # 토큰별 bbox 정렬

with torch.no_grad():
    logits = model(**encoding).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))  # receipt / business_card / unknown
```

```python
# 키워드 Fallback (LayoutLM 미사용·저신뢰 시)
receipt_score = sum(1 for kw in _RECEIPT_KEYWORDS if kw in text)  # 합계, 부가세...
card_score    = sum(1 for kw in _CARD_KEYWORDS if kw in text)     # Tel, @, 대표...
doc_type = RECEIPT if receipt_score >= card_score else BUSINESS_CARD
```

---

# Slide 9. 핵심 기술 ④ — Ollama + Llama-3 (로컬 LLM 구조화)

### 무엇을 하나?
OCR 원문을 **구조화된 JSON**으로 변환한다.

| 문서 | 추출 필드 |
|------|-----------|
| **영수증** | 상호명, 거래일, 품목, 합계, 부가세, 결제수단 |
| **명함** | 이름, 회사, 직책, 전화, 이메일, 주소, 웹사이트 |

### 기술 포인트
- **Ollama** (`localhost:11434`) — API 키 없는 완전 로컬 LLM 서버
- **Llama-3** + `format: json` — JSON 출력 강제
- **프롬프트 엔지니어링** — "OCR 원문 verbatim 복사, 카드명 환각 금지" 규칙
- **사후 검증** — 결제수단이 원문에 없으면 정규식으로 교정 (예: `새체크카드` → `NH카드`)
- **Pydantic v2** — 스키마 검증

### Fallback
Ollama 미연결 시 → 정규식(전화·이메일·금액·카드명·날짜) 추출

**담당:** NLP Engineer  
**코드:** `src/extraction/ollama_extractor.py`, `prompts.py`

### 핵심 코드

```python
# src/extraction/prompts.py — LLM 프롬프트 (환각 방지 규칙)
_BASE_INSTRUCTION = (
    "Copy every value VERBATIM from the OCR text. NEVER invent...\n"
    "payment_method must be the exact card name (e.g. 'NH카드', 'IBK비씨카드')..."
)
```

```python
# src/extraction/ollama_extractor.py — Ollama 호출 + 결제수단 검증
payload = {
    "model": "llama3", "prompt": prompt,
    "stream": False, "format": "json",   # JSON 출력 강제
    "options": {"temperature": 0.0},
}
resp = requests.post(f"{host}/api/generate", json=payload, timeout=120)

# LLM이 '새체크카드'라고 해도 원문에 없으면 → OCR에서 NH카드 추출
def _fix_payment_method(ocr_text, llm_value):
    if _squash(llm_value) in _squash(ocr_text):
        return llm_value
    candidates = _CARD_RE.findall(ocr_text)  # NH카드, IBK비씨카드...
    return candidates[0] if candidates else ""
```

---

# Slide 10. 핵심 기술 ⑤ — 저장 & UI

### SQLite
- 처리 이력 영속화 (파일명, 유형, OCR 원문, JSON, 타임스탬프)
- GUI **처리 이력** 탭에서 조회

### Excel 날짜별 누적
- `documents_20260610.xlsx` 형식 — **하루 1파일에 행 추가**
- 시트 분리: `Receipts` / `BusinessCards`
- 중복(파일+처리시각) 자동 제거

### PyQt6 데스크톱 GUI
- 파일·폴더 선택 다이얼로그
- 진행률 바 + 백그라운드 스레드 처리 (UI 멈춤 방지)
- 탭: 미리보기 / OCR / 추출 결과 / 처리 이력

**담당:** Dev Engineer  
**코드:** `src/storage/`, `app_gui.py`

### 핵심 코드

```python
# src/storage/excel_exporter.py — 날짜별 Excel 누적
filename = f"documents_{datetime.now():%Y%m%d}.xlsx"  # 하루 1파일
receipt_df = pd.concat([기존_데이터, new_receipts]).drop_duplicates(
    subset=["파일", "처리시각"], keep="last"
)
receipt_df.to_excel(writer, sheet_name="Receipts", index=False)
```

```python
# app_gui.py — 백그라운드 처리 (UI 멈춤 방지)
class ProcessThread(QThread):
    progress = pyqtSignal(str, float)
    def run(self):
        agent = DocumentAgent(persist=True)
        result, _ = agent.process_image(path, progress=cb, ...)
        export_to_excel(results)  # 처리 완료 후 Excel 자동 저장
```

---

# Slide 11. 하이브리드 파이프라인 & Fallback 설계

### 왜 하이브리드?
단일 모델만으로는 **인식 오류·환각·환경 의존성**을 모두 커버하기 어렵다.

| 단계 | 1차 (딥러닝) | 2차 (Fallback) |
|------|-------------|----------------|
| OCR | EasyOCR 이중 인식 | 원본/전처리 중 고품질 채택 |
| 분류 | LayoutLM | 키워드 규칙 |
| 구조화 | Ollama Llama-3 | 정규식 + 원문 대조 검증 |
| 금액 | LLM 추출 | 합계 키워드 라인 + 타당 범위(10원~1억) 검증 |

> **발표 시 강조:** 외부 API 없이도, 모델 일부 미설치 시에도 **데모가 끊기지 않는** 엔지니어링

---

# Slide 12. 팀 역할 분담 (4인) — 개요

| # | 역할 | 한 줄 요약 |
|---|------|-----------|
| 1 | **PM & AI Lead** | 전체 아키텍처·모델 선정·파이프라인 통합 |
| 2 | **CV Engineer** | 이미지 전처리 + OCR 최적화 |
| 3 | **NLP Engineer** | 문서 분류 + 로컬 LLM 구조화 |
| 4 | **Dev Engineer** | GUI·Excel·DB·사용자 경험 |

---

# Slide 13. 역할 분담 상세 — ① PM & AI Lead

### 담당 영역
- 프로젝트 기획, 일정·역할 조율
- **end-to-end 파이프라인 설계** 및 모듈 통합
- 기술 스택 선정 (EasyOCR / LayoutLM / Ollama 하이브리드)
- Fallback 전략 수립

### 주요 산출물
| 파일 | 내용 |
|------|------|
| `src/pipeline/agent.py` | DocumentAgent — 5단계 오케스트레이션 |
| `config/settings.yaml` | 모델명, 임계값, 경로 설정 |
| `src/schemas.py` | Pydantic 데이터 스키마 정의 |

### 발표·시연 담당
- **전체 아키텍처 슬라이드** 설명
- 파이프라인 end-to-end 데모 오케스트레이션
- Q&A: "왜 외부 API를 안 쓰나?" / "하이브리드가 뭔가?"

### 핵심 코드 — 파이프라인 오케스트레이터

```python
# src/pipeline/agent.py — DocumentAgent.process_image()
class DocumentAgent:
    def process_image(self, image_path, *, progress=_noop, ...) -> tuple[DocumentResult, PreprocessResult]:
        # 1) OpenCV 전처리
        pre = preprocess_image(image_path)

        # 2) EasyOCR 이중 인식 (전처리본 + 원본 → 고품질 채택)
        ocr_proc = run_ocr(pre.processed)
        ocr_orig = run_ocr(pre.original)
        ocr = ocr_proc if _score(ocr_proc) >= _score(ocr_orig) else ocr_orig

        # 3) LayoutLM / 키워드 분류
        classification = classify_document(ocr)

        # 4) Ollama JSON 구조화
        if classification.doc_type == DocumentType.RECEIPT:
            receipt = extract_receipt(ocr.full_text)
        elif classification.doc_type == DocumentType.BUSINESS_CARD:
            business_card = extract_business_card(ocr.full_text)

        # 5) SQLite 저장
        save_result(DocumentResult(...))
        return result, pre
```

```python
# src/schemas.py — Pydantic 데이터 스키마
class DocumentType(str, Enum):
    RECEIPT = "receipt"
    BUSINESS_CARD = "business_card"
    UNKNOWN = "unknown"

class ReceiptData(BaseModel):
    merchant: str = ""
    date: str = ""
    items: list[ReceiptItem] = []
    total: Optional[float] = None
    payment_method: str = ""
```

---

# Slide 14. 역할 분담 상세 — ② CV Engineer

### 담당 영역
- **OpenCV 이미지 전처리** (기울기·대비·샤프닝)
- **EasyOCR** 연동 및 Mac 환경 최적화
- OCR 인식률 향상 (이중 인식, 파라미터 튜닝)
- (선택) PaddleOCR Linux/Windows 대안 엔진 유지

### 주요 산출물
| 파일 | 내용 |
|------|------|
| `src/preprocessing/image_preprocess.py` | CLAHE, deskew, 컬러 보정 |
| `src/ocr/easy_ocr_engine.py` | EasyOCR 래퍼, mag_ratio 튜닝 |
| `src/ocr/paddle_ocr_engine.py` | PaddleOCR (비-Mac 대안) |

### 발표·시연 담당
- **Vision 파트** 슬라이드 (전처리 전/후, OCR bbox)
- `notebooks/demo_pipeline.ipynb` — 1~2단계 셀 시연
- Q&A: "전처리 없이 OCR하면?" / "왜 EasyOCR?"

---

# Slide 15. 역할 분담 상세 — ③ NLP Engineer

### 담당 영역
- **LayoutLM** 문서 분류 (영수증/명함/미상)
- **Ollama + Llama-3** 프롬프트 설계 및 JSON 파싱
- LLM **환각 방지** (결제수단 원문 대조, 금액 타당성 검증)
- 키워드 Fallback 분류 규칙
- (선택) LayoutLM 소량 파인튜닝

### 주요 산출물
| 파일 | 내용 |
|------|------|
| `src/classification/layoutlm_classifier.py` | LayoutLM + 키워드 fallback |
| `src/extraction/ollama_extractor.py` | Ollama 호출, 검증, regex fallback |
| `src/extraction/prompts.py` | 문서 유형별 프롬프트 |
| `scripts/train_layoutlm.py` | LayoutLM 파인튜닝 |

### 발표·시연 담당
- **NLP 파트** 슬라이드 (분류 confidence, JSON 구조화)
- Ollama 연결 상태 / Fallback 동작 설명
- Q&A: "LLM이 카드명을 틀리면?" / "LayoutLM 입력 형식은?"

---

# Slide 16. 역할 분담 상세 — ④ Dev Engineer

### 담당 영역
- **PyQt6 데스크톱 GUI** 개발 (파일 선택, 진행률, 결과 탭)
- **SQLite** 처리 이력 저장
- **Excel 날짜별 누적**보내기
- CLI (`main.py`) 보조 인터페이스
- 설치·실행 가이드 (`README.md`)

### 주요 산출물
| 파일 | 내용 |
|------|------|
| `app_gui.py` | PyQt6 메인 GUI (권장 실행) |
| `src/storage/db.py` | SQLite CRUD |
| `src/storage/excel_exporter.py` | 날짜별 Excel 누적 |
| `main.py` | CLI (배치 처리용) |

### 발표·시연 담당
- **라이브 GUI 데모** (파일 추가 → 처리 → Excel 확인)
- 처리 이력 탭, 출력 폴더 열기 시연
- Q&A: "Excel은 어떻게 쌓이나?" / "GUI가 멈추지 않는 이유는?"

---

# Slide 17. 역할 분담 — 발표 순서 제안 (4인)

| 순서 | 발표자 | 시간 | 내용 |
|------|--------|------|------|
| 1 | PM & AI Lead | 2~3분 | 배경, 목표, 전체 아키텍처 |
| 2 | CV Engineer | 2분 | OpenCV 전처리 + EasyOCR |
| 3 | NLP Engineer | 2분 | LayoutLM 분류 + Ollama 구조화 |
| 4 | Dev Engineer | 2~3분 | GUI 라이브 데모 + Excel/DB |
| 전체 | 4인 | 1분 | 기대 효과 + Q&A |

**총 발표 시간:** 약 10~12분 + Q&A

---

# Slide 18. 라이브 데모 시나리오 (Dev Engineer 주도)

1. `python app_gui.py` 실행
2. **파일 추가** → 영수증 사진 1장 + 명함 사진 1장
3. **처리 시작** 클릭 → 진행률 확인
4. **OCR 텍스트** 탭 — 추출 원문 확인
5. **추출 결과** 탭 — JSON 필드 (상호명, 합계, 결제수단 등)
6. **출력 폴더 열기** → `documents_YYYYMMDD.xlsx` 누적 확인
7. (보너스) Ollama 끄고 재실행 → Fallback 동작 설명

---

# Slide 19. 기대 효과 & 차별점

| 항목 | 내용 |
|------|------|
| **보안** | 데이터가 노트북 밖으로 나가지 않음 (온프레미스) |
| **비용** | API 호출료 0원, Ollama·EasyOCR 무료 오픈소스 |
| **자동화** | 명함/영수증 → Excel 정리, 수동 입력 대폭 감소 |
| **딥러닝 학습** | OCR(Detection/Rec) + Transformer(LayoutLM) + LLM 추론 경험 |
| **안정성** | 3중 Fallback — 발표·실사용 시 장애 최소화 |

---

# Slide 20. 한계 & 향후 개선

| 한계 | 개선 방향 |
|------|-----------|
| 저화질 사진에서 금액 OCR 오류 | 영수증 전용 Detection 모델 파인튜닝 |
| LayoutLM 학습 데이터 부족 | 라벨 데이터 확대 + confidence 임계값 조정 |
| LLM 구조화 지연 (30초~2분) | 경량 모델(llama3.2:1b) 또는 규칙 기반 우선 처리 |
| 명함 레이아웃 다양성 | LayoutLMv3 / Donut 등 end-to-end 모델 검토 |

---

# Slide 21. Q&A 예상 질문 & 답변

**Q. GPT API 쓰면 더 정확하지 않나?**  
A. 정확도는 높을 수 있으나, 영수증·명함에는 **개인정보·거래정보**가 포함됩니다. 본 프로젝트는 **보안·비용·오프라인**을 위해 로컬 LLM을 선택했습니다.

**Q. 딥러닝 모델을 직접 학습했나?**  
A. EasyOCR·LayoutLM·Llama-3는 **사전학습 모델**을 사용하고, LayoutLM은 소량 샘플로 **파인튜닝 가능**하도록 `train_layoutlm.py`를 구현했습니다.

**Q. 인식이 틀리면?**  
A. 단계별 Fallback + 결제수단·금액 **사후 검증**으로 환각을 줄입니다. GUI에서 OCR 원문과 추출 결과를 나란히 확인할 수 있습니다.

**Q. Mac에서만 되나?**  
A. PyQt6 GUI는 macOS/Windows/Linux 공통. OCR은 Mac은 EasyOCR, Linux/Windows는 PaddleOCR 설정 가능.

---

# 부록 A. 기술 스택 버전 (requirements.txt 기준)

```
OpenCV          opencv-python
EasyOCR         easyocr (PyTorch 기반)
LayoutLM        transformers + torch
Ollama LLM      llama3 (로컬)
검증            pydantic
저장            sqlite3, pandas, openpyxl
UI              PyQt6
```

---

# 부록 B. 프로젝트 디렉터리 (발표용 간략)

```
DeepLearning/
├── app_gui.py          ← GUI 실행 (권장)
├── main.py             ← CLI
├── config/settings.yaml
├── src/
│   ├── preprocessing/  ← CV
│   ├── ocr/            ← CV
│   ├── classification/ ← NLP
│   ├── extraction/     ← NLP
│   ├── storage/        ← Dev
│   └── pipeline/       ← PM
├── scripts/            ← LayoutLM 학습
├── data/output/        ← Excel 산출물
└── notebooks/          ← 단계별 시연
```

---

# 부록 C. 팀 역할 ↔ 파일 매핑 (한 장 요약)

```
┌─────────────────┬──────────────────────────────────────────┐
│ PM & AI Lead    │ agent.py · schemas.py · settings.yaml    │
├─────────────────┼──────────────────────────────────────────┤
│ CV Engineer     │ preprocessing/ · ocr/                      │
├─────────────────┼──────────────────────────────────────────┤
│ NLP Engineer    │ classification/ · extraction/ · scripts/ │
├─────────────────┼──────────────────────────────────────────┤
│ Dev Engineer    │ app_gui.py · storage/ · main.py          │
└─────────────────┴──────────────────────────────────────────┘
```

---

---

# 부록 D. 핵심 코드 전체 모음 (PPT 코드 슬라이드용)

> 발표 시 **코드 슬라이드**에 그대로 붙여넣기 가능.  
> 역할별로 해당 섹션만 발표해도 됩니다.

---

## D-1. PM & AI Lead — 파이프라인 통합 (`agent.py`)

```python
"""DocumentAgent: end-to-end 오케스트레이션."""
from src.classification.layoutlm_classifier import classify_document
from src.extraction.ollama_extractor import extract_receipt, extract_business_card
from src.ocr import run_ocr
from src.preprocessing.image_preprocess import preprocess_image
from src.schemas import DocumentResult, DocumentType
from src.storage.db import save_result


class DocumentAgent:
    def process_image(self, image_path, *, progress=_noop, verbose=False):
        warnings: list[str] = []

        # ① 전처리
        progress("이미지 전처리", 0.08)
        pre = preprocess_image(image_path)

        # ② OCR — 전처리본·원본 이중 인식
        ocr_proc = run_ocr(pre.processed, verbose=verbose)
        ocr_orig = run_ocr(pre.original, verbose=verbose)
        score = lambda o: sum(w.confidence for w in o.words)
        ocr = ocr_proc if score(ocr_proc) >= score(ocr_orig) else ocr_orig

        # ③ 문서 분류 (LayoutLM → 키워드 fallback)
        classification = classify_document(ocr)

        # ④ 정보 구조화 (Ollama → 정규식 fallback)
        receipt, business_card = None, None
        if classification.doc_type == DocumentType.RECEIPT:
            receipt = extract_receipt(ocr.full_text)
        elif classification.doc_type == DocumentType.BUSINESS_CARD:
            business_card = extract_business_card(ocr.full_text)

        result = DocumentResult(
            source_file=source, ocr=ocr,
            classification=classification,
            receipt=receipt, business_card=business_card,
            warnings=warnings,
        )

        # ⑤ DB 저장
        if self.persist and classification.doc_type != DocumentType.UNKNOWN:
            save_result(result)

        return result, pre
```

---

## D-2. CV Engineer — 이미지 전처리 (`image_preprocess.py`)

```python
def _estimate_skew(gray: np.ndarray) -> float:
    """minAreaRect로 기울기 각도 추정."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 15 or abs(angle) < 0.5:
        return 0.0
    return float(angle)


def _enhance_color(img: np.ndarray) -> np.ndarray:
    """LAB-CLAHE + 언샤프 마스크 (한글 획 보존)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)


def preprocess_image(image, *, do_deskew=True, max_side=1920) -> PreprocessResult:
    original = _read_image(image)          
    work = resize_if_needed(original, max_side)
    if do_deskew:
        angle = _estimate_skew(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
        work = rotate_if_needed(work, angle)
    processed = _enhance_color(work)
    return PreprocessResult(original=original, processed=processed, debug=debug)
```

---

## D-3. CV Engineer — EasyOCR (`easy_ocr_engine.py`)

```python
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


def run_ocr(image, *, verbose=False) -> OCRResult:
    reader = _get_reader()
    raw = reader.readtext(
        path,
        paragraph=False,
        canvas_size=2240,   # Detection 영역 확대
        mag_ratio=1.5,      # Recognition 확대
        text_threshold=0.6,
        low_text=0.35,
    )

    words: list[OCRWord] = []
    for bbox, text, conf in raw:
        xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
        words.append(OCRWord(
            text=text.strip(),
            confidence=float(conf),
            bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
        ))

    words.sort(key=lambda w: (w.bbox[1], w.bbox[0]))  # 위→아래, 좌→우
    return OCRResult(words=words, full_text="\n".join(w.text for w in words))
```

---

## D-4. NLP Engineer — LayoutLM 분류 (`layoutlm_classifier.py`)

```python
def _normalize_bbox(bbox, width, height) -> list[int]:
    """OCR 픽셀 좌표 → LayoutLM 0~1000 좌표계."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def _layoutlm_classify(ocr: OCRResult) -> ClassificationResult | None:
    words = [w.text for w in ocr.words]
    boxes = [_normalize_bbox(w.bbox, max_x, max_y) for w in ocr.words]

    encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", ...)
    encoding["bbox"] = torch.tensor([token_boxes])

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))

    return ClassificationResult(
        doc_type=DocumentType(labels[idx]),
        confidence=round(float(probs[idx]), 3),
        source="layoutlm",
    )


def classify_document(ocr: OCRResult) -> ClassificationResult:
    layout_result = _layoutlm_classify(ocr)
    if layout_result and layout_result.confidence >= threshold:
        return layout_result
    return _fallback_classify(ocr)  # 키워드 규칙
```

---

## D-5. NLP Engineer — Ollama 구조화 (`ollama_extractor.py` + `prompts.py`)

```python
# prompts.py — 프롬프트 (verbatim 복사 규칙)
def build_prompt(doc_type: str, ocr_text: str) -> str:
    schema = BUSINESS_CARD_SCHEMA if doc_type == "business_card" else RECEIPT_SCHEMA
    return (
        f"{_BASE_INSTRUCTION}\n"
        f"Document type: {kind}\n"
        f"JSON schema:\n{schema}\n"
        f'OCR text:\n"""\n{ocr_text}\n"""\n'
        f"JSON:"
    )


# ollama_extractor.py — Ollama API 호출
def _call_ollama(prompt: str) -> str | None:
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=120)
    return resp.json().get("response", "")


# 결제수단 환각 방지 (NH카드 → 새체크카드 오인 수정)
def _fix_payment_method(ocr_text: str, llm_value: str) -> str:
    if llm_value and _squash(llm_value) in _squash(ocr_text):
        return llm_value
    candidates = _CARD_RE.findall(ocr_text)  # [A-Za-z가-힣]+카드
    for cand in candidates:
        if any(issuer.lower() in cand.lower() for issuer in _CARD_ISSUERS):
            return cand
    return candidates[0] if candidates else ""


def extract_receipt(ocr_text: str) -> ReceiptData:
    for _ in range(retries):
        data = _extract_json(_call_ollama(build_prompt("receipt", ocr_text)))
        if data:
            receipt = ReceiptData(
                merchant=str(data.get("merchant", "")),
                date=_normalize_date(str(data.get("date", "")), ocr_text),
                payment_method=_fix_payment_method(ocr_text, str(data.get("payment_method", ""))),
                total=_to_float(data.get("total")),
            )
            if not _plausible_amount(receipt.total):
                receipt.total = _find_total(ocr_text)  # 합계 키워드 라인에서 추출
            return receipt
    return _regex_receipt(ocr_text)  # fallback
```

---

## D-6. Dev Engineer — Excel 누적 + GUI (`excel_exporter.py` + `app_gui.py`)

```python
# excel_exporter.py — 날짜별 1파일 누적
def export_to_excel(results, filename=None) -> Path:
    filename = filename or f"documents_{datetime.now():%Y%m%d}.xlsx"
    out_path = out_dir / filename

    receipt_df = pd.concat([
        _load_existing(out_path, "Receipts", _RECEIPT_COLUMNS),
        pd.DataFrame(_receipt_rows(results)),
    ], ignore_index=True).drop_duplicates(subset=["파일", "처리시각"], keep="last")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        receipt_df.to_excel(writer, sheet_name="Receipts", index=False)
        card_df.to_excel(writer, sheet_name="BusinessCards", index=False)
    return out_path
```

```python
# app_gui.py — PyQt6 백그라운드 스레드
class ProcessThread(QThread):
    progress = pyqtSignal(str, float)
    result_ready = pyqtSignal(object)

    def run(self) -> None:
        agent = DocumentAgent(persist=True)
        for path in self.files:
            result, _ = agent.process_image(
                path,
                progress=lambda s, p: self.progress.emit(s, p),
                source_name=path.name,
            )
            self.result_ready.emit(result)
        export_to_excel(results)  # documents_YYYYMMDD.xlsx
```

```python
# db.py — SQLite 이력 저장
def save_result(result: DocumentResult) -> int:
    conn.execute(
        "INSERT INTO documents (source_file, doc_type, confidence, ocr_text, structured, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (result.source_file, result.classification.doc_type.value,
         result.classification.confidence, result.ocr.full_text,
         json.dumps(result.structured_dict(), ensure_ascii=False),
         result.processed_at.isoformat()),
    )
```

---

## D-7. 설정 파일 (`config/settings.yaml`)

```yaml
ocr:
  engine: "easyocr"       # Mac: easyocr | Linux: paddle
  lang: "korean"

classification:
  model_name: "microsoft/layoutlm-base-uncased"
  labels: ["receipt", "business_card", "unknown"]
  confidence_threshold: 0.55

extraction:
  ollama_host: "http://localhost:11434"
  model: "llama3"
  temperature: 0.0
  timeout_seconds: 120

paths:
  output_dir: "data/output"
  database: "data/documents.db"
```

---

## D-8. 코드 슬라이드 배치 가이드 (4인 발표)

| 발표자 | 추천 코드 슬라이드 | 부록 섹션 |
|--------|-------------------|-----------|
| PM & AI Lead | `DocumentAgent.process_image()` 전체 흐름 | D-1, D-7 |
| CV Engineer | `_enhance_color()` + `run_ocr()` 이중 인식 | D-2, D-3 |
| NLP Engineer | `classify_document()` + `_fix_payment_method()` | D-4, D-5 |
| Dev Engineer | `export_to_excel()` + `ProcessThread` | D-6 |

---

*문서 작성일: 2026-06-10 · 프로젝트 저장소: smart-document-agent*
