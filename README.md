# 온프레미스 스마트 명함·영수증 자동 정리 에이전트

대학 딥러닝 과목 기말 프로젝트. **외부 API(GPT 등)를 사용하지 않고** 노트북 로컬 환경에서
Vision(이미지 처리)과 NLP(의미 파악)를 결합해 명함/영수증을 자동으로 구조화하고 저장합니다.

> 데이터 유출 걱정 없는 보안 솔루션 · 사무 자동화 · 딥러닝 모델 경량화/최적화 경험

## 아키텍처

```
이미지 입력
   └─► OpenCV 전처리 (그레이스케일/CLAHE/디노이즈/기울기 보정)
        └─► PaddleOCR (한국어 텍스트 추출 + bbox)
             └─► LayoutLM 분류 (영수증 / 명함 / 미상)   ← 가중치 없으면 키워드 fallback
                  └─► Ollama(Llama-3) JSON 구조화        ← 미연결 시 정규식 fallback
                       └─► SQLite 저장 + Excel 내보내기
                            └─► CLI (main.py)
```

## 기술 스택

| 단계 | 기술 |
|---|---|
| 이미지 전처리 | OpenCV |
| 텍스트 추출(OCR) | PaddleOCR (한국어) |
| 문서 분류 | LayoutLM (`microsoft/layoutlm-base-uncased`) |
| 정보 구조화 | Ollama 로컬 LLM (Llama-3) |
| 검증 | Pydantic v2 |
| 저장/내보내기 | SQLite, pandas + openpyxl |
| UI | Python CLI (`main.py`) |

## 프로젝트 구조

```
DeepLearning/
├── main.py                         # CLI 메인 프로그램 (권장)
├── app.py                          # Streamlit UI (선택)
├── requirements.txt
├── config/settings.yaml            # 모델/경로/임계값 설정
├── src/
│   ├── config.py                   # 설정 로더
│   ├── schemas.py                  # Pydantic 데이터 스키마
│   ├── preprocessing/image_preprocess.py
│   ├── ocr/paddle_ocr_engine.py
│   ├── classification/layoutlm_classifier.py
│   ├── extraction/ollama_extractor.py, prompts.py
│   ├── storage/db.py, excel_exporter.py
│   └── pipeline/agent.py           # end-to-end 오케스트레이터
├── scripts/download_models.py      # LayoutLM 가중치 다운로드
├── scripts/train_layoutlm.py       # 소량 파인튜닝
├── data/samples/                   # 데모/학습 이미지 + labels.csv
├── data/output/                    # 생성 Excel
└── notebooks/demo_pipeline.ipynb   # 단계별 시연
```

## 설치

```bash
cd /Users/kimyounghyun/DeepLearning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사전 준비

1. **Ollama** (구조화 단계)

```bash
# https://ollama.com 설치 후
ollama pull llama3
ollama list   # llama3 확인
```

2. **LayoutLM 가중치** (선택)

```bash
python scripts/download_models.py        # base 모델 캐싱
# 샘플 라벨 학습 (data/samples/labels.csv 준비 후)
python scripts/train_layoutlm.py
```

> 가중치가 없거나 신뢰도가 낮으면 **키워드 fallback**으로 자동 동작하므로
> Ollama/LayoutLM 없이도 데모가 가능합니다.

3. **샘플 이미지**: `data/samples/`에 명함/영수증 이미지를 넣고 `labels.csv`를 맞추세요.

## 실행 (Python CLI)

```bash
source .venv/bin/activate
python main.py                    # 대화형 메뉴
python main.py receipt.jpg          # 단일 이미지
python main.py -f data/samples    # 폴더 일괄 처리
python main.py --history          # DB 이력 조회
```

**첫 실행 시 PaddleOCR 모델 다운로드로 1~3분 걸릴 수 있습니다.** 터미널에 진행 메시지가 표시됩니다.

> (선택) 웹 UI: `streamlit run app.py`


단계별 디버깅은 `notebooks/demo_pipeline.ipynb`에서 셀 단위로 실행할 수 있습니다.

## 팀 역할 분담 (4인)

| 역할 | 담당 | 주요 파일 |
|---|---|---|
| **PM & AI Lead** | 아키텍처 설계, 모델 선정/통합 | `src/pipeline/agent.py`, `config/settings.yaml` |
| **CV Engineer** | OpenCV 전처리, PaddleOCR 최적화, 왜곡 보정 | `src/preprocessing/`, `src/ocr/` |
| **NLP Engineer** | 로컬 LLM(Ollama), 프롬프트/파싱 로직 | `src/extraction/`, `src/classification/` |
| **Dev Engineer** | CLI UI, Excel/DB 저장 자동화 | `main.py`, `src/storage/` |

## 폴백(Fallback) 설계

외부 의존성이 없어도 데모가 끊기지 않도록 단계별 대비책을 둡니다.

| 단계 | 정상 경로 | Fallback |
|---|---|---|
| 분류 | LayoutLM 추론 | 키워드 빈도 규칙 |
| 구조화 | Ollama JSON 생성 | 정규식(전화/이메일/금액) 추출 |
| OCR | angle-cls 포함 | angle-cls 미적용 재시도 |

## 기대 효과

- 데이터 유출 걱정 없는 **완전 로컬** 보안 솔루션
- 비정형 문서(명함/영수증)의 사무 자동화
- OCR + 문서이해(Transformer) + 로컬 LLM 추론을 결합한 경량 파이프라인 경험
