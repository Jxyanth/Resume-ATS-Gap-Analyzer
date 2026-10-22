# Resume ATS Gap Analyzer

A production-grade Python tool that evaluates a **resume** against a **job description (JD)** using:

- 📄 **PDF Text Extraction** (`PyPDF2`)
- 🧠 **Semantic Compatibility Scoring** (`sentence-transformers`)
- 🏷️ **Named Entity Recognition (NER)** (`HuggingFace transformers`)
- 🔍 **Set-Difference Gap Analysis**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ATSGapAnalyzer                                │
│                                                                     │
│  PDF / Text Input ──► Text Extraction & Normalisation               │
│                              │                                      │
│              ┌───────────────┴──────────────┐                       │
│              ▼                              ▼                       │
│   Semantic Similarity Engine       NER Entity Extraction            │
│   (0xnbk/nbk-ats-semantic-v1-en)  (yashpwr/resume-ner-bert-v2)    │
│              │                              │                       │
│              ▼                              ▼                       │
│   ATS Compatibility Score      E_Resume / E_JD Entity Sets         │
│              │                              │                       │
│              └───────────────┬──────────────┘                       │
│                              ▼                                      │
│                 Set Difference: E_JD \ E_Resume                     │
│                              │                                      │
│                              ▼                                      │
│                    Structured JSON Output                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Models Used

| Role | Model ID |
|------|----------|
| Semantic Similarity | [`0xnbk/nbk-ats-semantic-v1-en`](https://huggingface.co/0xnbk/nbk-ats-semantic-v1-en) |
| Named Entity Recognition | [`yashpwr/resume-ner-bert-v2`](https://huggingface.co/yashpwr/resume-ner-bert-v2) |

---

## Installation

### 1. Clone / Navigate to the project directory

```bash
cd "Resume ATS Checker"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will automatically download the HuggingFace models (~1–2 GB total).
> Models are cached locally at `~/.cache/huggingface/hub/`.

---

## Usage

### Run the built-in end-to-end demo

```bash
python ats_gap_analyzer.py
```

This runs two demos:
1. **Plain-text resume** analysis
2. **Mock PDF BytesIO stream** analysis

### Use as a library module

```python
from ats_gap_analyzer import ATSGapAnalyzer

analyzer = ATSGapAnalyzer()

# ── Option A: Plain-text resume ──
result = analyzer.analyze(
    resume_input="John Doe\nPython Developer with 5 years experience ...",
    jd_text="We need a Senior Python Engineer with Docker and Kubernetes ...",
    resume_is_pdf=False,
)

# ── Option B: PDF file path ──
result = analyzer.analyze(
    resume_input="path/to/resume.pdf",
    jd_text="We need a Senior Python Engineer with Docker and Kubernetes ...",
    resume_is_pdf=True,
)

# ── Option C: PDF as BytesIO stream ──
import io
with open("resume.pdf", "rb") as f:
    pdf_stream = io.BytesIO(f.read())

result = analyzer.analyze(
    resume_input=pdf_stream,
    jd_text="We need a Senior Python Engineer with Docker and Kubernetes ...",
    resume_is_pdf=True,
)

import json
print(json.dumps(result, indent=2))
```

---

## JSON Output Schema

```json
{
  "overall_match_score": 0.7821,
  "compatibility_percentage": "78.2%",
  "resume_entities": {
    "skills": ["docker", "fastapi", "nlp", "python", "pytorch"],
    "designations": ["data scientist", "software engineer"],
    "degrees": ["bachelor of science in computer science"],
    "other": []
  },
  "jd_entities": {
    "skills": ["docker", "kafka", "kubernetes", "mlflow", "nlp", "python", "pytorch"],
    "designations": ["machine learning engineer", "senior machine learning engineer"],
    "degrees": ["master of science"],
    "other": []
  },
  "missing_gaps": {
    "missing_skills": ["kafka", "kubernetes", "mlflow"],
    "missing_designations": ["senior machine learning engineer"],
    "missing_degrees": ["master of science"],
    "missing_other": []
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `overall_match_score` | `float` | Cosine similarity score clamped to [0.0, 1.0] |
| `compatibility_percentage` | `str` | Human-readable ATS compatibility percentage |
| `resume_entities` | `dict` | Categorised entities extracted from the resume |
| `jd_entities` | `dict` | Categorised entities extracted from the job description |
| `missing_gaps` | `dict` | Set difference: items in JD but absent from resume |

---

## Component API Reference

### `ATSGapAnalyzer`

```python
class ATSGapAnalyzer:
    def __init__(self, device: Optional[str] = None) -> None: ...
    
    def extract_text_from_pdf(self, pdf_input: Union[str, Path, io.BytesIO]) -> str: ...
    def compute_semantic_score(self, resume_text: str, jd_text: str) -> Tuple[float, str]: ...
    def extract_entities(self, text: str) -> Dict[str, List[str]]: ...
    def compute_gaps(self, resume_entities: dict, jd_entities: dict) -> Dict[str, List[str]]: ...
    def analyze(self, resume_input, jd_text: str, resume_is_pdf: bool = False) -> Dict: ...
```

---

## Gap Analysis Math

For each entity category `c` (skills, designations, degrees):

$$\text{Missing}_c = E_{JD,c} \setminus E_{\text{Resume},c}$$

Where:
- $E_{\text{Resume},c}$ = set of entities in category `c` extracted from the resume
- $E_{JD,c}$ = set of entities in category `c` extracted from the JD

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: PDF is encrypted` | Password-protected PDF | Decrypt PDF before passing it |
| Empty NER results | Text too short or model mismatch | Ensure text > 20 characters; check NER model ID |
| `CUDA out of memory` | Large model on small GPU | Pass `device="cpu"` to `ATSGapAnalyzer()` |
| PDF returns empty text | Scanned/image PDF | Use an OCR tool (e.g., `pytesseract`) first |
| Slow first run | Models downloading from HuggingFace Hub | Wait for download; subsequent runs use cache |

---

## Project Structure

```
Resume ATS Checker/
├── ats_gap_analyzer.py   # Main module: ATSGapAnalyzer class + main() demo
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
