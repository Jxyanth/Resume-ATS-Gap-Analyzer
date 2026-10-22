"""
app.py
======
FastAPI Web Application & UI Server for Resume ATS Gap Analyzer.
Serves a sleek, modern UI for analyzing resumes against Job Descriptions.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ats_gap_analyzer import ATSGapAnalyzer

logger = logging.getLogger("ATSApp")

app = FastAPI(
    title="Resume ATS Gap Analyzer",
    description="Analyze resumes against job descriptions with AI semantic matching and entity gap extraction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single ATS analyzer instance
analyzer = ATSGapAnalyzer()


@app.post("/api/analyze")
async def analyze_resume(
    resume_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    jd_text: str = Form(...),
):
    """
    Analyze resume against a job description.
    Accepts either an uploaded PDF file or pasted resume text.
    """
    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text is required.")

    if resume_file and resume_file.filename:
        # User uploaded a file
        contents = await resume_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = resume_file.filename.lower()
        if filename.endswith(".pdf"):
            try:
                pdf_stream = io.BytesIO(contents)
                result = analyzer.analyze(
                    resume_input=pdf_stream,
                    jd_text=jd_text,
                    resume_is_pdf=True,
                )
            except Exception as e:
                logger.error("Error analyzing PDF: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
        elif filename.endswith((".txt", ".md")):
            try:
                text_content = contents.decode("utf-8", errors="ignore")
                result = analyzer.analyze(
                    resume_input=text_content,
                    jd_text=jd_text,
                    resume_is_pdf=False,
                )
            except Exception as e:
                logger.error("Error analyzing text file: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to process text file: {str(e)}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a PDF or .txt file.",
            )
    elif resume_text and resume_text.strip():
        try:
            result = analyzer.analyze(
                resume_input=resume_text,
                jd_text=jd_text,
                resume_is_pdf=False,
            )
        except Exception as e:
            logger.error("Error analyzing resume text: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Please provide a resume by either uploading a PDF file or pasting the text.",
        )

    # Compute matched entities for better UI visibility
    resume_entities = result.get("resume_entities", {})
    jd_entities = result.get("jd_entities", {})
    missing_gaps = result.get("missing_gaps", {})

    matched_entities = {}
    for cat in ["skills", "designations", "degrees", "other"]:
        res_set = set(resume_entities.get(cat, []))
        jd_set = set(jd_entities.get(cat, []))
        matched_entities[f"matched_{cat}"] = sorted(list(res_set & jd_set))

    # Add matched_entities and summary statistics to response
    result["matched_entities"] = matched_entities
    
    total_jd_skills = len(jd_entities.get("skills", []))
    matched_skills = len(matched_entities.get("matched_skills", []))
    result["skill_match_rate"] = (
        round((matched_skills / total_jd_skills) * 100, 1) if total_jd_skills > 0 else None
    )

    return result


# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application UI."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Starting Resume ATS Checker Web UI at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
