"""
ats_gap_analyzer.py
====================
Resume ATS Gap Analyzer
------------------------
A production-grade Python module that evaluates a resume against a job description
using:
  1. PDF text extraction            (PyPDF2)
  2. Semantic compatibility scoring (sentence-transformers)
  3. Named-entity recognition       (HuggingFace transformers)
  4. Set-difference gap analysis    (Python sets)

Author  : Resume ATS Gap Analyzer Project
Version : 1.0.0
"""

from __future__ import annotations

import io
import json
import logging
import re
import string
import sys
# Workaround: PyTorch dev build in Python 3.12 has a torchvision NMS kernel mismatch.
# Since ATSGapAnalyzer only processes text (NLP), masking torchvision prevents transformers/SentenceTransformer
# from crashing during internal optional image-utility imports.
sys.modules.setdefault("torchvision", None)
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ATSGapAnalyzer")


# ─── Constants ────────────────────────────────────────────────────────────────

# Sentence-transformer model for semantic similarity
SEMANTIC_MODEL_ID: str = "0xnbk/nbk-ats-semantic-v1-en"

# HuggingFace token-classification model for NER
NER_MODEL_ID: str = "yashpwr/resume-ner-bert-v2"

# Entity labels to extract and their canonical category keys
ENTITY_LABEL_MAP: Dict[str, str] = {
    "SKILL": "skills",
    "SKILLS": "skills",
    "DESIGNATION": "designations",
    "TITLE": "designations",
    "JOB_TITLE": "designations",
    "DEGREE": "degrees",
    "EDUCATION": "degrees",
    "QUALIFICATION": "degrees",
}

# Fallback category for unlisted entity types
OTHER_CATEGORY: str = "other"

# Minimum NER confidence score to consider an entity valid
MIN_NER_SCORE: float = 0.50

# Maximum number of words an entity may have before we try to split it.
# The NER model sometimes labels multi-sentence chunks as one entity.
MAX_ENTITY_WORDS: int = 6

# Regex pattern used to split overly-long entity spans into sub-tokens.
# Splits at: comma, semicolon, bullet (• – | /), newline, or tab.
_ENTITY_SPLIT_RE = re.compile(r"[,;|/\n\t]|\s*[\u2022\u2013\u2014]\s*")

# ─── Helper Utilities ─────────────────────────────────────────────────────────


def _clean_text(raw: str) -> str:
    """
    Normalize raw extracted text.

    Steps:
        - Replace non-breaking spaces and exotic whitespace characters with regular spaces.
        - Collapse multiple consecutive whitespace characters into a single space.
        - Strip leading/trailing whitespace.

    Args:
        raw: The raw string to clean.

    Returns:
        A cleaned, single-block string.
    """
    # Normalize Unicode whitespace and line-break variants
    text = raw.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Remove soft hyphens (U+00AD) used for line-break hints in PDFs
    text = text.replace("\xad", "")
    # Collapse runs of whitespace (but preserve paragraph newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_entity(entity_word: str) -> str:
    """
    Normalise a single extracted entity string.

    Lowercases, strips surrounding punctuation/whitespace,
    and collapses internal whitespace.

    Args:
        entity_word: Raw entity word from NER output.

    Returns:
        Cleaned, lowercase entity string, or empty string if invalid.
    """
    cleaned = entity_word.lower().strip()
    # Strip leading/trailing punctuation
    cleaned = cleaned.strip(string.punctuation + " ")
    # Collapse internal whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _split_and_filter_entity(raw_word: str) -> List[str]:
    """
    Split an oversized NER entity span and return only valid sub-tokens.

    When the NER model merges a long run of same-label tokens (e.g. the entire
    SKILLS bullet list as one ``SKILL`` entity), this function:

    1. Splits the span at common delimiters (comma, semicolon, bullet, pipe, slash).
    2. Cleans each fragment via :func:`_clean_entity`.
    3. Discards fragments that are empty, too short (< 2 chars), or exceed
       ``MAX_ENTITY_WORDS`` words (likely still noise).

    If the original span is short enough (<= ``MAX_ENTITY_WORDS`` words) it is
    returned as-is (no splitting) so well-formed multi-word entities like
    "machine learning" are preserved.

    Args:
        raw_word: Raw entity string from the NER pipeline.

    Returns:
        List of clean, valid entity strings extracted from the span.
    """
    cleaned = _clean_entity(raw_word)
    if not cleaned:
        return []

    word_count = len(cleaned.split())

    # Short entity — return directly without splitting
    if word_count <= MAX_ENTITY_WORDS:
        return [cleaned] if len(cleaned) >= 2 else []

    # Long entity — try splitting at delimiters
    fragments: List[str] = []
    for fragment in _ENTITY_SPLIT_RE.split(cleaned):
        frag = _clean_entity(fragment)
        if not frag or len(frag) < 2:
            continue
        frag_words = len(frag.split())
        if frag_words > MAX_ENTITY_WORDS:
            # Still too long after splitting — discard as noise
            continue
        fragments.append(frag)

    return fragments


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D NumPy vectors.

    Formula:
        cos(θ) = (u · v) / (‖u‖ * ‖v‖)

    Args:
        u: First embedding vector.
        v: Second embedding vector.

    Returns:
        Cosine similarity score in [-1.0, 1.0].

    Raises:
        ValueError: If either vector has zero norm.
    """
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError(
            "Cannot compute cosine similarity: one or both input vectors have zero norm."
        )
    return float(np.dot(u, v) / (norm_u * norm_v))


# ─── Semantic Model Fallback Wrapper ─────────────────────────────────────────


class _AutoModelEncoder:
    """
    Fallback encoder used when ``SentenceTransformer`` cannot load a model due to
    missing pooling config (e.g. ``embedding_dimension`` not present in older models).

    Loads the model via ``AutoTokenizer`` + ``AutoModel`` and computes
    attention-mask-weighted mean pooling over the last hidden state,
    which is functionally equivalent to the standard sentence-transformers pipeline.

    Args:
        model_id: HuggingFace model identifier.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._device = device
        self._torch = torch
        logger.info("Fallback: loading tokenizer from %s ...", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Fallback: loading AutoModel from %s ...", model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        logger.info("Fallback AutoModel encoder ready (device=%s).", device)

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of texts into mean-pooled embeddings.

        Args:
            texts: List of plain-text strings to encode.
            convert_to_numpy: Ignored (always returns ndarray for compatibility).

        Returns:
            2-D NumPy array of shape ``(len(texts), hidden_size)``.
        """
        import torch  # noqa: PLC0415

        all_embeddings: List[np.ndarray] = []
        for text in texts:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            # Attention-mask weighted mean pooling
            token_emb = outputs.last_hidden_state  # (1, seq_len, hidden)
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
            summed = torch.sum(token_emb * mask, dim=1)  # (1, hidden)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # (1, 1)
            pooled = (summed / counts).squeeze(0)  # (hidden,)
            all_embeddings.append(pooled.cpu().numpy())
        return np.vstack(all_embeddings)


class ATSGapAnalyzer:
    """
    Resume ATS Gap Analyzer.

    Combines PDF extraction, semantic similarity scoring, and NER-based
    entity gap analysis to produce a structured JSON report comparing a
    resume against a job description.

    Attributes:
        semantic_model: Loaded SentenceTransformer model instance.
        ner_pipeline: HuggingFace token-classification pipeline instance.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialise the ATSGapAnalyzer.

        Models are loaded lazily on first use to avoid import-time penalties.

        Args:
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
                    Defaults to ``"cuda"`` if available, else ``"cpu"``.
        """
        self._device: str = device or self._detect_device()
        self._semantic_model = None
        self._ner_pipeline = None
        logger.info("ATSGapAnalyzer initialised (device=%s).", self._device)

    # ── Private: Device Detection ──────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        """
        Auto-detect the best available torch device.

        Returns:
            ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
        """
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ── Private: Lazy Model Loaders ────────────────────────────────────────────

    def _get_semantic_model(self):
        """
        Lazily load and cache the semantic embedding model.

        Strategy:
            1. Try loading via ``SentenceTransformer`` (preferred — uses the
               model's own pooling config when available).
            2. If that fails (e.g. the model predates the ``embedding_dimension``
               requirement introduced in sentence-transformers v6), fall back to
               a ``_AutoModelEncoder`` that does attention-weighted mean pooling
               using ``AutoModel`` + ``AutoTokenizer`` directly.

        Returns:
            An object with an ``encode(texts, ...) -> np.ndarray`` interface.

        Raises:
            RuntimeError: If both loading strategies fail.
        """
        if self._semantic_model is None:
            logger.info("Loading semantic model: %s ...", SEMANTIC_MODEL_ID)
            # -- Strategy 1: SentenceTransformer -----------------------------------
            st_error: Optional[Exception] = None
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                self._semantic_model = SentenceTransformer(
                    SEMANTIC_MODEL_ID, device=self._device
                )
                logger.info("Semantic model loaded via SentenceTransformer.")
            except Exception as exc:
                st_error = exc
                logger.warning(
                    "SentenceTransformer loading failed (%s). "
                    "Falling back to AutoModel mean-pooling encoder.",
                    exc,
                )
            # -- Strategy 2: AutoModel fallback ------------------------------------
            if self._semantic_model is None:
                try:
                    self._semantic_model = _AutoModelEncoder(
                        SEMANTIC_MODEL_ID, device=self._device
                    )
                    logger.info("Semantic model loaded via AutoModel fallback encoder.")
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Failed to load semantic model '{SEMANTIC_MODEL_ID}' "
                        f"via both SentenceTransformer ({st_error}) "
                        f"and AutoModel fallback ({fallback_exc})."
                    ) from fallback_exc
        return self._semantic_model

    def _get_ner_pipeline(self):
        """
        Lazily load and cache the HuggingFace NER pipeline.

        Returns:
            A loaded HuggingFace ``pipeline`` instance for token classification.

        Raises:
            RuntimeError: If the pipeline cannot be initialised.
        """
        if self._ner_pipeline is None:
            try:
                from transformers import pipeline  # noqa: PLC0415

                logger.info("Loading NER model: %s …", NER_MODEL_ID)
                self._ner_pipeline = pipeline(
                    task="token-classification",
                    model=NER_MODEL_ID,
                    aggregation_strategy="simple",
                    device=0 if self._device == "cuda" else -1,
                )
                logger.info("NER pipeline loaded successfully.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load NER pipeline '{NER_MODEL_ID}': {exc}"
                ) from exc
        return self._ner_pipeline

    # ── Public: PDF Text Extraction ────────────────────────────────────────────

    def extract_text_from_pdf(
        self, pdf_input: Union[str, Path, io.BytesIO]
    ) -> str:
        """
        Extract and clean text from a PDF file.

        Args:
            pdf_input: One of:
                - A file system path string or ``pathlib.Path`` object.
                - A ``io.BytesIO`` binary stream containing PDF bytes.

        Returns:
            Cleaned plain-text content of the PDF.

        Raises:
            FileNotFoundError: If a path is given but the file does not exist.
            ValueError: If ``pdf_input`` is an unsupported type.
            RuntimeError: If PDF parsing fails or the PDF appears to be encrypted/corrupted.
        """
        import PyPDF2  # noqa: PLC0415

        # Resolve file handle
        if isinstance(pdf_input, (str, Path)):
            path = Path(pdf_input)
            if not path.is_file():
                raise FileNotFoundError(f"PDF file not found: {path}")
            file_handle = open(path, "rb")
            close_after = True
        elif isinstance(pdf_input, io.BytesIO):
            file_handle = pdf_input
            close_after = False
        else:
            raise ValueError(
                f"Unsupported pdf_input type: {type(pdf_input).__name__}. "
                "Expected str, Path, or io.BytesIO."
            )

        pages_text: List[str] = []
        try:
            reader = PyPDF2.PdfReader(file_handle)
            if reader.is_encrypted:
                raise RuntimeError(
                    "The provided PDF is encrypted and cannot be read without a password."
                )
            num_pages = len(reader.pages)
            logger.info("PDF has %d page(s). Extracting text …", num_pages)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    raw_page_text = page.extract_text() or ""
                    pages_text.append(raw_page_text)
                except Exception as page_exc:
                    logger.warning(
                        "Could not extract text from page %d: %s", page_num, page_exc
                    )
        except PyPDF2.errors.PdfReadError as pdf_err:
            raise RuntimeError(
                f"PDF parsing failed (corrupted or unsupported format): {pdf_err}"
            ) from pdf_err
        finally:
            if close_after:
                file_handle.close()

        full_text = "\n".join(pages_text)
        cleaned = _clean_text(full_text)

        if not cleaned:
            logger.warning(
                "PDF text extraction returned empty content. "
                "The PDF may be image-based (scanned) and require OCR."
            )
        return cleaned

    # ── Public: Semantic Similarity ────────────────────────────────────────────

    def compute_semantic_score(
        self, resume_text: str, jd_text: str
    ) -> Tuple[float, str]:
        """
        Compute the semantic ATS compatibility score between a resume and JD.

        Uses ``0xnbk/nbk-ats-semantic-v1-en`` to generate 768-dimensional
        embeddings and calculates cosine similarity.

        Args:
            resume_text: Plain-text content of the resume.
            jd_text: Plain-text content of the job description.

        Returns:
            A tuple of:
                - ``overall_match_score`` (float): Raw cosine similarity in [0.0, 1.0].
                - ``compatibility_percentage`` (str): Human-readable percentage string.

        Raises:
            ValueError: If either input text is empty.
            RuntimeError: If the model fails to generate embeddings.
        """
        if not resume_text.strip():
            raise ValueError("resume_text cannot be empty.")
        if not jd_text.strip():
            raise ValueError("jd_text cannot be empty.")

        model = self._get_semantic_model()
        try:
            logger.info("Generating semantic embeddings …")
            embeddings = model.encode(
                [resume_text, jd_text],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            resume_vec: np.ndarray = embeddings[0]
            jd_vec: np.ndarray = embeddings[1]

            raw_similarity = _cosine_similarity(resume_vec, jd_vec)
            # Clamp to [0, 1] — cosine similarity can be slightly negative
            # for very dissimilar texts; ATS tools conventionally floor at 0
            score = max(0.0, min(1.0, raw_similarity))
            percentage = f"{score * 100:.1f}%"
            logger.info("Semantic compatibility score: %s", percentage)
            return score, percentage
        except Exception as exc:
            raise RuntimeError(
                f"Semantic similarity computation failed: {exc}"
            ) from exc

    # ── Public: NER Entity Extraction ─────────────────────────────────────────

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured named entities from text using the NER pipeline.

        Runs ``yashpwr/resume-ner-bert-v2`` with ``aggregation_strategy="simple"``
        and organises results into category buckets (skills, designations, degrees).

        Args:
            text: Plain-text string to run NER on.

        Returns:
            A dictionary mapping category names to deduplicated, lowercased entity lists:
            ``{"skills": [...], "designations": [...], "degrees": [...], "other": [...]}``.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If the NER pipeline fails.
        """
        if not text.strip():
            raise ValueError("Input text for entity extraction cannot be empty.")

        ner = self._get_ner_pipeline()
        categories: Dict[str, Set[str]] = {
            "skills": set(),
            "designations": set(),
            "degrees": set(),
            OTHER_CATEGORY: set(),
        }

        try:
            logger.info("Running NER pipeline on text (%d chars) …", len(text))
            # Split into chunks if text exceeds typical model token limits (~512 tokens)
            chunks = self._chunk_text(text)
            all_entities: List[Dict[str, Any]] = []
            for chunk in chunks:
                chunk_entities = ner(chunk)
                all_entities.extend(chunk_entities)

            logger.info("NER returned %d raw entities.", len(all_entities))

            for entity in all_entities:
                score: float = entity.get("score", 0.0)
                if score < MIN_NER_SCORE:
                    continue
                label: str = entity.get("entity_group", "").upper()
                word: str = entity.get("word", "")
                category = ENTITY_LABEL_MAP.get(label, OTHER_CATEGORY)

                # Split oversized spans and filter noise before adding to set
                for token in _split_and_filter_entity(word):
                    categories[category].add(token)

        except Exception as exc:
            raise RuntimeError(f"NER entity extraction failed: {exc}") from exc

        # Convert sets to sorted lists for deterministic JSON serialization
        return {cat: sorted(entities) for cat, entities in categories.items()}

    # ── Public: Gap Analysis ───────────────────────────────────────────────────

    def compute_gaps(
        self,
        resume_entities: Dict[str, List[str]],
        jd_entities: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Compute entity gaps between resume and job description using set difference.

        For each category, computes:
            Missing = E_JD \\ E_Resume

        Args:
            resume_entities: Categorised entities extracted from the resume.
            jd_entities: Categorised entities extracted from the job description.

        Returns:
            A dictionary of missing items per category:
            ``{"missing_skills": [...], "missing_designations": [...],
               "missing_degrees": [...], "missing_other": [...]}``.
        """
        # Collect all category keys from both dicts
        all_categories: Set[str] = set(resume_entities.keys()) | set(
            jd_entities.keys()
        )

        gaps: Dict[str, List[str]] = {}
        for category in sorted(all_categories):
            resume_set: Set[str] = set(resume_entities.get(category, []))
            jd_set: Set[str] = set(jd_entities.get(category, []))
            missing: Set[str] = jd_set - resume_set
            gap_key = f"missing_{category}"
            gaps[gap_key] = sorted(missing)

        total_missing = sum(len(v) for v in gaps.values())
        logger.info("Gap analysis complete: %d total missing items.", total_missing)
        return gaps

    # ── Public: Full Analysis Pipeline ────────────────────────────────────────

    def analyze(
        self,
        resume_input: Union[str, Path, io.BytesIO],
        jd_text: str,
        resume_is_pdf: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full end-to-end ATS gap analysis pipeline.

        Args:
            resume_input: Either:
                - A plain-text string of the resume (when ``resume_is_pdf=False``).
                - A PDF file path, ``pathlib.Path``, or ``io.BytesIO`` stream
                  (when ``resume_is_pdf=True``).
            jd_text: Plain-text job description string.
            resume_is_pdf: Set to ``True`` to trigger PDF text extraction.

        Returns:
            A structured dictionary with the following keys:
                - ``overall_match_score`` (float): Cosine similarity in [0.0, 1.0].
                - ``compatibility_percentage`` (str): Human-readable percentage.
                - ``resume_entities`` (dict): Categorised entities from resume.
                - ``jd_entities`` (dict): Categorised entities from JD.
                - ``missing_gaps`` (dict): Missing items per category.

        Raises:
            ValueError: If required inputs are empty or invalid.
            RuntimeError: On model loading, PDF parsing, or inference failures.
        """
        logger.info("═" * 60)
        logger.info("Starting ATS gap analysis pipeline …")
        logger.info("═" * 60)

        # Step 1 – Get resume text
        if resume_is_pdf:
            logger.info("Step 1/4 → Extracting text from resume PDF …")
            resume_text = self.extract_text_from_pdf(resume_input)  # type: ignore[arg-type]
        else:
            if not isinstance(resume_input, str):
                raise ValueError(
                    "When resume_is_pdf=False, resume_input must be a plain-text string."
                )
            resume_text = _clean_text(resume_input)

        if not resume_text:
            raise ValueError(
                "Resume text is empty after extraction. "
                "Verify the input PDF contains readable text (not a scanned image)."
            )

        jd_text_cleaned = _clean_text(jd_text)
        if not jd_text_cleaned:
            raise ValueError("jd_text cannot be empty.")

        # Step 2 – Semantic similarity
        logger.info("Step 2/4 → Computing semantic compatibility score …")
        overall_score, compat_percentage = self.compute_semantic_score(
            resume_text, jd_text_cleaned
        )

        # Step 3 – NER entity extraction
        logger.info("Step 3/4 → Extracting entities from resume …")
        resume_entities = self.extract_entities(resume_text)

        logger.info("Step 3/4 → Extracting entities from job description …")
        jd_entities = self.extract_entities(jd_text_cleaned)

        # Step 4 – Gap analysis
        logger.info("Step 4/4 → Computing entity gaps …")
        missing_gaps = self.compute_gaps(resume_entities, jd_entities)

        result: Dict[str, Any] = {
            "overall_match_score": round(overall_score, 4),
            "compatibility_percentage": compat_percentage,
            "resume_entities": resume_entities,
            "jd_entities": jd_entities,
            "missing_gaps": missing_gaps,
        }

        logger.info("═" * 60)
        logger.info("Analysis complete. Score: %s", compat_percentage)
        logger.info("═" * 60)
        return result

    # ── Private: Text Chunking ─────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1_800) -> List[str]:
        """
        Split long text into manageable chunks for NER processing.

        Splits on sentence boundaries where possible to avoid truncating entities.

        Args:
            text: Input text to chunk.
            max_chars: Maximum characters per chunk. Defaults to 1800
                       (conservative for 512-token BERT models).

        Returns:
            List of text chunks.
        """
        if len(text) <= max_chars:
            return [text]

        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_len = len(sentence)
            else:
                current_chunk.append(sentence)
                current_len += len(sentence) + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


# ─── Mock PDF Builder (for demo purposes) ─────────────────────────────────────


def _build_mock_pdf(text: str) -> io.BytesIO:
    """
    Create a minimal, valid PDF containing ``text`` and return it as a BytesIO stream.

    This is used in the ``main()`` demo to simulate PDF input without requiring
    a real PDF file on disk.

    Args:
        text: Plain-text content to embed in the PDF.

    Returns:
        A ``BytesIO`` stream of a valid, single-page PDF.
    """
    import PyPDF2  # noqa: PLC0415
    from PyPDF2 import PdfWriter  # noqa: PLC0415

    # Build PDF bytes manually using a minimal PDF structure
    # because PyPDF2 is a reader-only library (no write support for blank pages
    # with text). We create a raw minimal PDF structure for demo purposes.
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    # Truncate to avoid PDF size issues in demo
    safe_text = safe_text[:1500]

    # Minimal single-page PDF with embedded text (PDF 1.4 syntax)
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n"
    )
    stream_content = (
        b"BT\n/F1 10 Tf\n50 750 Td\n(" + safe_text.encode("latin-1", errors="replace") + b") Tj\nET"
    )
    stream_len = len(stream_content)
    pdf_bytes += (
        f"<< /Length {stream_len} >>\nstream\n".encode()
        + stream_content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000400 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n500\n%%EOF"
    )
    return io.BytesIO(pdf_bytes)


# ─── Main Demo ────────────────────────────────────────────────────────────────


def main() -> None:
    """
    End-to-end demonstration of the ATSGapAnalyzer.

    Uses sample resume and job description text strings to illustrate:
        1.  PDF text extraction via a mock in-memory BytesIO PDF.
        2.  Semantic ATS compatibility scoring.
        3.  NER-based entity extraction.
        4.  Set-difference gap analysis.
        5.  Structured JSON output.
    """
    # ── Sample Inputs ──────────────────────────────────────────────────────────
    SAMPLE_RESUME_TEXT = """
    John Doe
    Senior Data Scientist
    john.doe@email.com | linkedin.com/in/johndoe | github.com/johndoe

    EDUCATION
    Bachelor of Science in Computer Science – MIT, 2018

    EXPERIENCE
    Data Scientist – TechCorp Inc. (2020–2024)
    • Built NLP pipelines using Python, PyTorch, and Hugging Face Transformers.
    • Designed and deployed machine learning models using scikit-learn and XGBoost.
    • Created data visualisations with Matplotlib and Seaborn.
    • Managed data workflows using Apache Spark and SQL.
    • Utilised AWS SageMaker for model training and deployment.

    Software Engineer – StartupXYZ (2018–2020)
    • Developed REST APIs using FastAPI and Flask.
    • Built CI/CD pipelines with GitHub Actions and Docker.
    • Worked with PostgreSQL and Redis for data storage.

    SKILLS
    Python, PyTorch, TensorFlow, scikit-learn, XGBoost, NLP, Hugging Face, FastAPI,
    Flask, Docker, PostgreSQL, SQL, Apache Spark, AWS SageMaker, Git, Matplotlib,
    Seaborn, REST APIs, Machine Learning, Deep Learning
    """

    SAMPLE_JD_TEXT = """
    Job Title: Senior Machine Learning Engineer

    Company: AI Innovations Ltd.

    About the Role:
    We are looking for a talented Senior Machine Learning Engineer to join our
    growing AI team. You will be responsible for designing, building, and deploying
    cutting-edge ML systems at scale.

    Requirements:
    • Master of Science in Computer Science, Data Science, or a related field.
    • 5+ years of experience as a Machine Learning Engineer or Applied Scientist.
    • Proficiency in Python, PyTorch, and TensorFlow.
    • Strong experience with Kubernetes and Docker for container orchestration.
    • Hands-on expertise in MLflow, Kubeflow, or similar MLOps platforms.
    • Experience with Kafka for real-time data streaming.
    • Proficiency with Terraform or Ansible for infrastructure-as-code.
    • Solid understanding of LLMs (Large Language Models) and Prompt Engineering.
    • Experience with Vector Databases (e.g., Pinecone, Weaviate, Chroma).
    • Knowledge of distributed computing with Apache Spark and Dask.
    • Strong SQL skills and experience with dbt for data transformation.

    Preferred Qualifications:
    • Kubernetes Certified Application Developer (CKAD) certification.
    • Experience with Ray for distributed ML training.
    • Familiarity with Rust or Go for performance-critical components.
    """

    # Force UTF-8 output so any remaining Unicode chars print cleanly on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 70)
    print("  RESUME ATS GAP ANALYZER - End-to-End Demo")
    print("=" * 70 + "\n")

    # ── Initialise Analyzer ────────────────────────────────────────────────────
    analyzer = ATSGapAnalyzer()

    # ── Demo 1: Plain-Text Resume Input ───────────────────────────────────────
    print("--- MODE: Plain-text resume input ---\n")
    try:
        result = analyzer.analyze(
            resume_input=SAMPLE_RESUME_TEXT,
            jd_text=SAMPLE_JD_TEXT,
            resume_is_pdf=False,
        )
        print("\n[OK] Analysis Complete - Structured JSON Output:\n")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        logger.error("Analysis failed (plain-text mode): %s", exc, exc_info=True)
        sys.exit(1)

    # ── Demo 2: Mock PDF Resume Input ─────────────────────────────────────────
    print("\n\n--- MODE: Mock PDF BytesIO stream input ---\n")
    try:
        mock_pdf_stream = _build_mock_pdf(SAMPLE_RESUME_TEXT)
        result_pdf = analyzer.analyze(
            resume_input=mock_pdf_stream,
            jd_text=SAMPLE_JD_TEXT,
            resume_is_pdf=True,
        )
        print("\n[OK] PDF-mode Analysis Complete - Structured JSON Output:\n")
        print(json.dumps(result_pdf, indent=2))
    except Exception as exc:
        logger.error("Analysis failed (PDF mode): %s", exc, exc_info=True)
        print(
            "\n[WARN] PDF mode demo failed (this is expected if the mock PDF's xref "
            "offsets do not align). The plain-text result above is the primary demo."
        )

    print("\n" + "=" * 70)
    print("  Demo Complete.")
    print("=" * 70 + "\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
