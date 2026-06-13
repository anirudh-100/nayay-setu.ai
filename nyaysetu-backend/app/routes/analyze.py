"""/analyze — document understanding endpoints.

Two ways in, one engine:
  - ``POST /analyze``       JSON body with pasted ``document_text``.
  - ``POST /analyze/file``  multipart upload (PDF / .txt / .md); text is extracted
                            server-side, then analysed identically.

Both return the same :class:`AnalyzeResponse`. Errors map to friendly HTTP codes so the
frontend can show a clear message rather than a stack trace.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.schemas.analyze import MAX_DOC_CHARS, MIN_DOC_CHARS, AnalyzeRequest, AnalyzeResponse
from app.services.document_service import DocumentService
from app.services.llm_service import LLMError
from app.utils.extract import ExtractionError, extract_text
from app.utils.logger import get_logger

router = APIRouter(tags=["analyze"])
logger = get_logger(__name__)

# Reject oversized uploads early (bytes). 10 MB comfortably covers a multi-page PDF.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def get_document_service() -> DocumentService:
    return DocumentService()


def _analyze(text: str, question: str | None, language: str, service: DocumentService) -> AnalyzeResponse:
    try:
        return service.analyze(text, question, language)
    except LLMError as e:
        logger.error("LLM failure during analysis: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The legal assistant is temporarily unavailable. Please try again.",
        ) from e
    except FileNotFoundError as e:
        logger.error("Index missing: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not initialized. Run scripts/build_index.py.",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error analysing document")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.") from e


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, service: DocumentService = Depends(get_document_service)) -> AnalyzeResponse:
    return _analyze(request.document_text, request.question, request.language, service)


@router.post("/analyze/file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    question: str | None = Form(default=None),
    language: str = Form(default="en"),
    service: DocumentService = Depends(get_document_service),
) -> AnalyzeResponse:
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large (max 10 MB).",
        )
    try:
        text = extract_text(file.filename or "", data)
    except ExtractionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    if len(text) < MIN_DOC_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document has too little readable text to analyse.",
        )
    return _analyze(text[:MAX_DOC_CHARS], question, language, service)
