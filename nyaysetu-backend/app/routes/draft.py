"""/draft — the generic drafting engine.

  - ``GET  /draft/journeys``  lists available document types + their input fields, so the
                              frontend builds its picker and forms with zero hardcoding.
  - ``POST /draft``           drafts a document for a chosen journey.

Legal facts come from the journey's hand-authored knowledge module; the LLM only frames
the citizen's situation, so errors map to friendly HTTP codes rather than a wrong draft.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.draft import DraftRequest, DraftResponse, JourneyInfo
from app.services.drafting_service import (
    DraftingService,
    JourneyNotFound,
    MissingFields,
    list_journeys,
)
from app.services.llm_service import LLMError
from app.utils.logger import get_logger

router = APIRouter(tags=["draft"])
logger = get_logger(__name__)


def get_drafting_service() -> DraftingService:
    return DraftingService()


@router.get("/draft/journeys", response_model=list[JourneyInfo])
def journeys() -> list[JourneyInfo]:
    return list_journeys()


@router.post("/draft", response_model=DraftResponse)
def draft(
    request: DraftRequest, service: DraftingService = Depends(get_drafting_service)
) -> DraftResponse:
    try:
        return service.draft(
            journey_id=request.journey,
            fields=request.fields,
            applicant_name=request.applicant_name,
            applicant_address=request.applicant_address,
            language=request.language,
        )
    except JourneyNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown document type: {e}"
        ) from e
    except MissingFields as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except LLMError as e:
        logger.error("LLM failure during drafting: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The drafting assistant is temporarily unavailable. Please try again.",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error drafting document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error."
        ) from e
