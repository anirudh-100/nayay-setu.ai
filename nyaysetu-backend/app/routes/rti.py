"""/draft/rti — draft a Right to Information application from a plain-language request.

The citizen describes what they want to know; the engine returns a ready-to-file RTI
application plus the procedural scaffolding (fee, where to file, time limits, appeals).
Legal facts come from app.knowledge.rti (hand-authored from the official Act); the LLM
only frames the questions, so errors map to a friendly message rather than a wrong draft.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.rti import RTIDraftRequest, RTIDraftResponse
from app.services.llm_service import LLMError
from app.services.rti_service import RTIService
from app.utils.logger import get_logger

router = APIRouter(tags=["rti"])
logger = get_logger(__name__)


def get_rti_service() -> RTIService:
    return RTIService()


@router.post("/draft/rti", response_model=RTIDraftResponse)
def draft_rti(
    request: RTIDraftRequest, service: RTIService = Depends(get_rti_service)
) -> RTIDraftResponse:
    try:
        return service.draft(
            subject=request.subject,
            public_authority=request.public_authority,
            level=request.level,
            applicant_name=request.applicant_name,
            applicant_address=request.applicant_address,
            is_bpl=request.is_bpl,
            language=request.language,
        )
    except LLMError as e:
        # The model only frames questions; if it's down we'd rather not silently degrade.
        logger.error("LLM failure during RTI drafting: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The drafting assistant is temporarily unavailable. Please try again.",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error drafting RTI")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error."
        ) from e
