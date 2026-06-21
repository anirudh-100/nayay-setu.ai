from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.ask import AskRequest, AskResponse
from app.services.llm_service import LLMError
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

router = APIRouter(tags=["ask"])
logger = get_logger(__name__)


def get_rag_service() -> RAGService:
    # Constructed per-request; the heavy singletons (embedder, retriever) live inside.
    return RAGService()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RAGService = Depends(get_rag_service)) -> AskResponse:
    try:
        return service.answer(request.query, request.language, request.history)
    except LLMError as e:
        logger.error("LLM failure: %s", e)
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
        logger.exception("Unexpected error answering query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from e
