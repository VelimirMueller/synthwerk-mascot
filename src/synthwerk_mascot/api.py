"""`POST /v1/answers`: the mascot's one public route.

Retrieval only: the answer is the best passages of the public docs, each with its source.
No language model writes text, so every sentence in an answer is a quoted passage. The
question is not stored and not logged.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from synthwerk_mascot.embedder import Embedder, query_text
from synthwerk_mascot.index import VectorIndex
from synthwerk_mascot.platform.config import Settings
from synthwerk_mascot.ratelimit import RateLimiter

router = APIRouter(tags=["answers"])


class Question(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000)]


class Source(BaseModel):
    path: str = Field(description="File in the synthwerk repo, for example docs/roadmap.md")
    anchor: str = Field(description="Heading anchor in that file; empty for the top")
    url: str = Field(description="Link to the passage at the commit the index was built from")


class Passage(BaseModel):
    title: str = Field(description="Heading path of the passage")
    text: str = Field(description="The passage, quoted from the docs")
    score: float = Field(description="Cosine similarity to the question, 0 to 1")
    source: Source


class Answer(BaseModel):
    found: bool = Field(description="False: the docs hold nothing close enough to the question")
    passages: list[Passage]


def _client(request: Request) -> str:
    # uvicorn --proxy-headers sets this from X-Forwarded-For when the peer is a trusted proxy.
    return request.client.host if request.client else "unknown"


@router.post(
    "/v1/answers",
    responses={
        422: {"description": "The question is empty or too long."},
        429: {"description": "Rate limit. Retry-After says when to ask again."},
        503: {"description": "The model or the index is not loaded yet."},
    },
)
def answer(body: Question, request: Request, response: Response) -> Answer:
    settings: Settings = request.app.state.settings
    question = body.question.strip()
    if not question or len(question) > settings.max_question_chars:
        raise HTTPException(
            422, f"question must have 1 to {settings.max_question_chars} characters"
        )
    limiter: RateLimiter = request.app.state.limiter
    wait = limiter.acquire(_client(request))
    if wait is not None:
        raise HTTPException(429, "too many questions", headers={"Retry-After": str(wait)})
    embedder: Embedder | None = request.app.state.embedder
    index: VectorIndex | None = request.app.state.index
    if embedder is None or index is None:
        raise HTTPException(503, "not ready", headers={"Retry-After": "5"})

    query = embedder.embed([query_text(question)])[0]
    hits = index.search(query, settings.top_k, settings.min_score, settings.min_margin)
    response.headers["Cache-Control"] = "no-store"
    return Answer(
        found=bool(hits),
        passages=[
            Passage(
                title=hit.chunk.title,
                text=hit.chunk.text,
                score=round(hit.score, 3),
                source=Source(
                    path=hit.chunk.path, anchor=hit.chunk.anchor, url=index.url(hit.chunk)
                ),
            )
            for hit in hits
        ],
    )
