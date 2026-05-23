"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="but_with_subs")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health.

    Returns:
        Mapping with a single ``status`` key set to ``"ok"``.
    """
    return {"status": "ok"}
