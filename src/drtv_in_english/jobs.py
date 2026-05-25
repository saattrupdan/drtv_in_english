"""In-process job registry.

Each ``POST /prepare`` call creates a :class:`Job` keyed by a random
``job_id``. The job holds the HLS proxy registry and a growing list of
translated cues that NDJSON subscribers stream out as they appear.

Jobs live for the lifetime of the process — they are not persisted and
not garbage-collected. The expected usage pattern is one watcher per
backend instance.
"""

import secrets
import threading
from dataclasses import dataclass, field

from .data_models import Chunk, CueEvent
from .hls_proxy import HlsRegistry


@dataclass
class Job:
    """Per-watch state shared between the HLS proxy and translation stream."""

    job_id: str
    title: str
    hls_master_url: str
    registry: HlsRegistry
    chunks: list[Chunk]
    cues: list[CueEvent] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)

    def append_cues(self, new_cues: list[CueEvent]) -> None:
        """Append ``new_cues`` and wake any subscribers."""
        with self.condition:
            self.cues.extend(new_cues)
            self.condition.notify_all()

    def mark_done(self, error: str | None = None) -> None:
        """Mark the translation finished (optionally with an error)."""
        with self.condition:
            self.done = True
            self.error = error
            self.condition.notify_all()


class JobRegistry:
    """Thread-safe map of ``job_id`` to :class:`Job`."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        title: str,
        hls_master_url: str,
        registry: HlsRegistry,
        chunks: list[Chunk],
    ) -> Job:
        """Insert a new job with a freshly-generated ``job_id``.

        Returns:
            The created :class:`Job`.
        """
        job_id = secrets.token_urlsafe(12)
        job = Job(
            job_id=job_id,
            title=title,
            hls_master_url=hls_master_url,
            registry=registry,
            chunks=chunks,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job for ``job_id`` or None.

        Returns:
            The job if present, else None.
        """
        with self._lock:
            return self._jobs.get(job_id)
