
from __future__ import annotations


class PipelineSystemError(RuntimeError):

    def __init__(self, stage: str, message: str):
        self.stage = str(stage)
        super().__init__(message)

