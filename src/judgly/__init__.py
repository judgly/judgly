"""judgly: calibrated, deterministic judgments from open language models.

A frozen open model reads a state once; each typed question (Choice, Binary, Score) branches
from that cached state, and the probabilities of the answer letters at one position become the
answer, optionally recalibrated by a small fitted head. No text is generated.
Early release (0.x).
"""

from importlib.metadata import version as _version

from judgly._native import native_version
from judgly.engine import Engine, EngineConfig, JudglyError
from judgly.models import (Answer, Binary, BinaryAnswer, Choice, ChoiceAnswer, Decision,
                           Question, Request, Score, ScoreAnswer)
from judgly.packs import HeadsNotAvailable, ModelMismatch, Pack, list_packs

__version__ = _version("judgly")

__all__ = ["Engine", "EngineConfig", "JudglyError", "Choice", "Binary", "Score", "Question",
           "Request", "Answer", "ChoiceAnswer", "BinaryAnswer", "ScoreAnswer", "Decision",
           "Pack", "list_packs", "HeadsNotAvailable", "ModelMismatch", "native_version",
           "__version__"]
