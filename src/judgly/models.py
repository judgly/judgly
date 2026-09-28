"""The request and response of a decision as pydantic models.

A question is one of three types, told apart by ``type``:

- ``Choice``: pick one of 2 to 26 named options.
- ``Binary``: true or false (``type="bool"``).
- ``Score``: a level from 1 to ``levels`` (2 to 9).

``format`` names the head used to calibrate the answer (for example ``"stance"``); without it,
or when the loaded pack has no head for it, the pack's fallback head ``"*"`` is used.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Choice", "Binary", "Score", "Question", "Request",
           "ChoiceAnswer", "BinaryAnswer", "ScoreAnswer", "Answer", "Decision"]


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: str = Field(description="The question, as the model reads it.")
    format: str | None = Field(default=None, description="Head to use; None for the fallback.")


class Choice(_Question):
    """Pick one of the options. Keys are what the answer reports; values are what the model
    reads. A list of strings is taken as options whose key and text are the same."""

    type: Literal["choice"] = "choice"
    options: dict[str, str] = Field(min_length=2, max_length=26)

    @field_validator("options", mode="before")
    @classmethod
    def _list_to_dict(cls, v: object) -> object:
        if isinstance(v, (list, tuple)):
            if len(set(v)) != len(v):
                raise ValueError("options must be distinct")
            return {str(o): str(o) for o in v}
        return v


class Binary(_Question):
    """True or false."""

    type: Literal["bool"] = "bool"


class Score(_Question):
    """A level from 1 to ``levels``."""

    type: Literal["score"] = "score"
    levels: int = Field(ge=2, le=9)


Question = Annotated[Union[Choice, Binary, Score], Field(discriminator="type")]


class Request(BaseModel):
    """One state and the questions asked about it: the JSON that ``Engine.decide_json`` takes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schema")
    state: str
    questions: dict[str, Question] = Field(min_length=1)

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True)


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_mass: float = Field(description="Probability the model put on the option letters at "
                                         "all, mean over rotations; near 1 is healthy.")
    rotation_spread: float = Field(description="Range of the top option's probability across "
                                               "the option orders asked; 0 with one order.")
    n_rotations: int
    format: str | None
    head: str | None = Field(description="SHA-256 of the head file used; None for raw (H0) "
                                         "probabilities.")
    head_format: str | None = Field(description="The heads entry used: the format or \"*\".")


class ChoiceAnswer(_Answer):
    type: Literal["choice"] = "choice"
    probs: dict[str, float]
    top: str


class BinaryAnswer(_Answer):
    type: Literal["bool"] = "bool"
    p_true: float
    top: bool


class ScoreAnswer(_Answer):
    type: Literal["score"] = "score"
    probs: list[float] = Field(description="Probability of level 1, 2, ...")
    mean: float
    top: int


Answer = Annotated[Union[ChoiceAnswer, BinaryAnswer, ScoreAnswer], Field(discriminator="type")]


class Tokens(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state: int
    questions: int


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state: float
    questions: float


class Decision(BaseModel):
    """The answers to one request, with what produced them."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schema")
    answers: dict[str, Answer]
    model_sha256: str
    template_sha256: str
    truncated: bool = Field(description="The state was cut to max_state_tokens.")
    tokens: Tokens
    timing_ms: Timing

    def __getitem__(self, question_id: str) -> ChoiceAnswer | BinaryAnswer | ScoreAnswer:
        return self.answers[question_id]
