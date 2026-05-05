"""In-memory Pydantic models for BBQ and StereoSet raw entries."""

from pydantic import BaseModel


class Answer(BaseModel):
    """Answer option metadata for one candidate response."""

    text: str
    tag: str  # "old", "nonOld", "unknown"


class AnswerInfo(BaseModel):
    """Grouped answer metadata payload for BBQ answer options."""

    ans0: Answer
    ans1: Answer
    ans2: Answer


class AdditionalMetadata(BaseModel):
    """Additional source metadata attached to normalized examples."""

    subcategory: str
    stereotyped_groups: list[str]
    version: str
    source: str


class BBQEntry(BaseModel):
    """Raw BBQ entry schema used during preprocessing and validation."""

    example_id: int
    question_index: str
    question_polarity: str
    context_condition: str
    category: str  # file-based category
    answer_info: AnswerInfo
    additional_metadata: AdditionalMetadata
    context: str
    question: str
    ans0: str
    ans1: str
    ans2: str
    label: int


class Label(BaseModel):
    """Human annotation label record for a StereoSet sentence."""

    label: str  # "stereotype", etc.
    human_id: str


class Sentence(BaseModel):
    """StereoSet sentence schema with labels and gold category."""

    sentence: str
    id: str
    labels: list[Label]
    gold_label: str


class StereoSetEntry(BaseModel):
    """Raw StereoSet entry schema used during dataset transformation."""

    id: str
    target: str
    bias_type: str
    context: str
    sentences: list[Sentence]
    type: str  # "intersentence" or "intrasentence" added
