from pydantic import BaseModel


class Answer(BaseModel):
    text: str
    tag: str  # "old", "nonOld", "unknown"


class AnswerInfo(BaseModel):
    ans0: Answer
    ans1: Answer
    ans2: Answer


class AdditionalMetadata(BaseModel):
    subcategory: str
    stereotyped_groups: list[str]
    version: str
    source: str


class BBQEntry(BaseModel):
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
    label: str  # "stereotype", etc.
    human_id: str


class Sentence(BaseModel):
    sentence: str
    id: str
    labels: list[Label]
    gold_label: str


class StereoSetEntry(BaseModel):
    id: str
    target: str
    bias_type: str
    context: str
    sentences: list[Sentence]
    type: str  # "intersentence" or "intrasentence" added
