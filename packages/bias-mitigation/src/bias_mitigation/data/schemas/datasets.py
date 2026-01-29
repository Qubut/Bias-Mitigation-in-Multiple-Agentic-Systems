from typing import Optional

from sqlalchemy import JSON
from sqlmodel import Field, Relationship, SQLModel


class AdditionalMetadata(SQLModel):
    subcategory: str
    stereotyped_groups: list[str]
    version: str
    source: str


class BBQAnswer(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    index: int
    text: str
    tag: str
    bbq_id: int | None = Field(default=None, foreign_key='bbq.id')
    bbq: Optional['BBQ'] = Relationship(back_populates='answers')


class BBQ(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    example_id: int = Field(index=True, unique=True)
    question_index: str
    question_polarity: str
    context_condition: str
    category: str
    additional_metadata: AdditionalMetadata = Field(sa_type=JSON)
    context: str
    question: str
    ans0: str
    ans1: str
    ans2: str
    label: int
    answers: list[BBQAnswer] = Relationship(back_populates='bbq', cascade_delete=True)


class StereoSetLabel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    label: str
    human_id: str
    sentence_id: int | None = Field(default=None, foreign_key='stereosetsentence.id')
    sentence: Optional['StereoSetSentence'] = Relationship(back_populates='labels')


class StereoSetSentence(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    sentence: str
    sentence_id: str = Field(index=True, unique=True)
    gold_label: str
    stereoset_id: str | None = Field(default=None, foreign_key='stereoset.id')
    stereoset: Optional['StereoSet'] = Relationship(back_populates='sentences')
    labels: list[StereoSetLabel] = Relationship(back_populates='sentence', cascade_delete=True)


class StereoSet(SQLModel, table=True):
    id: str = Field(primary_key=True)
    target: str
    bias_type: str
    context: str
    type: str
    sentences: list[StereoSetSentence] = Relationship(
        back_populates='stereoset', cascade_delete=True
    )
