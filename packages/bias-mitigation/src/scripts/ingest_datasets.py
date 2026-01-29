import asyncio
import json
import os
from pathlib import Path

import dotenv
from loguru import logger
from returns.future import FutureResult, future_safe
from returns.result import Failure, Result, safe
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from bias_mitigation.data.schemas import (
    BBQ,
    AdditionalMetadata,
    BBQAnswer,
    StereoSet,
    StereoSetLabel,
    StereoSetSentence,
)

dotenv.load_dotenv()


@safe
def parse_bbq_file(file_path: Path, cat: str) -> list[BBQ]:
    entries = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data['category'] = cat

            # Build answers
            answers = []
            for i in range(3):
                ans_key = f'ans{i}'
                info_key = f'ans{i}'
                text = data[ans_key]
                tag = data['answer_info'][info_key][
                    1
                ]  # second item is tag ("old", "nonOld", "unknown")
                answers.append(BBQAnswer(index=i, text=text, tag=tag))

            # Additional metadata
            add_meta = AdditionalMetadata(**data['additional_metadata'])

            # Main entry (exclude raw answer fields and answer_info)
            bbq_data = {
                'example_id': data['example_id'],
                'question_index': data['question_index'],
                'question_polarity': data['question_polarity'],
                'context_condition': data['context_condition'],
                'category': data['category'],
                'additional_metadata': add_meta,
                'context': data['context'],
                'question': data['question'],
                'ans0': data['ans0'],
                'ans1': data['ans1'],
                'ans2': data['ans2'],
                'label': data['label'],
                'answers': answers,
            }
            entries.append(BBQ(**bbq_data))
    return entries


@safe
def parse_stereoset_file(file_path: Path, entry_type: str) -> list[StereoSet]:
    entries = []
    with file_path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
        raw_data = raw.get('data', raw) if isinstance(raw, dict) else raw

        # Fixed line: prefer lowercase key
        raw_data = raw_data.get(entry_type) or raw_data.get(entry_type.capitalize()) or raw_data

        for d in raw_data:
            sentences = []
            for s in d['sentences']:
                labels = [
                    StereoSetLabel(label=l['label'], human_id=l['human_id']) for l in s['labels']
                ]
                sentence = StereoSetSentence(
                    sentence=s['sentence'],
                    sentence_id=s['id'],
                    gold_label=s['gold_label'],
                    labels=labels,
                )
                sentences.append(sentence)
            entry = StereoSet(
                id=d['id'],
                target=d['target'],
                bias_type=d['bias_type'],
                context=d['context'],
                type=entry_type,
                sentences=sentences,
            )
            entries.append(entry)
    return entries
@future_safe
async def insert_chunk(session: AsyncSession, chunk: list[SQLModel]) -> None:
    async with session.begin_nested():
        session.add_all(chunk)
    await session.commit()


async def load_bbq(session: AsyncSession, base_path: Path) -> None:
    bbq_dir = base_path / 'bbq'
    if not bbq_dir.exists():
        raise FileNotFoundError(f'BBQ directory not found: {bbq_dir}')

    categories = [
        'Age',
        'Disability_status',
        'Gender_identity',
        'Nationality',
        'Physical_appearance',
        'Race_ethnicity',
        'Race_x_SES',
        'Race_x_gender',
        'Religion',
        'SES',
        'Sexual_orientation',
    ]

    for cat in categories:
        file_path = bbq_dir / f'{cat}.jsonl'
        if not file_path.exists():
            logger.warning(f'Skipping missing file: {file_path}')
            continue

        parse_result: Result[list[BBQ], Exception] = parse_bbq_file(file_path, cat)
        if isinstance(parse_result, Failure):
            logger.exception(f'Failed parsing BBQ {cat}: {parse_result.failure()}')
            continue

        entries = parse_result.unwrap()
        chunk_size = 1000
        chunks = [entries[i : i + chunk_size] for i in range(0, len(entries), chunk_size)]

        insert_futures: list[FutureResult[None, Exception]] = [
            insert_chunk(session, chunk) for chunk in chunks
        ]
        results = await asyncio.gather(*(fr.awaitable() for fr in insert_futures))

        failures = [r for r in results if isinstance(r, Failure)]
        if failures:
            for fail in failures:
                logger.exception(f'Insert failure in BBQ {cat}: {fail.failure()}')
        else:
            logger.info(f'Loaded {len(entries)} BBQ entries for {cat}')


async def load_stereoset(session: AsyncSession, base_path: Path) -> None:
    stereoset_dir = base_path / 'stereoset'
    if not stereoset_dir.exists():
        raise FileNotFoundError(f'StereoSet directory not found: {stereoset_dir}')

    files = [('intersentence.json', 'intersentence'), ('intrasentence.json', 'intrasentence')]

    for filename, entry_type in files:
        file_path = stereoset_dir / filename
        if not file_path.exists():
            logger.warning(f'Skipping missing file: {file_path}')
            continue

        parse_result: Result[list[StereoSet], Exception] = parse_stereoset_file(
            file_path, entry_type
        )
        if isinstance(parse_result, Failure):
            logger.exception(f'Failed parsing StereoSet {filename}: {parse_result.failure()}')
            continue

        entries = parse_result.unwrap()
        chunk_size = 1000
        chunks = [entries[i : i + chunk_size] for i in range(0, len(entries), chunk_size)]

        insert_futures: list[FutureResult[None, Exception]] = [
            insert_chunk(session, chunk) for chunk in chunks
        ]
        results = await asyncio.gather(*(fr.awaitable() for fr in insert_futures))

        failures = [r for r in results if isinstance(r, Failure)]
        if failures:
            for fail in failures:
                logger.exception(f'Insert failure in StereoSet {filename}: {fail.failure()}')
        else:
            logger.info(f'Loaded {len(entries)} StereoSet entries from {filename}')


async def main() -> None:
    db_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./datasets.db')
    engine = create_async_engine(db_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        base_path = Path('datasets')
        await asyncio.gather(
            load_bbq(session, base_path),
            load_stereoset(session, base_path),
        )


def run() -> None:
    asyncio.run(main())


if __name__ == '__main__':
    run()
