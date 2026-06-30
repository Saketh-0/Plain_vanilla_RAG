"""
Runs every question in data/test.csv through the pipeline and writes
outputs/submission.csv matching the columns in sample_submission.csv:
id, prediction, pred_source, pred_section.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import config
from .pipeline import RagPipeline


def run_batch(limit: int | None = None) -> None:
    df = pd.read_csv(config.TEST_CSV)
    if limit:
        df = df.head(limit)

    pipeline = RagPipeline()
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Answering"):
        result = pipeline.answer(row.question)
        rows.append(
            {
                "id": row.id,
                "prediction": result.answer,
                "pred_source": "; ".join(result.source_documents),
                "pred_section": "; ".join(result.source_sections),
            }
        )

    out_df = pd.DataFrame(rows, columns=["id", "prediction", "pred_source", "pred_section"])
    config.ROOT.joinpath("outputs").mkdir(exist_ok=True)
    out_path = config.ROOT / "outputs" / "submission.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} predictions -> {out_path}")
    print(f"Answerable: {out_df['prediction'].ne(config.UNANSWERABLE_MESSAGE).sum()} / {len(out_df)}")
