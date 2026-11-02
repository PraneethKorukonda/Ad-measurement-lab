"""Run the SQL layer with DuckDB.

DuckDB runs the same SQL a warehouse (BigQuery, Snowflake) would, in-process,
so the pipeline is reproducible on a laptop and in CI. Raw tables are
registered from pandas (simulated data) or read straight from compressed CSV
files (real data); every transformation lives in ``sql/`` so it can be reviewed
as SQL rather than buried in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


def connect(tables: dict[str, pd.DataFrame] | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    for name, df in (tables or {}).items():
        con.register(f"_{name}_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}_df")
        con.unregister(f"_{name}_df")
    return con


def sql(sql_file: str | Path, **params) -> str:
    p = Path(sql_file)
    text = (p if p.is_absolute() else SQL_DIR / p).read_text()
    for k, v in params.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def run_script(con, sql_file: str | Path, **params) -> None:
    con.execute(sql(sql_file, **params))


def query(con, sql_file: str | Path, **params) -> pd.DataFrame:
    return con.execute(sql(sql_file, **params)).df()


@dataclass
class PipelineResult:
    analysis: pd.DataFrame
    raw_quality: pd.DataFrame
    analysis_quality: pd.DataFrame
    campaign_summary: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool((self.analysis_quality["failing"] == 0).all())


def build_campaign_tables(raw: dict[str, pd.DataFrame]) -> PipelineResult:
    """Raw event tables -> quality reports + one analysis row per user."""
    con = connect(raw)
    raw_q = query(con, "campaign/00_quality_raw.sql")
    raw_q["failing_rate"] = raw_q["failing"] / raw_q["total"].clip(lower=1)
    run_script(con, "campaign/01_staging.sql")
    run_script(con, "campaign/02_analysis_table.sql")
    ana_q = query(con, "campaign/03_quality_analysis.sql")
    summary = query(con, "campaign/04_campaign_summary.sql")
    analysis = con.execute("SELECT * FROM analysis_users").df()
    con.close()
    return PipelineResult(analysis=analysis, raw_quality=raw_q, analysis_quality=ana_q,
                          campaign_summary=summary)
