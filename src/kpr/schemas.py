"""Frozen data contracts shared by every session.

Every table exchanged between sessions is declared here: its producer (the only session
allowed to write it), where it is stored under KPR_DATA_DIR, its key columns and the type
of each column. Change this file only through a contract PR (PARALLEL_SESSIONS.md §6).

Column types are coarse on purpose ("string", "date", "float", "int", "bool") so that
pandas dtype details (int32 vs int64, string vs object) never break a pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

STRING, DATE, FLOAT, INT, BOOL = "string", "date", "float", "int", "bool"


@dataclass(frozen=True)
class Table:
    name: str
    producer: str  # "data" | "quant" | "ml"
    folder: str  # relative to KPR_DATA_DIR
    keys: tuple[str, ...]
    columns: dict[str, str]
    allow_extra: bool = False  # True: extra columns allowed (feature matrices)
    nullable: tuple[str, ...] = field(default_factory=tuple)

    @property
    def filename(self) -> str:
        return f"{self.name}.parquet"


def _t(name, producer, folder, keys, columns, allow_extra=False, nullable=()):
    return Table(name, producer, folder, tuple(keys), dict(columns), allow_extra, tuple(nullable))


TABLES: dict[str, Table] = {
    t.name: t
    for t in [
        # ---- produced by the data session ------------------------------------------------
        _t("prices", "data", "interim", ["company_id", "date"], {
            "company_id": STRING, "date": DATE, "close_adj": FLOAT, "volume": FLOAT,
            "shares_out": FLOAT, "ret": FLOAT,
        }, nullable=["shares_out", "ret"]),
        _t("market", "data", "interim", ["index_id", "date"], {
            "index_id": STRING, "date": DATE, "close": FLOAT, "ret": FLOAT,
        }, nullable=["ret"]),
        _t("vix", "data", "interim", ["date"], {"date": DATE, "vix": FLOAT}),
        _t("wiki_views", "data", "interim", ["page_id", "date"], {
            "page_id": STRING, "date": DATE, "views_en": FLOAT, "views_local": FLOAT,
            "views_total": FLOAT,
        }, nullable=["views_en", "views_local", "views_total"]),
        _t("gdelt_daily", "data", "interim", ["person_id", "date"], {
            "person_id": STRING, "date": DATE, "n_articles": INT, "tone_mean": FLOAT,
            "tone_neg_share": FLOAT, "tone_std": FLOAT,
        }, nullable=["tone_mean", "tone_neg_share", "tone_std"]),
        _t("trends_ratio", "data", "interim", ["person_id", "year"], {
            "person_id": STRING, "year": INT, "ratio_person_company": FLOAT,
        }, nullable=["ratio_person_company"]),
        _t("earnings", "data", "interim", ["company_id", "date"], {
            "company_id": STRING, "date": DATE, "quarter": STRING, "source": STRING,
        }, nullable=["quarter"]),
        _t("mandates", "data", "interim", ["person_id", "company_id", "role", "start"], {
            "person_id": STRING, "company_id": STRING, "role": STRING, "start": DATE,
            "end": DATE, "founder": BOOL,
        }, nullable=["end"]),
        _t("headlines", "data", "interim", ["person_id", "date", "headline_id"], {
            "person_id": STRING, "date": DATE, "headline_id": STRING, "title": STRING,
            "source": STRING,
        }),
        _t("panel", "data", "processed/panel", ["person_id", "date"], {
            "person_id": STRING, "date": DATE, "company_id": STRING, "views": FLOAT,
            "z_wiki": FLOAT, "n_articles": INT, "z_news": FLOAT, "tone_rel": FLOAT,
            "ret": FLOAT, "earnings_window": BOOL, "vix": FLOAT,
        }, nullable=["views", "z_wiki", "z_news", "tone_rel", "ret", "vix"]),
        # ---- produced by the quant session -----------------------------------------------
        _t("events", "quant", "processed/events", ["event_id"], {
            "event_id": STRING, "person_id": STRING, "company_id": STRING, "date_utc": DATE,
            "j0": DATE, "z_wiki": FLOAT, "z_news": FLOAT, "n_articles": INT,
            "tone_rel": FLOAT, "tone_neg_share": FLOAT, "sign": STRING, "car_0_2": FLOAT,
            "car_m5_m1": FLOAT, "car_0_1": FLOAT, "car_0_5": FLOAT, "excluded": BOOL,
            "exclusion_reason": STRING, "is_natural_experiment": BOOL, "split": STRING,
        }, nullable=["tone_rel", "tone_neg_share", "car_0_2", "car_m5_m1", "car_0_1",
                     "car_0_5", "exclusion_reason"]),
        # ---- produced by the ml session --------------------------------------------------
        _t("headlines_tone", "ml", "processed/tone", ["person_id", "date"], {
            "person_id": STRING, "date": DATE, "finbert_pos": FLOAT, "finbert_neg": FLOAT,
            "finbert_net": FLOAT, "n_headlines": INT,
        }),
        _t("features", "ml", "processed/features", ["event_id"], {"event_id": STRING},
           allow_extra=True),
        _t("scores", "ml", "processed/scores", ["person_id", "company_id", "year"], {
            "person_id": STRING, "company_id": STRING, "year": INT,
            "s_minus_raw": FLOAT, "s_plus_raw": FLOAT, "s_minus": FLOAT, "s_plus": FLOAT,
            "s_minus_lo": FLOAT, "s_minus_hi": FLOAT, "s_plus_lo": FLOAT, "s_plus_hi": FLOAT,
            "asymmetry": FLOAT, "v_pct": FLOAT, "v_usd": FLOAT, "v_lo": FLOAT, "v_hi": FLOAT,
            "pi_share": FLOAT, "n_events": INT, "method_agreement": FLOAT,
            "succession_flag": BOOL,
        }, nullable=["asymmetry", "v_pct", "v_usd", "v_lo", "v_hi", "pi_share",
                     "method_agreement", "s_minus_lo", "s_minus_hi", "s_plus_lo", "s_plus_hi"]),
    ]
}

SIGNS = ("negative", "neutral", "positive")
SPLITS = ("train", "validation", "test", "reserved")
SESSIONS = ("data", "quant", "ml")


class SchemaError(ValueError):
    """Raised when a DataFrame does not match its declared contract."""


def _type_ok(series: pd.Series, kind: str) -> bool:
    api = pd.api.types
    non_null = series.dropna()
    if kind == DATE:
        return api.is_datetime64_any_dtype(series)
    if kind == FLOAT:
        return api.is_float_dtype(series) or api.is_integer_dtype(series)
    if kind == INT:
        return api.is_integer_dtype(series) or (
            api.is_float_dtype(series) and bool((non_null % 1 == 0).all())
        )
    if kind == BOOL:
        return api.is_bool_dtype(series)
    if kind == STRING:
        return api.is_string_dtype(series) or api.is_object_dtype(series)
    raise SchemaError(f"unknown column kind {kind!r}")


def validate(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Check `df` against the contract of `table_name`; return it unchanged if valid."""
    if table_name not in TABLES:
        raise SchemaError(f"unknown table {table_name!r}; declared: {sorted(TABLES)}")
    table = TABLES[table_name]
    problems: list[str] = []

    missing = [c for c in table.columns if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    extra = [c for c in df.columns if c not in table.columns]
    if extra and not table.allow_extra:
        problems.append(f"undeclared columns {extra}")

    for col, kind in table.columns.items():
        if col not in df.columns:
            continue
        if not _type_ok(df[col], kind):
            problems.append(f"column {col!r} should be {kind}, got {df[col].dtype}")
        if col not in table.nullable and df[col].isna().any():
            problems.append(f"column {col!r} has missing values but is not nullable")

    if not missing and len(df) and df.duplicated(list(table.keys)).any():
        problems.append(f"duplicate keys on {list(table.keys)}")

    if table_name == "events" and "sign" in df.columns:
        bad = set(df["sign"].dropna()) - set(SIGNS)
        if bad:
            problems.append(f"sign must be one of {SIGNS}, got {sorted(bad)}")
    if table_name == "events" and "split" in df.columns:
        bad = set(df["split"].dropna()) - set(SPLITS)
        if bad:
            problems.append(f"split must be one of {SPLITS}, got {sorted(bad)}")

    if problems:
        raise SchemaError(f"table {table_name!r}: " + "; ".join(problems))
    return df
