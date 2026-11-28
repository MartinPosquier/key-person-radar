"""Owner: data session. GDELT query building and aggregation (tasks D5, D8, dossier §5.3)."""

import numpy as np
import pandas as pd

from kpr.collect.gdelt import alias_rows, build_daily, build_query, decide, extrapolate
from kpr.schemas import validate

TIB = 1024**4


def test_query_filters_partitions_and_reads_only_four_columns():
    sql = build_query([("jensen_huang", "jensen huang"), ("x", "o'leary")], "2019-06-01",
                      "2019-07-01")
    assert "_PARTITIONTIME >= TIMESTAMP('2019-06-01')" in sql
    assert "_PARTITIONTIME < TIMESTAMP('2019-07-01')" in sql
    assert r"'o\'leary'" in sql  # quotes are escaped
    for col in ("g.DATE", "g.Persons", "g.V2Tone", "g.DocumentIdentifier"):
        assert col in sql
    assert "SELECT *" not in sql.upper()


def test_alias_shared_by_two_persons_is_dropped():
    persons = [{"person_id": "a", "aliases": ["luca de meo", "a b"]},
               {"person_id": "b", "aliases": ["luca de meo"]}]
    assert alias_rows(persons) == [("a", "a b")]


def test_extrapolation_and_decision_rule():
    total = extrapolate(10 * 10**9, "2019-06", "2015-02-19", "2025-12-31")
    assert np.isclose(total, 10e9 * 3969 / 30)
    assert decide(0.4 * TIB, TIB, 0.5) == "GDELT"
    assert decide(0.6 * TIB, TIB, 0.5) == "FNSPID"


def test_build_daily_fills_days_without_articles_and_hides_thin_tone():
    raw = pd.DataFrame({"person_id": ["a", "a"], "date": ["2020-01-01", "2020-01-03"],
                        "n_articles": [5, 2], "tone_mean": [-1.0, 2.0],
                        "tone_neg_share": [0.6, 0.0], "tone_std": [1.0, 0.5]})
    df = build_daily(raw, ["a", "b"], "2020-01-01", "2020-01-03", 3)
    validate(df, "gdelt_daily")
    assert len(df) == 6
    a = df[df.person_id == "a"].set_index("date")
    assert a["n_articles"].tolist() == [5, 0, 2]
    assert a["tone_mean"].iloc[0] == -1.0 and np.isnan(a["tone_mean"].iloc[2])
