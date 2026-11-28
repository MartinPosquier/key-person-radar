"""Owner: data session. FNSPID streaming subset (tasks D8, D12)."""

import os
import tempfile

from kpr.collect.fnspid import subset

CSV = """Unnamed: 0,Date,Article_title,Stock_symbol,Url,Publisher,Author,Article
0,2023-12-16 23:00:00 UTC,Jensen Huang says hi,XXX,u1,P,,"long
multi-line text"
1,2023-12-15 10:00:00 UTC,Nothing here,NVDA,u2,,,
2,2023-12-14 10:00:00 UTC,Other stuff,ZZZ,u3,,,
"""


def test_subset_keeps_our_tickers_and_titles_naming_a_person():
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        fh.write(CSV)
    try:
        df = subset(path, {"NVDA"}, ["jensen huang"])
    finally:
        os.remove(path)
    assert df["url"].tolist() == ["u1", "u2"]
    assert str(df["date_utc"].iloc[0]) == "2023-12-16 23:00:00"
