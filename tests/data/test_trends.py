"""Owner: data session. Google Trends ratios (task D7, dossier §4.4 step 2, §4.5)."""

import numpy as np
import pandas as pd

from kpr.collect.trends import COMPANY_HINTS, PERSON_HINTS, choose_entity, yearly_ratio


def test_choose_entity_prefers_the_business_person_over_a_homonym():
    sugg = [{"mid": "/m/2", "title": "Tim Cook", "type": "Historian"},
            {"mid": "/m/1", "title": "Tim Cook", "type": "CEO of Apple"},
            {"mid": "/m/3", "title": "Tim Cook book", "type": "Book"}]
    assert choose_entity(sugg, "tim cook", PERSON_HINTS)["mid"] == "/m/1"
    assert choose_entity(sugg[:1], "tim cook", PERSON_HINTS) is None
    assert choose_entity([{"mid": "/m/9", "title": "Nvidia", "type": "Technology company"}],
                         "Nvidia", COMPANY_HINTS)["mid"] == "/m/9"


def test_yearly_ratio_is_the_median_of_weekly_ratios_skipping_zero_company_weeks():
    idx = pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-19", "2021-01-03"])
    weekly = pd.DataFrame({"person": [10, 20, 5, 1], "company": [100, 100, 0, 50]}, index=idx)
    r = yearly_ratio(weekly)
    assert np.isclose(r[2020], 0.15) and np.isclose(r[2021], 0.02)
