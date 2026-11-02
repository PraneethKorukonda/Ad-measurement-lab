"""Multiplicity tools and real-data smoke tests (skipped when data are absent)."""

import numpy as np
import pandas as pd
import pytest

from adlift.data.fetch import available
from adlift.estimators.hetero import benjamini_hochberg, winners_curse


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = benjamini_hochberg(p)
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
    assert np.all(q >= p)


def test_winners_curse_picks_max_estimate():
    t = pd.DataFrame({"segment": list("abc"), "estimate": [0.1, 0.3, 0.2], "truth": [0.2, 0.1, 0.15]})
    wc = winners_curse(t)
    assert wc["segment"] == "b" and wc["overstatement"] == pytest.approx(0.2) and wc["true_rank"] == 3


@pytest.mark.realdata
@pytest.mark.skipif(not available("marketing_ab.csv.gz"), reason="run `adlift fetch` first")
def test_ads_psa_quality_and_lift():
    from adlift.studies import ads_psa

    r = ads_psa.run()
    q = r["quality"].set_index("check_name")["value"]
    assert q["rows"] == 588101 and q["duplicate_user_ids"] == 0 and q["null_cells"] == 0
    o = r["overall"].iloc[0]
    assert 0.005 < o["lift"] < 0.01 and o["ci_low"] > 0


@pytest.mark.realdata
@pytest.mark.skipif(not available("meridian_geo_all_channels.csv"), reason="run `adlift fetch` first")
def test_meridian_sql_aggregation_matches_national_file():
    from adlift.data.fetch import path
    from adlift.studies.meridian import load

    df, quality = load()
    nat = pd.read_csv(path("meridian_national_all_channels.csv"))
    assert len(df) == len(nat) == 156
    assert np.allclose(df["spend_channel0"].to_numpy(), nat["Channel0_spend"].to_numpy())
    rev = nat["conversions"] * nat["revenue_per_conversion"]
    assert np.corrcoef(df["revenue"], rev)[0, 1] > 0.999
