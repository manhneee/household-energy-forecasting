from src.data.metadata import SOLAR_CONTAMINATED, TABLE2, household_table


def test_table2_has_twenty_houses_and_skips_14():
    assert 14 not in TABLE2
    assert len(TABLE2) == 20
    assert set(SOLAR_CONTAMINATED) == {3, 11, 21}


def test_numeric_table_covers_all_published_ids():
    table = household_table()
    assert table["household_id"].tolist() == sorted(TABLE2)
    assert table.loc[table["household_id"] == 2, "dwelling_year"].isna().all()
