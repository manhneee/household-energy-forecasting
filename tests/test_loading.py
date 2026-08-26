from src.data.loading import discover_house_files, load_raw_houses


def test_discover_kaggle_house_underscore_names(tmp_path):
    (tmp_path / "House_1.csv").write_text("Time,Aggregate\n2014-01-01 00:00:00,100\n")
    (tmp_path / "House_21.csv").write_text("Time,Aggregate\n2014-01-01 00:00:00,100\n")
    found = discover_house_files(tmp_path)
    assert set(found) == {1, 21}


def test_discover_does_not_invent_house_14(tiny_refit):
    found = discover_house_files(tiny_refit)
    assert 14 not in found
    assert set(found) == {1, 2, 3, 5}


def test_exclude_solar_houses(tiny_refit):
    houses = load_raw_houses(tiny_refit, exclude=[3, 11, 21])
    assert 3 not in houses
    assert set(houses) == {1, 2, 5}


def test_rejects_house_14_file(tmp_path):
    (tmp_path / "CLEAN_House14.csv").write_text("DateTime,Aggregate\n2014-01-01 00:00:00,1\n")
    try:
        discover_house_files(tmp_path)
        assert False, "should have raised"
    except RuntimeError as exc:
        assert "House 14" in str(exc)
