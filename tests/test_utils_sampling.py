"""Test local file operations, data helpers, and repeatable sampling."""

import pickle
import random
from timeit import default_timer

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine

from tsdst import sampling as s, utils as u


def test_database_helpers_with_local_sqlite(tmp_path, monkeypatch):
    database = tmp_path / "test.db"
    monkeypatch.setattr(u, "create_engine", lambda *args, **kwargs: create_engine("sqlite:///" + str(database)))
    monkeypatch.setattr(u, "getpass", lambda **kwargs: "local-password")
    frame = pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})
    u.dsn_saveTable(frame, "sample", "user", "dialect", "dsn", to_sql_args={"index": False})
    assert_frame_equal(u.dsn_getTable("select * from sample", "user", "dialect", "dsn"), frame)
    results = u.dsn_getTable(["select * from sample", "select count(*) as n from sample"],
                             "user", "dialect", "dsn", pw="provided")
    assert len(results) == 2 and results[1].iloc[0, 0] == 2


def test_database_connection_disposed_after_error(monkeypatch):
    from unittest.mock import Mock
    engine = Mock()
    monkeypatch.setattr(u, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(pd, "read_sql", Mock(side_effect=ValueError("bad query")))
    with pytest.raises(ValueError):
        u.dsn_getTable("bad query", "user", "dialect", "dsn", pw="provided")
    engine.dispose.assert_called_once()


def test_time_and_local_output(tmp_path, capsys):
    assert u.pretty_print_time(3661) == "01:01:01"
    assert u.pretty_print_time(10, 11.25, decimals=2) == "00:00:01.25"
    directory = str(tmp_path / "new")
    assert u.checkDir(directory, verbose=False) is False
    assert u.checkDir(directory, verbose=False) is True
    text = u.print_message_with_time("Done ", 1, display_realtime=False, log=False)
    assert "Done 00:00:01" in text
    u.print_time("Done ", 1, display_realtime=False, log=True, log_dir="logs")
    assert (tmp_path / "logs" / "pmwt.log").exists()
    u.updateProgBar(1, 2, default_timer())
    assert "50.00%" in capsys.readouterr().out
    u.save_checkpoint({"value": 2}, "Saved ", ts=1, display_realtime=False,
                      log=False, checkpoint_dir="checkpoints")
    with (tmp_path / "checkpoints" / "chkpnt.pkl").open("rb") as file:
        assert pickle.load(file) == {"value": 2}


def test_dataframe_helpers():
    frame = pd.DataFrame({"first": [1, 2], "long_name": [3, 4]}, index=[10, 20])
    assert u.move_columns_to_end(frame, "first").columns.tolist() == ["long_name", "first"]
    assert u.keywordSearch(["red apple", "green apple", "red pear"], ["apple"], ["green"]) == ["red apple"]
    assert_frame_equal(u.toFrame(frame.to_numpy(), frame), frame)
    for dtype, kind in [("boolean", "categorical"), ("string", "categorical"),
                        ("Int64", "numeric"), ("datetime64[ns]", "date")]:
        values = [None, None] if kind == "date" else [1, None] if dtype == "Int64" else [True, None] if dtype == "boolean" else ["a", None]
        assert u.inferFeatureType(pd.Series(values, dtype=dtype)) == kind
    assert u.inferFeatureType(pd.Series([1, 2, None]), n_unique=2) == "categorical"


def test_array_helpers():
    x = np.array([0, 1, 2])
    assert u.reshape_to_vector(x).shape == (3, 1)
    assert u.reshape_to_vector(x, axis=0).shape == (1, 3)
    with pytest.raises(ValueError):
        u.reshape_to_vector(x, axis=2)
    assert_array_equal(u.one_hot_decode(u.one_hot_encode(x)), x)
    assert_array_equal(u.decision_boundary_1D(np.array([.1, .5, .9])).ravel(), [0, 1, 1])
    assert_array_equal(u.decision_boundary(np.array([[.9, .1], [.2, .8]])), [[1, 0], [0, 1]])


def test_weighted_sample_does_not_change_weights_or_global_seed():
    weights = [.5, .3, .2]
    before = weights.copy()
    state = random.getstate()
    first = u.sample(["a", "b", "c"], 3, prob=weights, random_state=2)
    assert sorted(first) == ["a", "b", "c"]
    assert weights == before
    assert random.getstate() == state
    assert u.sample(["a", "b"], 4, replace=True, random_state=3) == u.sample(["a", "b"], 4, replace=True, random_state=3)
    with pytest.raises(ValueError):
        u.sample([1], 2)


def test_random_and_balanced_samples():
    frame = pd.DataFrame({"x": range(12), "y": [0] * 8 + [1] * 4})
    result = s.downSample(frame, "y", random_state=3)
    assert result.y.value_counts().to_dict() == {0: 4, 1: 4}
    first = s.genRandSampFromDF(frame, 5, random_state=3)
    assert_frame_equal(first, s.genRandSampFromDF(frame, 5, random_state=3))
    assert s.genRandSampFromDF(frame.to_numpy(), 3, random_state=2).shape == (3, 2)


@pytest.mark.parametrize("placement", ["random", "spaced"])
def test_latin_hypercube_preserves_rows_and_strata(placement):
    frame = pd.DataFrame({"x": range(23), "y": np.arange(23) * 10})
    result = s.latinHypercube1D(frame, 5, random_state=2, shuffle_after=False, bin_placement=placement)
    assert result.shape == (5, 2)
    assert result.x.is_unique
    assert_allclose(result.y, result.x * 10)
    assert np.all(np.diff(result.x) > 0)
    array = s.latinHypercube1D(frame.to_numpy(), 5, random_state=2, stratified=False)
    assert array.shape == (5, 2)
