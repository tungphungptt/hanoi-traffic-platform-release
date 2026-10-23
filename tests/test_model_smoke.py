from __future__ import annotations

import numpy as np
import pytest

from tools.model_smoke import _fixed_dimension, _validate_finite_outputs


def test_fixed_dimension_accepts_expected_and_symbolic_dimensions():
    assert _fixed_dimension(1, 1, "batch") == 1
    assert _fixed_dimension("batch_size", 1, "batch") == 1


def test_fixed_dimension_rejects_contract_mismatch():
    with pytest.raises(ValueError, match="does not match expected"):
        _fixed_dimension(640, 320, "width")


def test_model_smoke_outputs_must_exist_and_be_finite():
    assert _validate_finite_outputs([np.array([[0.2, 0.8]], dtype=np.float32)], "model")[0].shape == (1, 2)
    with pytest.raises(ValueError, match="no outputs"):
        _validate_finite_outputs([], "model")
    with pytest.raises(ValueError, match="NaN/Inf"):
        _validate_finite_outputs([np.array([[float("nan")]], dtype=np.float32)], "model")
