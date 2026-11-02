"""Tests for converting module-relative tables to the boresight frame."""

from importlib.resources import as_file, files

import numpy as np
import pytest

from primecam_focalplane.module_to_boresight import (
    DEFAULT_CONFIG_PATH,
    ConversionError,
    ModuleToBoresightOffset,
    fyst_offset_to_toast_quat,
    resolve_module_table,
)


def test_default_config_path_points_to_bundled_config():
    config_resource = (
        files("primecam_focalplane") / "fp_files" / "fp_config.yaml"
    )

    with as_file(config_resource) as config_path:
        assert DEFAULT_CONFIG_PATH == config_path.resolve()
        assert DEFAULT_CONFIG_PATH.is_file()


def test_resolve_bundled_f280_table():
    table_config = resolve_module_table("primecam_f280")

    assert table_config.input_path.name == "primecam_f280_dettable.h5"
    assert table_config.input_path.is_file()
    assert table_config.table_path == "dettable_stack"


def test_unknown_module_table_is_rejected():
    with pytest.raises(ConversionError, match="was not found"):
        resolve_module_table("not_a_module")


def test_zero_offset_produces_unit_quaternion():
    quat = fyst_offset_to_toast_quat(0.0, 0.0)

    assert quat.shape == (4,)
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_apply_preserves_module_quaternions():
    converter = ModuleToBoresightOffset("primecam_f280")
    source = converter.load()
    original_quats = np.array(source["quat"], copy=True)

    converted = converter.apply(source)

    assert "module_quat" not in source.colnames
    np.testing.assert_allclose(converted["module_quat"], original_quats)
    assert converted["quat"].shape == original_quats.shape
    assert not np.shares_memory(converted["quat"], source["quat"])

