"""Tests for resources that must be present in the installed package."""

from importlib.resources import as_file, files

import pytest
import yaml
from astropy.table import QTable


MODULE_KEYS = (
    "f280",
    "f350",
    "f420",
    "f850",
    "f_fake1",
    "f_fake2",
    "f_fake3",
)

TABLE_FILENAMES = tuple(
    f"primecam_{module_key}_dettable.h5" for module_key in MODULE_KEYS
)


def package_files():
    return files("primecam_focalplane") / "fp_files"


def test_bundled_config_lists_every_module():
    config_resource = package_files() / "fp_config.yaml"

    assert config_resource.is_file()
    config = yaml.safe_load(config_resource.read_text(encoding="utf-8"))

    assert set(config["modules"]) == set(MODULE_KEYS)


@pytest.mark.parametrize("filename", TABLE_FILENAMES)
def test_detector_table_is_bundled_and_readable(filename):
    table_resource = package_files() / "tables" / filename

    assert table_resource.is_file(), f"missing package resource: {filename}"
    with as_file(table_resource) as table_path:
        table = QTable.read(table_path, path="dettable_stack")

    assert len(table) > 0
    assert {"name", "quat", "module", "index"} <= set(table.colnames)
    assert table["quat"].shape == (len(table), 4)

