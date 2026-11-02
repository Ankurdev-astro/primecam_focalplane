"""Tests for constructing Prime-Cam detector tables."""

from importlib.resources import as_file, files
from types import SimpleNamespace

import pytest

import primecam_focalplane.fp_primecam as fp_primecam
from primecam_focalplane.fp_primecam import (
    ConfigError,
    PrimeCamConfig,
    PrimeCamFocalplane,
)


@pytest.fixture(scope="module")
def config_path():
    config_resource = (
        files("primecam_focalplane") / "fp_files" / "fp_config.yaml"
    )
    with as_file(config_resource) as path:
        yield path


def test_load_f280_config(config_path):
    config = PrimeCamConfig.from_yaml(config_path, "f280")

    assert config.detector.band_name == "f280"
    assert config.module.name == "primecam_f280"
    assert len(config.module.wafers) == 3
    assert config.output.table_path == "dettable_stack"


def test_unknown_module_is_rejected(config_path):
    with pytest.raises(ConfigError, match="module 'not_a_module' was not found"):
        PrimeCamConfig.from_yaml(config_path, "not_a_module")


def test_build_f280_detector_table(config_path):
    config = PrimeCamConfig.from_yaml(config_path, "f280")
    table = PrimeCamFocalplane(config).build()

    assert len(table) > 0
    assert table["quat"].shape == (len(table), 4)
    assert set(table["module"]) == {"primecam_f280"}
    assert list(table["index"]) == list(range(len(table)))
    assert {
        "name",
        "quat",
        "module",
        "wafer_slot",
        "band",
        "pixel",
        "pol",
        "index",
    } <= set(table.colnames)


def test_main_builds_all_configured_modules(config_path, monkeypatch):
    built_modules = []

    class RecordingFocalplane:
        def __init__(self, config):
            self.config = config

        def write(self, path):
            assert path == config_path.resolve()
            built_modules.append(self.config.detector.band_name)

    monkeypatch.setattr(
        fp_primecam,
        "parse_args",
        lambda: SimpleNamespace(config=config_path, module="all"),
    )
    monkeypatch.setattr(fp_primecam, "PrimeCamFocalplane", RecordingFocalplane)

    assert fp_primecam.main() == 0
    assert built_modules == [
        "f280",
        "f350",
        "f420",
        "f850",
        "f_fake1",
        "f_fake2",
        "f_fake3",
    ]

