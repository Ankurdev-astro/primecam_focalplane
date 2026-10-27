"""Primary script for building CCAT modules in TOAST QTable format.

The selected module is read from a YAML configuration and written to an HDF5
file using the configured output filename and table path.

Usage
-----
Run from the repository root, selecting a module key from ``fp_config.yaml``::

    python fp_scripts/fp_primecam.py fp_scripts/fp_config.yaml --module f280
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import toast.qarray as qa
import yaml
from astropy.table import Column, QTable, vstack
from toast.instrument_coords import xieta_to_quat
from toast.instrument_sim import rhomb_dim, rhombus_hex_layout, rhomb_xieta_row_col


class ConfigError(ValueError):
    """Raised when a focal-plane configuration is invalid."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a YAML mapping")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number")
    return float(value)


def _positive(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ConfigError(f"'{name}' must be greater than zero")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer")
    return value


@dataclass(frozen=True)
class PolarizationConfig:
    offset_deg: float
    even_row_deg: float
    odd_row_deg: float
    detector_b_offset_deg: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolarizationConfig":
        return cls(
            offset_deg=_number(data["offset_deg"], "polarization.offset_deg"),
            even_row_deg=_number(
                data["even_row_deg"], "polarization.even_row_deg"
            ),
            odd_row_deg=_number(data["odd_row_deg"], "polarization.odd_row_deg"),
            detector_b_offset_deg=_number(
                data["detector_b_offset_deg"],
                "polarization.detector_b_offset_deg",
            ),
        )


@dataclass(frozen=True)
class PSDConfig:
    net_uk_sqrt_s: float
    fmin_hz: float
    fknee_hz: float
    alpha: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PSDConfig":
        return cls(
            net_uk_sqrt_s=_positive(data["net_uk_sqrt_s"], "psd.net_uk_sqrt_s"),
            fmin_hz=_positive(data["fmin_hz"], "psd.fmin_hz"),
            fknee_hz=_positive(data["fknee_hz"], "psd.fknee_hz"),
            alpha=_number(data["alpha"], "psd.alpha"),
        )


@dataclass(frozen=True)
class DetectorConfig:
    band_name: str
    n_pixels: int
    pixel_size_mm: float
    rhombus_gap_mm: float
    bandcenter_ghz: float
    bandwidth_ghz: float
    fwhm_arcmin: float
    sample_rate_hz: float
    polarization: PolarizationConfig
    psd: PSDConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorConfig":
        band_name = data["band_name"]
        if not isinstance(band_name, str) or not band_name:
            raise ConfigError("'detector.band_name' must be a non-empty string")

        n_pixels = _integer(data["n_pixels"], "detector.n_pixels")
        if n_pixels <= 0 or n_pixels % 3 != 0:
            raise ConfigError("'detector.n_pixels' must be positive and divisible by 3")

        nrhombus = n_pixels // 3
        dim = rhomb_dim(nrhombus)
        if dim * dim != nrhombus:
            raise ConfigError(
                "'detector.n_pixels / 3' must form a square rhombus layout"
            )

        return cls(
            band_name=band_name,
            n_pixels=n_pixels,
            pixel_size_mm=_positive(
                data["pixel_size_mm"], "detector.pixel_size_mm"
            ),
            rhombus_gap_mm=_number(
                data["rhombus_gap_mm"], "detector.rhombus_gap_mm"
            ),
            bandcenter_ghz=_positive(
                data["bandcenter_ghz"], "detector.bandcenter_ghz"
            ),
            bandwidth_ghz=_positive(
                data["bandwidth_ghz"], "detector.bandwidth_ghz"
            ),
            fwhm_arcmin=_positive(data["fwhm_arcmin"], "detector.fwhm_arcmin"),
            sample_rate_hz=_positive(
                data["sample_rate_hz"], "detector.sample_rate_hz"
            ),
            polarization=PolarizationConfig.from_dict(
                _mapping(data["polarization"], "detector.polarization")
            ),
            psd=PSDConfig.from_dict(_mapping(data["psd"], "detector.psd")),
        )


@dataclass(frozen=True)
class KillConfig:
    enabled: bool
    row_offsets: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KillConfig":
        enabled = data["enabled"]
        if not isinstance(enabled, bool):
            raise ConfigError("'kill.enabled' must be true or false")

        raw_offsets = data.get("row_offsets", [])
        if not isinstance(raw_offsets, list):
            raise ConfigError("'kill.row_offsets' must be a list of integers")
        offsets = tuple(
            _integer(value, "kill.row_offsets entry") for value in raw_offsets
        )
        if enabled and not offsets:
            raise ConfigError("enabled detector removal needs at least one row offset")
        if any(value < 0 for value in offsets):
            raise ConfigError("'kill.row_offsets' entries cannot be negative")
        return cls(enabled=enabled, row_offsets=offsets)


@dataclass(frozen=True)
class WaferConfig:
    name: str
    position: int
    rotation_deg: float
    kill: KillConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaferConfig":
        name = data["name"]
        if not isinstance(name, str) or not name or "_" in name:
            raise ConfigError(
                "'wafer.name' must be a non-empty string without underscores"
            )
        position = _integer(data["position"], f"wafer {name}.position")
        if position not in (0, 1, 2):
            raise ConfigError(f"wafer {name} position must be 0, 1, or 2")
        return cls(
            name=name,
            position=position,
            rotation_deg=_number(data["rotation_deg"], f"wafer {name}.rotation_deg"),
            kill=KillConfig.from_dict(_mapping(data["kill"], f"wafer {name}.kill")),
        )


@dataclass(frozen=True)
class ModuleConfig:
    name: str
    plate_scale_deg_per_mm: float
    wafer_center_distance_mm: float
    wafer_center_angle_deg: float
    center_rotation_deg: float
    rotation_axis: tuple[float, float, float]
    wafers: tuple[WaferConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleConfig":
        name = data["name"]
        if not isinstance(name, str) or not name:
            raise ConfigError("'module.name' must be a non-empty string")

        raw_axis = data["rotation_axis"]
        if not isinstance(raw_axis, list) or len(raw_axis) != 3:
            raise ConfigError("'module.rotation_axis' must contain three numbers")
        axis = tuple(
            _number(value, "module.rotation_axis entry") for value in raw_axis
        )
        if np.isclose(np.linalg.norm(axis), 0.0):
            raise ConfigError("'module.rotation_axis' cannot be the zero vector")

        raw_wafers = data["wafers"]
        if not isinstance(raw_wafers, list) or not raw_wafers:
            raise ConfigError("'module.wafers' must be a non-empty list")
        wafers = tuple(WaferConfig.from_dict(_mapping(x, "wafer")) for x in raw_wafers)
        names = [wafer.name for wafer in wafers]
        if len(names) != len(set(names)):
            raise ConfigError("wafer names must be unique")
        positions = [wafer.position for wafer in wafers]
        if len(positions) != len(set(positions)):
            raise ConfigError("wafer positions must be unique")

        return cls(
            name=name,
            plate_scale_deg_per_mm=_positive(
                data["plate_scale_deg_per_mm"], "module.plate_scale_deg_per_mm"
            ),
            wafer_center_distance_mm=_positive(
                data["wafer_center_distance_mm"],
                "module.wafer_center_distance_mm",
            ),
            wafer_center_angle_deg=_number(
                data["wafer_center_angle_deg"], "module.wafer_center_angle_deg"
            ),
            center_rotation_deg=_number(
                data["center_rotation_deg"], "module.center_rotation_deg"
            ),
            rotation_axis=axis,
            wafers=wafers,
        )


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str
    file: str
    table_path: str
    overwrite: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputConfig":
        out_dir = data["out_dir"]
        filename = data["file"]
        table_path = data.get("table_path", "dettable_stack")
        overwrite = data.get("overwrite", False)
        if not isinstance(out_dir, str) or not out_dir:
            raise ConfigError("'output.out_dir' must be a non-empty string")
        if not isinstance(filename, str) or not filename:
            raise ConfigError("'output.file' must be a non-empty string")
        if Path(filename).is_absolute():
            raise ConfigError("'output.file' must be relative to 'output.out_dir'")
        if not isinstance(table_path, str) or not table_path:
            raise ConfigError("'output.table_path' must be a non-empty string")
        if not isinstance(overwrite, bool):
            raise ConfigError("'output.overwrite' must be true or false")
        return cls(
            out_dir=out_dir,
            file=filename,
            table_path=table_path,
            overwrite=overwrite,
        )

    def filename_for(self, module_name: str) -> str:
        """Render the output filename using values from the module config."""
        try:
            filename = self.file.format(name=module_name)
        except (KeyError, IndexError, ValueError) as error:
            raise ConfigError(
                "'output.file' may only use the '{name}' format field"
            ) from error
        if not filename:
            raise ConfigError("formatted 'output.file' cannot be empty")
        if Path(filename).is_absolute():
            raise ConfigError("formatted 'output.file' must be relative")
        return filename


@dataclass(frozen=True)
class PrimeCamConfig:
    detector: DetectorConfig
    module: ModuleConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, config_path: Path, module_name: str) -> "PrimeCamConfig":
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                raw = yaml.safe_load(config_file)
        except OSError as error:
            raise ConfigError(f"could not read {config_path}: {error}") from error
        except yaml.YAMLError as error:
            raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

        data = _mapping(raw, "configuration")
        try:
            modules = _mapping(data["modules"], "modules")
            if module_name not in modules:
                available = ", ".join(sorted(str(name) for name in modules))
                raise ConfigError(
                    f"module '{module_name}' was not found; "
                    f"available modules: {available or 'none'}"
                )
            selected = _mapping(
                modules[module_name], f"modules.{module_name}"
            )
            return cls(
                detector=DetectorConfig.from_dict(
                    _mapping(selected["detector"], f"modules.{module_name}.detector")
                ),
                module=ModuleConfig.from_dict(
                    _mapping(selected["module"], f"modules.{module_name}.module")
                ),
                output=OutputConfig.from_dict(
                    _mapping(selected["output"], f"modules.{module_name}.output")
                ),
            )
        except KeyError as error:
            raise ConfigError(f"missing required configuration key: {error.args[0]}") from error


class PrimeCamFocalplane:
    """Build a PrimeCam focal-plane detector table from validated parameters."""

    def __init__(self, config: PrimeCamConfig):
        self.config = config
        self._axis = np.asarray(config.module.rotation_axis, dtype=np.float64)

    @staticmethod
    def _natural_sum(n: int) -> int:
        return int(n * (n + 1) * 0.5)

    @classmethod
    def _kill_detector_indices(cls, dim: int, offset: int) -> set[int]:
        detector_locations_per_wafer = dim * dim
        middle_pixel = cls._natural_sum(dim) - 1
        left_pixel = cls._natural_sum(dim - offset + 1) - 1
        right_pixel = (
            detector_locations_per_wafer - cls._natural_sum(dim - offset) - 1
        )
        return {
            index + rhombus * detector_locations_per_wafer
            for index in (middle_pixel, left_pixel, right_pixel)
            for rhombus in range(3)
        }

    def _build_wafer_layout(self, wafer: WaferConfig) -> dict[str, dict[str, Any]]:
        detector = self.config.detector
        module = self.config.module
        nrhombus = detector.n_pixels // 3
        dim = rhomb_dim(nrhombus)

        pixel_separation = (
            module.plate_scale_deg_per_mm * detector.pixel_size_mm * u.degree
        )
        width = (dim - 1) * pixel_separation
        gap = (
            module.plate_scale_deg_per_mm * detector.rhombus_gap_mm * u.degree
        )

        pol = detector.polarization
        pol_a = np.zeros(nrhombus, dtype=np.float64)
        for pixel in range(nrhombus):
            row, _ = rhomb_xieta_row_col(nrhombus, pixel)
            base_angle = pol.even_row_deg if row % 2 == 0 else pol.odd_row_deg
            pol_a[pixel] = base_angle + pol.offset_deg
        pol_b = pol_a + pol.detector_b_offset_deg

        prefix = f"{wafer.name}_p"
        suffix_a = f"_PC_{detector.band_name}_A"
        suffix_b = f"_PC_{detector.band_name}_B"
        layout_a = rhombus_hex_layout(
            nrhombus,
            width,
            prefix,
            suffix_a,
            gap=gap,
            pol=u.Quantity(pol_a, u.degree),
        )
        layout_b = rhombus_hex_layout(
            nrhombus,
            width,
            prefix,
            suffix_b,
            gap=gap,
            pol=u.Quantity(pol_b, u.degree),
        )

        killed: set[int] = set()
        if wafer.kill.enabled:
            for offset in wafer.kill.row_offsets:
                if offset > dim:
                    raise ConfigError(
                        f"kill row offset {offset} exceeds wafer dimension {dim}"
                    )
                killed.update(self._kill_detector_indices(dim, offset))

        layout = {}
        for name, properties in (layout_a | layout_b).items():
            pixel_number = int(name.split("_")[1].removeprefix("p"))
            if pixel_number not in killed:
                layout[name] = properties

        sorted_layout = dict(sorted(layout.items()))
        print(
            f"Band: {detector.band_name}; Wafer: {wafer.name}; "
            f"detectors: {len(sorted_layout)}"
        )
        return sorted_layout

    def _make_detector_table(
        self, focalplane: dict[str, dict[str, Any]]
    ) -> QTable:
        detector = self.config.detector
        names = sorted(focalplane)
        n_detectors = len(names)
        gamma = u.Quantity(
            [focalplane[name]["gamma"] for name in names], u.radian
        )

        table = QTable(
            [
                Column(name="name", data=names),
                Column(
                    name="quat", data=[focalplane[name]["quat"] for name in names]
                ),
                Column(
                    name="module",
                    data=[self.config.module.name] * n_detectors,
                ),
                Column(name="wafer_slot", data=[name.split("_")[0] for name in names]),
                Column(name="band", data=[f"PC_{name.split('_')[3]}" for name in names]),
                Column(name="pixel", data=[name.split("_")[1][1:] for name in names]),
                Column(name="pol", data=[name.split("_")[4] for name in names]),
                Column(name="psi_pol", length=n_detectors, unit=u.rad),
                Column(name="gamma", length=n_detectors, unit=u.rad),
                Column(name="fwhm", length=n_detectors, unit=u.arcmin),
                Column(name="psd_fmin", length=n_detectors, unit=u.Hz),
                Column(name="psd_fknee", length=n_detectors, unit=u.Hz),
                Column(name="psd_alpha", length=n_detectors, unit=None),
                Column(
                    name="psd_net",
                    length=n_detectors,
                    unit=u.K * np.sqrt(1.0 * u.second),
                ),
                Column(name="bandcenter", length=n_detectors, unit=u.GHz),
                Column(name="bandwidth", length=n_detectors, unit=u.GHz),
            ]
        )

        table["psi_pol"] = [
            0.0 if name.endswith("A") else np.pi / 2 for name in names
        ] * u.rad
        table["gamma"] = gamma
        table["fwhm"] = detector.fwhm_arcmin * u.arcmin
        table["psd_fmin"] = detector.psd.fmin_hz * u.Hz
        table["psd_fknee"] = detector.psd.fknee_hz * u.Hz
        table["psd_alpha"] = detector.psd.alpha
        table["psd_net"] = (
            detector.psd.net_uk_sqrt_s
            * 1.0e-6
            * u.K
            * np.sqrt(1.0 * u.second)
        )
        table["bandcenter"] = detector.bandcenter_ghz * u.GHz
        table["bandwidth"] = detector.bandwidth_ghz * u.GHz
        return table

    def _wafer_centers(self) -> list[np.ndarray]:
        module = self.config.module
        angle_radians = np.radians(module.wafer_center_angle_deg)
        wafer_spacing_mm = np.sqrt(
            2
            * module.wafer_center_distance_mm**2
            * (1 - np.cos(angle_radians))
        )
        radius = (
            0.5
            * wafer_spacing_mm
            * module.plate_scale_deg_per_mm
            * np.pi
            / 180.0
        )
        centers = [
            xieta_to_quat(-radius, radius / np.sqrt(3.0), 0.0),
            xieta_to_quat(radius, radius / np.sqrt(3.0), 0.0),
            xieta_to_quat(0.0, -2.0 * radius / np.sqrt(3.0), 0.0),
        ]
        center_rotation = qa.from_axisangle(
            self._axis, np.radians(module.center_rotation_deg)
        )
        return list(qa.mult(center_rotation, centers))

    def build(self) -> QTable:
        centers = self._wafer_centers()
        placed_tables = []

        for wafer in self.config.module.wafers:
            table = self._make_detector_table(self._build_wafer_layout(wafer))
            wafer_rotation = qa.from_axisangle(
                self._axis, np.radians(wafer.rotation_deg)
            )
            for detector_index, detector_quat in enumerate(table["quat"]):
                rotated = qa.mult(wafer_rotation, detector_quat)
                table["quat"][detector_index] = qa.mult(
                    centers[wafer.position], rotated
                )
            placed_tables.append(table)

        focalplane = vstack(placed_tables)
        focalplane["index"] = np.arange(len(focalplane))
        return focalplane

    def write(self, config_path: Path) -> Path:
        output_dir = Path(self.config.output.out_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self.config.output.filename_for(self.config.module.name)
        output = (output_dir / filename).resolve()
        if output_dir not in output.parents:
            raise ConfigError("'output.file' must stay inside 'output.out_dir'")
        output.parent.mkdir(parents=True, exist_ok=True)

        table = self.build()
        print(f"Writing {len(table)} detectors to {output} ...")
        table.write(
            output,
            path=self.config.output.table_path,
            serialize_meta=True,
            overwrite=self.config.output.overwrite,
        )
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PrimeCam detector table from a YAML configuration."
    )
    parser.add_argument("config", type=Path, help="path to the YAML configuration")
    parser.add_argument(
        "--module",
        required=True,
        help="module key to build (for example, f280 or f350)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = PrimeCamConfig.from_yaml(config_path, args.module)
        print(f"Building module: {args.module}")
        PrimeCamFocalplane(config).write(config_path)
    except ConfigError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
