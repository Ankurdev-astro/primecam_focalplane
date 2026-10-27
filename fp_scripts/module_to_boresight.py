"""Convert module-relative detector quaternions to boresight-relative ones.

The module name is resolved from ``fp_config.yaml``.  Its configured detector
table is loaded, and the static Prime-Cam module offset is left-multiplied into
every quaternion in the table's ``quat`` column.  The original values are
preserved in ``module_quat`` and ``quat`` becomes relative to the FYST
boresight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import toast.qarray as qa
import yaml
from astropy.table import QTable
from fyst_trajectories import get_primecam_offset
from toast.instrument_coords import xieta_to_quat


DEFAULT_CONFIG_PATH = Path(__file__).with_name("fp_config.yaml")
DEFAULT_TABLE_PATH = "dettable_stack"

# Provisional deployment assignment: the instrument module name comes from the
# nested ``module.name`` field in fp_config.yaml.  This intentionally records
# the separate assumption that primecam_f280 occupies geometric position i6.
MODULE_POSITIONS: Mapping[str, str] = MappingProxyType(
    {
        "primecam_f280": "i6",
        "primecam_f350": "i5",
        "primecam_f420": "i4",
        "primecam_f850": "c",
        "primecam_f_fake1": "i1",
        "primecam_f_fake2": "i2",
        "primecam_f_fake3": "i3",
    }
)

# The physical clocking of this module has not yet been supplied.  Keep the
# aligned zero-clocking assumption explicit and easy to replace.
MODULE_CLOCKING_DEG: Mapping[str, float] = MappingProxyType(
    {
        "primecam_f280": 0.0,
        "primecam_f350": 0.0,
        "primecam_f420": 0.0,
        "primecam_f850": 0.0,
        "primecam_f_fake1": 0.0,
        "primecam_f_fake2": 0.0,
        "primecam_f_fake3": 0.0,
    }
)


class ConversionError(ValueError):
    """Raised when a module table cannot be resolved or converted."""


def fyst_offset_to_toast_quat(
    dx_arcmin: float,
    dy_arcmin: float,
    clocking_deg: float = 0.0,
) -> np.ndarray:
    """Convert FYST great-circle dx/dy to a TOAST xi/eta quaternion."""

    dx_rad = np.radians(dx_arcmin / 60.0)
    dy_rad = np.radians(dy_arcmin / 60.0)
    radius_rad = np.hypot(dx_rad, dy_rad)
    scale = np.sinc(radius_rad / np.pi)
    xi = dx_rad * scale
    eta = dy_rad * scale
    gamma = np.radians(clocking_deg)
    return xieta_to_quat(xi, eta, gamma)


@dataclass(frozen=True)
class ModuleTableConfig:
    """The input-table fields needed from one fp_config.yaml module entry."""

    input_path: Path
    table_path: str


def resolve_module_table(
    module_name: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ModuleTableConfig:
    """Find a module by ``module.name`` and resolve its configured HDF5 file."""

    config_path = Path(config_path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        module_config = next(
            entry
            for entry in config["modules"].values()
            if entry["module"]["name"] == module_name
        )
        name = module_config["module"]["name"]
        output = module_config["output"]
        filename = output["file"].format(name=name)
        table_path = output.get("table_path", DEFAULT_TABLE_PATH)
        output_dir = Path(output["out_dir"]).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
        input_path = (output_dir / filename).resolve()
    except OSError as error:
        raise ConversionError(f"could not read {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConversionError(f"invalid YAML in {config_path}: {error}") from error
    except StopIteration as error:
        raise ConversionError(
            f"module.name '{module_name}' was not found in {config_path}"
        ) from error
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ConversionError(f"invalid module configuration: {error}") from error

    if not input_path.is_file():
        raise ConversionError(f"detector table was not found: {input_path}")

    return ModuleTableConfig(input_path=input_path, table_path=table_path)


class ModuleToBoresightOffset:
    """Load and place one module's detector table in the boresight frame.

    :meth:`apply` returns a copied QTable in which ``module_quat`` is relative
    to the module center and ``quat`` is relative to the FYST boresight.  No
    file is written.

    Example
    -------
    Use the module's assigned position (currently ``primecam_f280 -> i6``)::

        module_offset = ModuleToBoresightOffset("primecam_f280")
        detector_table = module_offset.apply()
    """

    def __init__(
        self,
        module_name: str,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        clocking_deg: float | None = None,
    ) -> None:
        self.module_name = module_name
        self.table_config = resolve_module_table(module_name, config_path)

        try:
            self.position = MODULE_POSITIONS[module_name]
        except KeyError as error:
            raise ConversionError(
                f"no focal-plane position is assigned to '{module_name}'"
            ) from error
        # FYST uses "c" for the center position (equivalent to i0).
        self.offset = get_primecam_offset(self.position)

        if clocking_deg is None:
            try:
                clocking_deg = MODULE_CLOCKING_DEG[module_name]
            except KeyError as error:
                raise ConversionError(
                    f"no module clocking is assigned to '{module_name}'"
                ) from error
        self.clocking_deg = float(clocking_deg)

    @property
    def module_offset_quat(self) -> np.ndarray:
        return fyst_offset_to_toast_quat(
            self.offset.dx,
            self.offset.dy,
            self.clocking_deg,
        )

    def load(self) -> QTable:
        """Load the configured module-relative detector table."""

        try:
            table = QTable.read(
                self.table_config.input_path,
                path=self.table_config.table_path,
            )
        except Exception as error:
            raise ConversionError(
                f"could not read table '{self.table_config.table_path}' from "
                f"{self.table_config.input_path}: {error}"
            ) from error

        if "quat" not in table.colnames:
            raise ConversionError(
                f"{self.table_config.input_path} has no 'quat' column"
            )
        quats = np.asarray(table["quat"])
        if quats.ndim != 2 or quats.shape[1] != 4:
            raise ConversionError(
                f"'quat' must have shape (N, 4), found {quats.shape}"
            )
        return table

    def apply(self, table: QTable | None = None) -> QTable:
        """Return a copy with module- and boresight-relative quaternions."""

        if table is None:
            table = self.load()
        elif "quat" not in table.colnames:
            raise ConversionError("input table has no 'quat' column")

        module_relative_quats = np.array(table["quat"], copy=True)
        if module_relative_quats.ndim != 2 or module_relative_quats.shape[1] != 4:
            raise ConversionError(
                "input 'quat' must have shape (N, 4), "
                f"found {module_relative_quats.shape}"
            )

        boresight_relative_quats = qa.mult(
            self.module_offset_quat,
            module_relative_quats,
        )
        converted = table.copy(copy_data=True)
        # Returned-table convention: module_quat is relative to the module
        # center, while quat is relative to the FYST boresight.
        converted["module_quat"] = module_relative_quats
        converted["quat"] = boresight_relative_quats
        return converted
