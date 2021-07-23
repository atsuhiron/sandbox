from typing import Tuple
import dataclasses
import enum


class WallType(enum.Enum):
    # Everything can not leak.
    CLOSE = 0

    # Emission can leak, but particle can not.
    PERMEATION = 2

    # Any particle and emission can leak
    OPEN = 1


@dataclasses.dataclass
class FieldProfile:
    field_shape: Tuple[int, int]
    wall_type_top: WallType = WallType.CLOSE
    wall_type_bottom: WallType = WallType.CLOSE
    wall_type_left: WallType = WallType.CLOSE
    wall_type_right: WallType = WallType.CLOSE
