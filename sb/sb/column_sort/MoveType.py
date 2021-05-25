from enum import Enum


class MoveType(Enum):
    NONE = 0

    # Move type other than the two listed below.
    # Low priority
    NORMAL = 1

    # Move to an empty column.
    # Middle priority
    TO_EMPTY = 2

    # Move to a column has single color.
    # High priority
    TO_SINGLE_COLOR = 3