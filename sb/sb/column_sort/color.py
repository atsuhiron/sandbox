from enum import Enum


class EColors(Enum):
    NONE = 0
    BLUE = 1
    LIGHT_BLUE = 2
    RED = 3
    PINK = 4
    ORANGE = 5
    YELLOW = 6
    GREEN = 7
    LIGHT_GREEN = 8
    LIME_GREEN = 9
    GRAY = 10
    BROWN = 11
    PURPLE = 12

    def to_color_string(self) -> str:
        if self == self.NONE:
            return "#00000000"
        if self == self.BLUE:
            return "b"
        if self == self.LIGHT_BLUE:
            return "lightskyblue"
        if self == self.RED:
            return "r"
        if self == self.PINK:
            return "pink"
        if self == self.ORANGE:
            return "orange"
        if self == self.YELLOW:
            return "gold"
        if self == self.GREEN:
            return "g"
        if self == self.LIGHT_GREEN:
            return "mediumspringgreen"
        if self == self.LIME_GREEN:
            return "yellowgreen"
        if self == self.GRAY:
            return "lightgray"
        if self == self.BROWN:
            return "sienna"
        if self == self.PURPLE:
            return "purple"
        return "k"
