from typing import List
from typing import Tuple
from sb.column_sort.color import EColors


class Column:
    LEN: int = 4

    def __init__(self, element_color: Tuple[EColors, EColors, EColors, EColors] = None):
        self.elements: List[EColors] = []
        if element_color is not None:
            assert len(element_color) <= self.LEN, "Too many colors"
            self.set_elements_from_tuple(element_color)
        self._check_length = True

    def get_empty_num(self) -> int:
        return self.LEN - len(self)

    def get_top_elements(self) -> Tuple[EColors, int]:
        col = EColors.NONE
        num = 0
        for ii in reversed(range(len(self))):
            if col == self.elements[ii]:
                num += 1
            else:
                col = self.elements[ii]
                num = 1
        return col, num

    def push_elements(self, added: List[EColors]):
        if self._check_length:
            assert len(added) <= self.get_empty_num(), "Too many elements, with moving. {}>{}".format(len(added), self.get_empty_num())
        self.elements = added + self.elements

    def pop_elements(self) -> List[EColors]:
        col, num = self.get_top_elements()
        for ii in range(num):
            self.elements.pop(0)
        return [col] * num

    def get_elements_to_tuple(self) -> Tuple[EColors, EColors, EColors, EColors]:
        pad = [EColors.NONE] * self.get_empty_num()
        return tuple(pad + self.elements)

    def set_elements_from_tuple(self, elements_tuple: Tuple[EColors, EColors, EColors, EColors]):
        self.elements = [ele for ele in elements_tuple if ele is not EColors.NONE]

    def is_sorted(self) -> bool:
        if len(self) == 0:
            return True
        if len(self) != self.LEN:
            return False
        return self.is_single_color()

    def is_single_color(self) -> bool:
        return len(set(self.elements)) == 1

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        if len(self) != len(other):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(self.elements))


if __name__ == "__main__":
    ecs = [
        EColors.RED,
        EColors.RED,
        EColors.BROWN
    ]
    sample_col = Column(ecs)
    print(sample_col.get_empty_num())
    print(sample_col.get_top_elements())