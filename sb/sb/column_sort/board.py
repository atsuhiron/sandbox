from typing import List
from typing import Tuple
import itertools
import collections

from sb.column_sort.color import EColors
from sb.column_sort.column import Column


class Board:
    def __init__(self, columns: List[Tuple[EColors, EColors, EColors, EColors]] = None):
        self.columns: List[Column] = []
        if columns:
            self.set_board(columns)
        self._check_can_move = True

    def initial_check(self):
        all_elements = []
        for col in self.get_board():
            all_elements += col
        counter = collections.Counter(all_elements)
        for color in counter.keys():
            if counter[color] % 4 != 0:
                print(counter)
                assert False, "Bad color number"

    def move(self, from_index: int, to_index: int):
        if self._check_can_move:
            assert self.can_move(from_index, to_index), "cannot move"
        colors = self.columns[from_index].pop_elements()
        self.columns[to_index].push_elements(colors)

    def can_move(self, from_index: int, to_index: int) -> bool:
        to_col = self.columns[to_index]
        to_color, _ = to_col.get_top_elements()
        if to_color is EColors.NONE:
            # empty column
            return True

        from_col = self.columns[from_index]
        from_color, from_num = from_col.get_top_elements()

        if from_color != to_color:
            return False

        to_room = to_col.get_empty_num()
        return to_room >= from_num

    def get_board(self) -> List[Tuple[EColors, EColors, EColors, EColors]]:
        return [col.get_elements_to_tuple() for col in self.columns]

    def set_board(self, input_columns: List[Tuple[EColors, EColors, EColors, EColors]]):
        self.columns = [Column(col) for col in input_columns]

    def list_up_all_movable_pair(self) -> List[Tuple[int, int]]:
        return [(from_index, to_index) for from_index, to_index in itertools.permutations(range(len(self)), 2) if self.can_move(from_index, to_index)]

    def is_end(self) -> bool:
        return all([col.is_sorted() for col in self.columns])

    def __len__(self) -> int:
        return len(self.columns)

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        if len(self) != len(other):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(set(self.get_board())))

    def __str__(self):
        v = [str([color_ele.value for color_ele in column]) for column in self.get_board()]
        return "\n".join(v)


if __name__ == "__main__":
    cols = [
        (EColors.RED, EColors.BLUE, EColors.RED, EColors.BLUE),
        (EColors.BLUE, EColors.RED, EColors.BLUE, EColors.RED),
        (EColors.NONE, ) * 4
    ]

    board = Board(cols)
    print("--- get_board ---")
    print(board.get_board())
    print("--- list_up_all_movable_pair ---")
    print(board.list_up_all_movable_pair())