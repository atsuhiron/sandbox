from typing import List
from typing import Tuple
from typing import Set

from sb.column_sort.color import EColors
from sb.column_sort.board import Board
import sb.column_sort.graphix as graphix
import sb.column_sort.depth_first_search as dfs

import importlib
importlib.reload(graphix)


def get_branch(board: Board) -> Set[Board]:
    movable_list: List[Tuple[int, int]] = board.list_up_all_movable_pair()
    new_boards = []
    for movable_pair in movable_list:
        new_board = Board()
        new_board.set_board(board.get_board())
        new_board.move(*movable_pair)
        new_boards.append(new_board)
    return set(new_boards)


def is_end(board: Board) -> bool:
    return board.is_end()


if __name__ == "__main__":
    cols1 = [
        (EColors.ORANGE, EColors.ORANGE, EColors.RED, EColors.BLUE),
        (EColors.RED, EColors.BLUE, EColors.RED, EColors.BLUE),
        (EColors.ORANGE, EColors.RED, EColors.BLUE, EColors.ORANGE),
        (EColors.NONE,) * 4,
        (EColors.NONE,) * 4
    ]
    cols2 = [
        (EColors.GREEN, EColors.BLUE, EColors.PINK, EColors.RED),
        (EColors.LIME_GREEN, EColors.ORANGE, EColors.LIME_GREEN, EColors.LIGHT_GREEN),
        (EColors.GRAY, EColors.LIGHT_GREEN, EColors.BROWN, EColors.PINK),
        (EColors.PURPLE, EColors.BLUE, EColors.GREEN, EColors.BROWN),
        (EColors.BROWN, EColors.GREEN, EColors.RED, EColors.LIGHT_GREEN),
        (EColors.LIGHT_BLUE, EColors.YELLOW, EColors.GRAY, EColors.ORANGE),
        (EColors.PURPLE, EColors.RED, EColors.YELLOW, EColors.YELLOW),

        (EColors.GREEN, EColors.BLUE, EColors.PURPLE, EColors.LIGHT_GREEN),
        (EColors.PURPLE, EColors.LIME_GREEN, EColors.LIME_GREEN, EColors.BLUE),
        (EColors.ORANGE, EColors.LIGHT_BLUE, EColors.LIGHT_BLUE, EColors.GRAY),
        (EColors.BROWN, EColors.PINK, EColors.RED, EColors.GRAY),
        (EColors.PINK, EColors.LIGHT_BLUE, EColors.YELLOW, EColors.ORANGE),
        (EColors.NONE,) * 4,
        (EColors.NONE,) * 4
    ]
    origin_board = Board(cols2)
    origin_board.initial_check()

    res = dfs.dfs(origin_board, is_end, get_branch)

    graphix.show_board(res.path)
