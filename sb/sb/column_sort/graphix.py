from typing import Tuple
from typing import List
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from sb.column_sort.board import Board
from sb.column_sort.column import Column

IntPair = Tuple[int, int]


class PlotConst:
    y_pitch = 0.1
    col_top_y = (1, 0.5)
    plot_args = {"marker": "o", "markersize": 20}


def show_board(boards: List[Board]):
    fig = plt.figure()

    # plot board
    ax_board = fig.add_axes([0.1, 0, 0.8, 0.8])
    nums = _calc_ul_column_num(len(boards[0]))
    num_offsets = (0, nums[0])
    path_index: int = 0
    path_max_index = len(boards) - 1

    plot_board(ax_board, boards[path_index], nums, num_offsets, path_index)

    # draw button
    ax_btn_back = fig.add_axes([0.1, 0.02, 0.4, 0.03])
    button_back = Button(ax_btn_back, "<", color="limegreen", hovercolor='0.9')
    ax_btn_prog = fig.add_axes([0.5, 0.02, 0.4, 0.03])
    button_prog = Button(ax_btn_prog, ">", color="limegreen", hovercolor='0.9')

    def update_back(event):
        nonlocal path_index
        if path_index <= 0:
            return
        path_index -= 1

        plot_board(ax_board, boards[path_index], nums, num_offsets, path_index)
        fig.canvas.draw_idle()

    def update_prog(event):
        nonlocal path_index
        if path_index >= path_max_index:
            return
        path_index += 1

        plot_board(ax_board, boards[path_index], nums, num_offsets, path_index)
        fig.canvas.draw_idle()

    button_back.on_clicked(update_back)
    button_prog.on_clicked(update_prog)
    plt.show()


def plot_board(ax_board: plt.Axes, board: Board, nums: IntPair, num_offsets: IntPair, index: int):
    ax_board.clear()
    ax_board.set_title(str(index))
    for num, offset, top_y in zip(nums, num_offsets, PlotConst.col_top_y):
        # for loop by raw

        col_top_x = np.linspace(0, 1, num + 2)[1:-1]
        for cc in range(num):
            # flor loop by column

            column = board.get_board()[cc + offset]
            for ee in range(Column.LEN):
                ax_board.plot([col_top_x[cc]], [top_y - ee*PlotConst.y_pitch],
                              color=column[ee].to_color_string(), **PlotConst.plot_args)
    ax_board.set_xlim([0, 1.1])
    ax_board.set_ylim([0, 1.1])
    ax_board.axis("off")


def _calc_ul_column_num(column_num: int) -> IntPair:
    if column_num <= 5:
        return column_num, 0
    if column_num % 2 == 0:
        return int(column_num / 2), int(column_num / 2)
    return int(column_num / 2) + 1, int(column_num / 2)
