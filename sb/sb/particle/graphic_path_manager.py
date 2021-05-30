import os
import shutil
import glob
from typing import List
import subprocess


class GraphicPathManager:
    BASE_PATH = "sb/particle/"
    F_PATH = BASE_PATH + "images/frames/"
    M_PATH = BASE_PATH + "images/movies/"
    IMG_EXT = "png"
    MOV_EXT = "mp4"
    F_NAME = "frame_{}." + IMG_EXT

    def __init__(self, name: str, frame_num_order: int = 5):
        self.name = name
        self.frame_dir = self.F_PATH + self.name + "/"

        os.makedirs(self.frame_dir, exist_ok=True)

        self.max_num = 10 ** frame_num_order - 1
        self.order = frame_num_order

    def remove_frame_dir(self):
        shutil.rmtree(self.frame_dir)

    def has_frames(self) -> bool:
        return bool(self.get_all_exist_frame_path())

    def get_frame_path(self, index: int) -> str:
        if index > self.max_num:
            raise IndexError("Max index")
        index_str = str(index).zfill(self.order)
        return self.frame_dir + self.F_NAME.format(index_str)

    def get_mov_path(self) -> str:
        return "{}{}.{}".format(self.M_PATH, self.name, self.MOV_EXT)

    def get_all_exist_frame_path(self) -> List[str]:
        return glob.glob(self.frame_dir + "*." + self.IMG_EXT)

    def gen_mov(self, in_fps: int = 30, out_fps: int = 30):
        com_temp = "ffmpeg -r {0} -i {1} -vcodec libx264 -pix_fmt yuv420p -r {2} {3}"
        com = com_temp.format(in_fps,
                              self.frame_dir + self.F_NAME.format("%0" + str(self.order) + "d"),
                              out_fps,
                              self.get_mov_path())
        print(com)
        subprocess.call(com)
