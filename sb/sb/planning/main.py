from typing import Dict
from typing import List
import itertools as it

from tqdm import tqdm

from plan import Plan
from plan import TimeTable

timetables = {
    "s1": TimeTable([
        Plan.parse_plan("13:30", "15:30"),
        Plan.parse_plan("18:25", "20:25"),
        Plan.parse_plan("21:05", "23:05"),
        Plan.parse_plan("09:40", "11:45"),
        Plan.parse_plan("13:10", "20:15"),
    ], "s1"),
    # "s2": TimeTable([
    #     Plan.parse_plan("09:00", "11:05"),
    #     Plan.parse_plan("14:05", "16:10"),
    #     Plan.parse_plan("16:35", "18:40"),
    #     Plan.parse_plan("21:40", "23:45"),
    #     Plan.parse_plan("12:15", "14:20"),
    # ], "s2"),
    "s3": TimeTable([
        Plan.parse_plan("11:55", "14:55"),
        Plan.parse_plan("18:20", "21:20"),
        Plan.parse_plan("14:45", "17:45"),
        Plan.parse_plan("20:40", "23:40"),
    ], "s3")
}


def find_executable_schedule(timetable_map: Dict[str, TimeTable]) -> List[TimeTable]:
    number_list = [len(timetable_map[k]) for k in timetable_map.keys()]
    all_ptn_num = 1
    for i in number_list:
        all_ptn_num *= i

    executables = []
    all_ptn = it.product(*[list(range(num)) for num in number_list])
    for ptn in tqdm(all_ptn, total=all_ptn_num):
        virtual_sche = TimeTable([timetable_map[k][i] for i, k in zip(ptn, timetable_map.keys())])
        if virtual_sche.is_executable_schedule():
            executables.append(virtual_sche)
    return executables


if __name__ == "__main__":
    ret = find_executable_schedule(timetables)
    for sche in ret:
        sche.pprint()