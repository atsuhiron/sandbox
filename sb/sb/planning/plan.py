from __future__ import annotations
import datetime as dt
import dataclasses
from typing import List
from typing import Optional


def _to_datetime(t: dt.time, y: int = 0, m: int = 0, d: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, t.hour, t.minute, t.second, t.microsecond)


def _short_str(time: dt.time) -> str:
    return f"{str(time.hour).zfill(2)}:{str(time.minute).zfill(2)}"

@dataclasses.dataclass
class Plan:
    start: dt.time
    end: dt.time
    title: str = None

    def is_after(self, that: Plan) -> bool:
        return self.start > that.end

    def is_before(self, that: Plan) -> bool:
        return self.end < that.start

    @staticmethod
    def parse_plan(s: str, e: str) -> Plan:
        start_hd = s.split(":")
        assert len(start_hd) == 2
        end_hd = e.split(":")
        assert len(end_hd) == 2
        return Plan(dt.time(int(start_hd[0]), int(start_hd[1])), dt.time(int(end_hd[0]), int(end_hd[1])))


class TimeTable:
    def __init__(self, plans: List[Plan], title: str = None):
        self.plans = sorted(plans, key=lambda pl: pl.start)
        self.title = title
        if self.title is not None:
            for plan in self.plans:
                plan.title = self.title

    def find_future_nearest_plan(self, target_time: dt.time) -> Optional[Plan]:
        for pl in self.plans:
            if pl.start > target_time:
                return pl
        return None

    def calc_whole_time(self) -> dt.timedelta:
        return _to_datetime(self.plans[-1].end) - _to_datetime(self.plans[0].start)

    def is_executable_schedule(self) -> bool:
        if len(self.plans) <= 1:
            return True

        for i, pl in enumerate(self.plans):
            if i == 0:
                continue
            if pl.is_after(self.plans[i - 1]):
                continue
            return False
        return True

    def pprint(self):
        print(f"Title: {self.title}")
        for plan in self.plans:
            print(f"{_short_str(plan.start)} ~ {_short_str(plan.end)} {plan.title}")

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, item) -> Plan:
        return self.plans[item]


if __name__ == "__main__":
    t1 = dt.time(13, 8)
    t2 = dt.time(12, 12)
    p1 = Plan(t1, t1)
    p2 = Plan(t2, t2)
    tt = TimeTable([p1, p2])
    print(tt.plans)
    print(p1.is_before(p2))

