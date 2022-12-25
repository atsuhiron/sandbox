import abc
from typing import List
from typing import Tuple
import itertools as it
import tqdm

class Constraint(metaclass=abc.ABCMeta):
    def __init__(self, letter: str, loc: int):
        assert len(letter) == 1
        assert 0 <= loc <= 4

        self.letter = letter
        self.loc = loc

    @abc.abstractmethod
    def is_ok(self, word: str) -> bool:
        pass


class Yellow(Constraint):
    def is_ok(self, word: str) -> bool:
        if self.letter not in word:
            return False
        if word[self.loc] == self.letter:
            return False
        return True


class Green(Constraint):
    def is_ok(self, word: str) -> bool:
        return word[self.loc] == self.letter


def is_ok(word: str, constraints: List[Constraint]) -> bool:
    return all([constraint.is_ok(word) for constraint in constraints])


def reconstruct(letters: Tuple[str, str, str, str, str]) -> str:
    return "".join(letters)


if __name__ == "__main__":
    alp_list = list("qweyupasdfghjklcvbm")
    con_list: List[Constraint] = [
        Yellow("a", 0),
        Yellow("l", 1),
        Green("u", 2),
        Yellow("e", 4),
        Yellow("e", 3)
    ]

    cand = []
    for _letters in tqdm.tqdm(it.product(alp_list, repeat=5), total=len(alp_list)**5):
        w = reconstruct(_letters)
        if not is_ok(w, con_list):
            continue
        cand.append(w)

    print(cand)
    print("Total scan: {}".format(len(alp_list)**5))
    print("Candidate : {}".format(len(cand)))
