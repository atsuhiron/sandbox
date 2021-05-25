from typing import List
from typing import Set
from typing import Callable
from typing import Any


class ParentChild:
    def __init__(self, child, parent):
        self.child = child
        self.parent = parent

    def __eq__(self, other):
        if not isinstance(other, ParentChild):
            return False
        return self.child == other.child

    def __hash__(self):
        return hash(self.child)


class DFSResult:
    def __init__(self, node_path: List[Any], status: str, c: int):
        self.path = node_path
        self.status = status
        self.c = c

    def __str__(self):
        return str({"path": self.path, "status": self.status, "search N": self.c})


def dfs(start_node: Any,
        is_goal_func: Callable[[Any], bool],
        get_branch_func: Callable[[Any], Set[Any]],
        sort_node_func: Callable[[Any], int] = None) -> DFSResult:
    node_stack = [start_node]
    explored_nodes: Set[ParentChild] = set()

    if sort_node_func is None:
        sort_node_func = lambda x:0

    c = 0
    while len(node_stack) != 0:
        current_node = node_stack.pop()

        if is_goal_func(current_node):
            print("goal")
            break

        c += 1
        branches = get_branch_func(current_node)
        new_explored = [ParentChild(branch, current_node) for branch in branches]
        branches = set(new_explored) - explored_nodes

        explored_nodes = explored_nodes | branches
        push_stacklet = sorted([branch.child for branch in branches], key=sort_node_func)
        node_stack = node_stack + push_stacklet
    else:
        # Not Found
        print("Path is not found")
        return DFSResult([], "Path is not found", c)

    explored_dict = {pair.child: pair for pair in explored_nodes}
    node_path: List[str] = [current_node]
    while True:
        back_current = node_path[-1]
        if back_current == start_node:
            break
        its_parent = explored_dict[back_current].parent
        node_path.append(its_parent)
    return DFSResult(list(reversed(node_path)), "Found", c)


if __name__ == "__main__":
    import json
    with open("sb/column_sort/dfs.json") as f:
        pref = json.load(f)

    start_pref = "tokyo"
    goal_pref = "nagasaki"

    is_goal_func_pref = lambda a_pref: a_pref == goal_pref
    get_branch_func_pref = lambda a_pref: set(pref[a_pref])

    dfsres = dfs(start_pref, is_goal_func_pref, get_branch_func_pref)
    print(dfsres)