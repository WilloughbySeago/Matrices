from typing import List


def delta(i: int, j: int) -> int:
    """
    This is the Kronecker delta in 2D
    :param i: int
    :param j: int
    :return: int
    """
    if i == j:
        return 1
    else:
        return 0


def levi_civita(lst):
    for i in range(1, len(lst) + 1):
        count = lst.count(i)
        if count != 1:
            return 0
    parity = 1
    lst = [i - 1 for i in lst]  # arrays start at 0, indices start at 1 so subtract 1 from all values
    for index in range(len(lst) - 1):
        if index != lst[index]:
            parity *= -1
            new = lst.index(index)
            lst[index], lst[new] = lst[new], lst[index]
    return parity
