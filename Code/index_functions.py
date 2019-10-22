from typing import List


def delta(i: int, j: int) -> int:
    """This is the Kronecker delta in 2D

    :param i: int
    :param j: int
    :return: int
    """
    if i == j:
        return 1
    else:
        return 0


def levi_civita(lst):
    """This is the Levi Civita symbol for any number of dimensions

    returns 1 for an even permutation, -1 for an odd permutation and 0 if an index is repeated
    :param lst: list length n containing only the elements in {1, 2, 3,..., n}
    :return: -1, 0, 1
    """
    # check for repeated indices
    for i in range(1, len(lst) + 1):
        count = lst.count(i)
        if count != 1:
            return 0
    parity = 1
    # arrays start at 0, indices start at 1 so subtract 1 from all values
    lst = [i - 1 for i in lst]
    for index in range(len(lst) - 1):
        # check if the item is in the right place
        if index != lst[index]:
            parity *= -1
            # swap to where it should be
            new = lst.index(index)
            lst[index], lst[new] = lst[new], lst[index]
    return parity
