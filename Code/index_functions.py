def delta(i, j):
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


def levi_civita(*args):
    """
    This is the Levi-Civita symbol for 2 or 3 variables
    :param args: int
    :return: int
    """
    if len(args) == 3:
        i = args[0]
        j = args[1]
        k = args[2]
        if (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
            return 1
        elif (i, j, k) in [(3, 2, 1), (2, 1, 3), (1, 3, 2)]:
            return -1
        else:
            return 0
    elif len(args) == 2:
        i = args[0]
        j = args[1]
        if (i, j) == (1, 2):
            return 1
        elif (i, j) == (2, 1):
            return -1
        else:
            return 0
