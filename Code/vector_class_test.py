from Code.vector_class import *
from Code.errors import *


def test_can_create_vector():
    v = Vector([])
    assert isinstance(v, Vector)


def test_dot_prod():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert v1.dot_prod(v2) == 32


def test_dot_prod_fail_dim():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6, 7])
    try:
        v1.dot_prod(v2)
    except DimensionError:
        assert True
    else:
        assert False


def test_dot_prod_fail_type():
    v1 = Vector([1, 2, 3])
    try:
        v1.dot_prod(2)
    except TypeError:
        assert True
    else:
        assert False


def test_cross_prod():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    v = v1.cross_prod(v2)
    assert v.vec == [-3, 6, -3]


def test_cross_prod_fail_dim():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6, 7])
    try:
        v1.cross_prod(v2)
    except DimensionError:
        assert True
    else:
        assert False


def test_cross_prod_fail_type():
    v1 = Vector([1, 2, 3])
    try:
        v1.cross_prod(1)
    except TypeError:
        assert True
    else:
        assert False


def test_vec_add():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    v = v1.vec_add(v2)
    assert v.vec == [5, 7, 9]


def test_vec_add_fail_dim():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6, 7])
    try:
        v1.vec_add(v2)
    except DimensionError:
        assert True
    else:
        assert False


def test_vec_add_fail_type():
    v1 = Vector([1, 2, 3])
    try:
        v1.vec_add(1)
    except TypeError:
        assert True
    else:
        assert False


def test_vec_mult():
    v1 = Vector([1, 2, 3])
    v = v1.vec_mult(2)
    assert v.vec == [2, 4, 6]


def test_vec_mult_fail_type():
    v1 = Vector([1, 2, 3])
    try:
        v1.vec_mult('non scalar')
    except TypeError:
        assert True
    else:
        assert False


def test_mag():
    v1 = Vector([3, 4])
    assert v1.mag() == 5


def test_unit():
    v1 = Vector([2, 0, 0])
    assert v1.unit().vec == [1, 0, 0]


def test_angle():
    v1 = Vector([1, 0, 0])
    v2 = Vector([0, 1, 0])
    assert v1.angle(v2) == math.pi / 2


def test_elementwise():
    f = lambda x: 2 * x
    v1 = Vector([1, 2, 3])
    v = v1.elementwise(f)
    assert v.vec == [2, 4, 6]


def test_vec_mean():
    v1 = Vector([1, 2, 3])
    mean = v1.vec_mean()
    assert mean == 2


def test_vec_sum():
    v1 = Vector([1, 2, 3])
    assert v1.vec_sum() == 6


def test_vec_is_equal_true():
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 3])
    assert v1.vec_is_equal(v2)


def test_vec_is_equal_false_vec_diff():
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 4])
    assert not v1.vec_is_equal(v2)


def test_vec_is_equal_false_len_diff():
    v1 = Vector([1, 2, 3, 4])
    v2 = Vector([1, 2, 3])
    assert not v1.vec_is_equal(v2)


def test_vec_is_equal_false_type_diff():
    v1 = Vector([1, 2, 3])
    assert not v1.vec_is_equal(1)
