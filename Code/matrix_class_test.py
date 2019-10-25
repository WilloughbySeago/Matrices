from Code.matrix_class import *
# from Code.vector_class import *
from Code.errors import *
import io
import sys
import math


# class methods
def test_can_create_matrix():
    mat = Matrix(1, 1)
    assert isinstance(mat, Matrix)


def test_can_create_matrix_with_list():
    mat = Matrix(1, 1, [[1]])
    assert isinstance(mat, Matrix)


def test_can_show_2by2():
    m = Matrix(2, 2, [[1, 1], [1, 1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '[1   1]\n[1   1]\n\n'


def test_can_show_1by1():
    m = Matrix(1, 1, [[1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '[1]\n\n'


def test_repr_2by2():
    m = Matrix(2, 2, [[1, 1], [1, 1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '[1   1]\n[1   1]\n\n'


def test_repr_1by1():
    m = Matrix(1, 1, [[1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '[1]\n\n'


def test_dim():
    m = Matrix(1, 1, [[1]])
    output = io.StringIO()
    sys.stdout = output
    m.dim()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '1 x 1\n'


def test_is_square_true():
    m = Matrix(2, 2)
    assert m.is_square()


def test_is_square_false():
    m = Matrix(1, 2)
    assert not m.is_square()


def test_add_scalar():
    m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m = m.add(1)
    assert m.mat == Matrix(2, 3, [[2, 3, 4], [5, 6, 7]]).mat


def test_add_matrix():
    m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(2, 3, [[5, 3, 4], [6, 1, 7]])
    m = m1.add(m2)
    assert m.mat == Matrix(2, 3, [[6, 5, 7], [10, 6, 13]]).mat


def test_add_dim_error():
    m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(3, 2)
    try:
        m1.add(m2)
    except DimensionError:
        assert True
    else:
        assert False


def test_mult_scalar():
    m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m = m.mult(2)
    assert m.mat == Matrix(2, 3, [[2, 4, 6], [8, 10, 12]]).mat


def test_mult_matrix():
    m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(3, 2, [[3, 4], [5, 1], [6, 1]])
    m = m1.mult(m2)
    assert m.mat == Matrix(2, 2, [[31, 9], [73, 27]]).mat


def test_mult_dim_error():
    m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    m2 = Matrix(2, 2, [[3, 4], [5, 1]])
    try:
        m1.mult(m2)
    except DimensionError:
        assert True
    else:
        assert False


def test_two_by_two_det():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    det = m1.two_x_two_det()
    assert det == -2


def test_two_by_two_det_not_two_by_two_arg():
    m = Matrix(3, 2)
    try:
        m.two_x_two_det()
    except DimensionError:
        assert True
    else:
        assert False


def test_cofactor():
    m = Matrix(2, 2, [[1, 2], [3, 4]])
    cofactor = m.cofactor(1, 2)
    assert cofactor == -2


def test_minor():
    m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    minor = m.minor(2, 3)
    assert minor.mat == Matrix(2, 2, [[1, 2], [7, 8]]).mat


def test_transpose():
    m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    transpose = m.transpose()
    assert transpose.mat == Matrix(3, 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]).mat


def test_reshape():
    m = Matrix(3, 2, [[1, 2], [4, 5], [7, 8]])
    reshaped = m.reshape(2, 3)
    assert reshaped.mat == Matrix(2, 3, [[1, 2, 4], [5, 7, 8]]).mat


def test_to_list():
    m = Matrix(3, 2, [[1, 2], [4, 5], [7, 8]])
    l = m.to_list()
    assert l == [1, 2, 4, 5, 7, 8]


def test_elementwise_1_var():
    m = Matrix(3, 2, [[1, 2], [4, 5], [7, 8]])
    f = lambda x: x ** 2
    result = m.elementwise(f)
    assert result.mat == Matrix(3, 2, [[1, 4], [16, 25], [49, 64]]).mat


def test_elementwise_2_var():
    m1 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m2 = Matrix(3, 3, [[6, 3, 7], [1, 7, 2], [8, 2, 3]])
    f = lambda x, y: x + 2 * y
    m = m1.elementwise(f, m2)
    assert m.mat == Matrix(3, 3, [[13, 8, 17], [6, 19, 10], [23, 12, 15]]).mat


def test_mat_mean():
    m = Matrix(2, 2, [[1, 2], [3, 4]])
    mean = m.mat_mean()
    assert mean == 2.5


def test_mat_sum():
    m = Matrix(2, 2, [[1, 2], [3, 4]])
    total = m.mat_sum()
    assert total == 10


def test_to_vec():
    m = Matrix(1, 2, [[1, 2]])
    assert m.to_vec().vec == [1, 2]


def test_mat_is_equal_true():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 4]])
    assert m1.mat_is_equal(m2)


def test_mat_is_equal_false():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 5]])
    assert not m1.mat_is_equal(m2)


def test_trace():
    m1 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert m1.trace() == 15


def test_trace_dim_error():
    m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    try:
        m1.trace()
    except DimensionError:
        assert True
    else:
        assert False


def test_add_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[5, 6], [7, 8]])
    m = m1 + m2
    assert m.mat == Matrix(2, 2, [[6, 8], [10, 12]]).mat


def test_sub_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[5, 6], [7, 8]])
    m = m1 - m2
    assert m.mat == Matrix(2, 2, [[-4, -4], [-4, -4]]).mat


def test_sub_scalar():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m = m1 - 5
    assert m.mat == Matrix(2, 2, [[-4, -3], [-2, -1]]).mat


def test_mul_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m = m1 * 2
    assert m.mat == Matrix(2, 2, [[2, 4], [6, 8]]).mat


def test_mul_type_error():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[5, 6], [7, 8]])
    try:
        m = m1 * m2
    except TypeError:
        assert True
    else:
        assert False


def test_matmul_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[5, 6], [7, 8]])
    m = m1 @ m2
    assert m.mat == Matrix(2, 2, [[19, 22], [43, 50]]).mat


def test_truediv_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m = m1 / 2
    assert m.mat == Matrix(2, 2, [[0.5, 1], [1.5, 2]]).mat


def test_truediv_type_error():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[5, 6], [7, 8]])
    try:
        m = m1 / m2
    except TypeError:
        assert True
    else:
        assert False


def test_eq_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 4]])
    assert m1 == m2


def test_eq_false():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 5]])
    assert not m1 == m2


def test_ne_false():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 4]])
    assert not m1 != m2


def test_ne_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m2 = Matrix(2, 2, [[1, 2], [3, 5]])
    assert m1 != m2


def test_neg_operator():
    m1 = Matrix(2, 2, [[1, 2], [3, 4]])
    m = -m1
    assert m.mat == Matrix(2, 2, [[-1, -2], [-3, -4]]).mat


def test_size():
    m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
    size = m1.size()
    assert size == (3, 2)


# static methods
def test_identity():
    i = identity(3)
    assert i.mat == Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).mat


def test_from_range():
    m = from_range(2, 3, 4)
    assert m.mat == Matrix(2, 3, [[4, 5, 6], [7, 8, 9]]).mat


def test_from_list():
    m = from_list([1, 2, 3, 4, 5, 6], 2, 3)
    assert m.mat == Matrix(2, 3, [[1, 2, 3], [4, 5, 6]]).mat


def test_from_list_dim_error():
    try:
        from_list([1, 2, 3], 2, 2)
    except DimensionError:
        assert True
    else:
        assert False


def test_rotate():
    assert rotate(math.pi).mat == Matrix(3, 3, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).mat
