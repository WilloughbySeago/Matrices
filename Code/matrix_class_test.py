from Code.matrix_class import *
# from Code.vector_class import *
from Code.errors import *
import io
import sys


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
    assert output.getvalue() == '\n[1   1]\n[1   1]\n\n'


def test_can_show_1by1():
    m = Matrix(1, 1, [[1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '\n[1]\n\n'


def test_repr_2by2():
    m = Matrix(2, 2, [[1, 1], [1, 1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '\n[1   1]\n[1   1]\n\n'


def test_repr_1by1():
    m = Matrix(1, 1, [[1]])
    output = io.StringIO()
    sys.stdout = output
    m.show()
    sys.stdout = sys.__stdout__
    print(output.getvalue())
    assert output.getvalue() == '\n[1]\n\n'


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


def test_two_by_tw0_det_not_two_by_two_arg():
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
