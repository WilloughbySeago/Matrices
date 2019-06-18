from Code.matrix_class import *
# from Code.vector_class import *
# from Code.errors import *
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
