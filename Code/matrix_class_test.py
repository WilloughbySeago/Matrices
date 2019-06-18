from Code.matrix_class import *
from Code.vector_class import *
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


