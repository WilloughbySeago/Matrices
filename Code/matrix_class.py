from Code.index_functions import *
from Code.vector_class import *
from typing import Union, List, Optional, Any, TypeVar
import math
from itertools import permutations

TNum = TypeVar("TNum", int, float, complex)


class Matrix:
    """This is a class of matrices that is definitely worse than numpy"""

    def __init__(self, rows: int, cols: int, mat: Optional[List[List[TNum]]] = None):
        """Initialise the matrix

        :param rows: int
        :param cols: int
        :param mat: list or None
        """
        # mat is the list of lists containing the data, Matrix is the object
        self.mat = mat
        if self.mat is not None:  # list provided, overwrite given rows/cols
            self.rows = len(self.mat)
            self.cols = len(self.mat[0])
        else:
            self.rows = rows
            self.cols = cols
        self._create()

    def __getattr__(self, item):
        if item == "size":
            return self.rows, self.cols

    def __repr__(self):
        string = ""
        # find the widest (in terms of characters) entry in the matrix
        max_length = 0
        for k in range(self.rows):
            for j in range(self.cols):
                x = len(str(self.mat[k][j]))
                if x > max_length:
                    max_length = x
        for k in range(self.rows):
            row = []
            for j in range(self.cols):
                element = str(self.mat[k][j])
                while len(element) < max_length:
                    element = " " + element  # add whitespace before the element
                row.append(element)  # ignore# until it is the length of the longest element
            row_string = "["
            for l in range(len(row)):
                if l < len(row) - 1:
                    row_string += row[l] + "   "  # add this as a seperator between rows
                else:
                    row_string += row[l]
            row_string += "]"
            string += row_string + "\n"
        return string

    def _create(self):
        """This is a function that will create and return a 2D list of elements

        It will be a zero matrix if no list is given or will be the list if it is given
        Do not call this method from outside the class (or even __init__) it won't do anything
        :return: Matrix
        """
        if self.mat is None:
            mat = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            return Matrix(self.rows, self.cols, mat)
        else:
            return self

    def show(self) -> None:  # deprecate
        """This is a function that prints the matrix in a nice to look at format"""
        print(repr(self))

    def dim(self) -> None:
        """This is a function that prints the dimensions of a matrix"""
        print(f"{self.rows} x {self.cols}")

    def is_square(self) -> bool:
        """This is a function that checks if the matrix is square

        :return: bool
        """
        if self.rows == self.cols:
            return True
        else:
            return False

    def add(self, other: Union[int, float, Any]):
        """This will add a scalar elementwise or add two matrices if the dimensions match

        :param other: Matrix or Scalar
        :return: Matrix
        """
        if isinstance(other, Matrix):  # matrix addition
            if self.rows == other.rows and self.cols == other.cols:  # check addition possible
                m = []
                for row in range(0, self.rows):
                    m.append([])
                    for col in range(0, self.cols):
                        m[row].append(0)
                for row in range(0, self.rows):
                    for col in range(0, self.cols):
                        x = self.mat[row][col]
                        y = other.mat[row][col]
                        m[row][col] = x + y
                return Matrix(self.rows, self.cols, m)

            else:
                raise DimensionError("Cannot add two matrices of those dimensions")

        elif type(other) in [int, float, complex]:  # scalar addition
            m = []
            for row in range(0, self.rows):
                m.append([])
                for col in range(0, self.cols):
                    m[row].append(0)
            for row in range(0, self.rows):
                for col in range(0, self.cols):
                    m[row][col] = self.mat[row][col] + other
            return Matrix(self.rows, self.cols, m)

        else:
            raise TypeError("Cannot add that type, try an integer, float, complex number or matrix")

    def mult(self, other: Union[int, float, complex, Any]):
        """This will multiply a matrix by a scalar or another matrix

        :param other: Matrix or Scalar
        :return: Matrix
        """
        if type(other) in [int, float, complex]:  # scalar multiplication
            m = []
            for row in range(self.rows):
                m.append([])
                for col in range(self.cols):
                    m[row].append(0)
            for row in range(self.rows):
                for col in range(self.cols):
                    m[row][col] = self.mat[row][col] * other
            return Matrix(self.rows, self.cols, m)

        elif isinstance(other, Matrix):  # matrix multiplication
            if self.cols != other.rows:  # check multiplication possible
                raise DimensionError("Cannot multiply two matrices of those dimensions")
            else:
                m = []
                for i in range(0, self.rows):
                    m.append([])
                    for j in range(0, other.cols):
                        m[i].append(0)
                for i in range(0, self.rows):
                    for j in range(0, other.cols):
                        a = Vector(self.mat[i])  # create vector from row of self
                        b = []
                        for k in range(0, other.rows):
                            b.append(other.mat[k][j])
                        b = Vector(b)  # create vector from column of other
                        c = a.dot_prod(b)
                        m[i][j] = c
                return Matrix(self.rows, other.cols, m)

        else:
            raise TypeError("""Cannot multiply by that type, try an integer, float, complex number or matrix""")

    def cofactor(self, row: int, col: int) -> Union[int, float, complex]:
        """This is a function that will return the cofactor of a given spot in a matrix

        :param row: int
        :param col: int
        :return: Scalar
        """
        return (-1) ** (col - 1 + row - 1) * self.mat[row - 1][col - 1]

    def minor(self, row: int, col: int):
        """This is a function that will return the minor of a given spot in a matrix

        :param row: int
        :param col: int
        :return: Scalar
        """
        m = []
        for i in range(0, self.rows):
            if i != row - 1:
                m.append([])
                for j in range(0, self.cols):
                    if j != col - 1:
                        if i < row:
                            m[i].append(self.mat[i][j])
                        else:
                            m[i - 1].append(self.mat[i][j])
        m = Matrix(self.rows - 1, self.cols - 1, m)
        return m.det()

    def det(self):
        """This is a function that will return the determinant of an n x n matrix

        :return: scalar
        """
        if not self.is_square():
            # determinant only defined for square matrices
            raise DimensionError(f"Tried to take the determinant of a non-square matrix:\n{str(self)}")
        lst = list(range(1, self.cols + 1))
        total = 0
        for perm in permutations(lst):
            # for each permutation calculate the value to add to the sum
            current_product = 1
            perm_list = list(perm)
            current_product *= levi_civita(perm_list)
            for i in range(self.cols):
                current_product *= self.mat[i][perm_list[i] - 1]
            total += current_product
        return total

    def transpose(self):
        """This is a function that will return the transpose of a matrix

        :return: Matrix
        """
        m = []
        for i in range(self.cols):
            m.append([])
            for j in range(self.rows):
                m[i].append(self.mat[j][i])
        return Matrix(self.rows, self.cols, m)

    def hermitian_transpose(self):
        """This function returns the Hermitian transpose (aka Hermitian conjugate) of the matrix

        :return: Matrix
        """
        m = []
        for i in range(self.cols):
            m.append([])
            for j in range(self.rows):
                element = self.mat[j][i]
                if type(element) == complex:
                    element = element.conjugate()
                m[i].append(element)
        return Matrix(self.rows, self.cols, m)

    def inverse(self):
        """This function returns the inverse of a matrix

        if the matrix isn't square you will get a DimensionError
        if the determinant is zero there is no inverse so None is returned
        :return: Matrix or None
        """
        # check conditions for inverse to exist
        if not self.is_square():
            raise DimensionError(f"No inverse for a {self.dim()} matrix:\n{self}")
        det = self.det()
        if det == 0:
            return None
        # inverse exists if you get this far

        # create blank matrix of 0s with same dimensions to fill in values later
        inv_mat = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        inverse = Matrix(self.rows, self.cols, inv_mat)

        # calculate each individual value
        for i in range(1, self.rows + 1):
            for j in range(1, self.cols + 1):
                minor = self.minor(j, i)
                sign = 1 if ((i + j) % 2 == 0) else -1
                element = sign * minor / det
                inverse.mat[i - 1][j - 1] = element
        return inverse

    def reshape(self, row: int, col: int):
        """This is a function that takes a matrix and then reshapes it to the given dimensions

        :param row: int
        :param col: int
        :return: Matrix
        """
        components = self.to_list()
        old_size = len(components)
        new_size = row * col
        if old_size == new_size:  # check that reshape is possible
            m = []
            index = 0
            for i in range(row):
                m.append([])
                for j in range(col):
                    m[i].append(components[index])
                    index += 1
            return Matrix(row, col, m)
        else:
            if old_size > new_size:
                raise DimensionError("There are more components than fit in a matrix of that size")
            else:
                raise DimensionError("There are not enough components to fill a matrix of this size")

    def to_list(self) -> List[Union[int, float, complex, Any]]:
        """This is a function that returns a list of all components of matrix

        :return: list
        """
        components = []
        for i in range(self.rows):
            for j in range(self.cols):
                components.append(self.mat[i][j])
        return components

    def elementwise(self, function, other: Optional[Any] = None):
        """This is a function that applies a function either to a matrix or between two matrices elementwise

        To apply a function with a scalar variable pass that variable to the function yourself
        :param function: function
        :param other: Matrix
        :return: Matrix
        """
        if other is None:  # one variable (the element from the matrix) function
            m = []
            for i in range(self.rows):
                m.append([])
                for j in range(self.cols):
                    m[i].append(function(self.mat[i][j]))
            return Matrix(self.rows, self.cols, m)
        else:  # two variable (one element from each matrix) function
            if self.rows == other.rows and self.cols == other.cols:
                m = []
                for i in range(self.rows):
                    m.append([])
                    for j in range(self.cols):
                        a = self.mat[i][j]
                        b = other.mat[i][j]
                        m[i].append(function(a, b))
                return Matrix(self.rows, self.cols, m)
            else:
                raise DimensionError("These matrices are of different dimensions")

    def mat_mean(self) -> Union[int, float, complex]:
        """This is a function that will return the mean of all elements in the matrix

        :return: scalar"""
        total = self.mat_sum()
        return total / (self.rows * self.cols)

    def mat_sum(self) -> Union[int, float, complex]:
        """This is a function that will return the sum of all elemets in the matrix

        :return: Scalar"""
        total = 0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.mat[i][j]
        return total

    def to_vec(self):
        """Takes a matrix (1 x n or n x 1) and returns a vector

        :return: Vector
        """
        if self.rows == 1:
            v = self.mat[0]
            return Vector(v)
        elif self.cols == 1:
            v = []
            for i in range(self.rows):
                v.append(self.mat[i][0])
            return Vector(v)
        else:
            raise DimensionError(f"Cannot make a vector of those dimensions: {self.rows} x {self.cols}")

    def mat_is_equal(self, other):
        """This function returns true if self and other are the same matrix (not the same object)

        :return: bool
        """
        if not isinstance(other, Matrix):
            return False
        if (self.rows, self.cols) == (other.rows, other.cols):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.mat[i][j] != other.mat[i][j]:
                        return False
            return True
        return False

    def trace(self):
        """This function returns the trace (sum of the leading diagonal) of a matrix

        :return: float
        """
        if not self.is_square():
            raise DimensionError(f"Trace is only defined for a square matrix. This matrix is {self.rows}x{self.cols}")
        total = 0
        for i in range(self.rows):
            total += self.mat[i][i]
        return total

    def __add__(self, other):
        """Matrix or scalar addition"""
        return self.add(other)

    def __radd__(self, other):
        """Matrix or scalar addition"""
        return self.add(other)

    def __sub__(self, other):
        """Matrix or scalar subtraction"""
        if isinstance(other, Matrix):
            neg_other = other.mult(-1)
            return self.add(neg_other)
        elif type(other) in [int, float, complex]:
            return self.add(-other)

    def __rsub__(self, other):
        """Matrix or scalar subtraction"""
        if isinstance(other, Matrix):
            neg_other = other.mult(-1)
            return self.add(neg_other)
        elif type(other) in [int, float, complex]:
            return self.add(-other)

    def __mul__(self, other):
        """Scalar or elementwise multiplication"""
        if type(other) in [int, float, complex]:
            return self.mult(other)
        else:
            raise TypeError(f"Unsupported operation '*' between types '{type(self)}' and '{type(other)}'")

    def __rmul__(self, other):
        """Scalar or elementwise multiplication"""
        if type(other) in [int, float, complex]:
            return self.mult(other)
        else:
            raise TypeError(f"Unsupported operation '*' between types '{type(self)}' and '{type(other)}'")

    def __matmul__(self, other):
        """Matrix multiplication"""
        return self.mult(other)

    def __truediv__(self, other):
        """Scalar division"""
        if type(other) in [int, float, complex]:
            inverse = 1 / other
            return self.mult(inverse)
        else:
            raise TypeError(f"Unsupported operation '/' between types '{type(self)}' and '{type(other)}'")

    def __eq__(self, other):
        """Test for equality"""
        if not isinstance(other, Matrix):
            return False
        return self.mat == other.mat

    def __ne__(self, other):
        """Test for inequality"""
        return not self == other

    def __neg__(self):
        """Negate every element of the matrix"""
        return self.mult(-1)


def identity(dimension: int):
    """This function returns dimension x dimension identity matrix

    :param dimension: int
    :return: Matrix
    """
    m = []
    for i in range(dimension):
        m.append([])
        for j in range(dimension):
            if i == j:
                m[i].append(1)
            else:
                m[i].append(0)
    return Matrix(dimension, dimension, m)


def from_range(rows: int, cols: int, start=0):
    """This function returns a matrix of the given dimensions where each element is just the next in range(maximum)

    :param rows: int
    :param cols: int
    :param start: int
    :return: Matrix
    """
    maximum = rows * cols
    components = [x for x in range(start, start + maximum)]
    return from_list(components, rows, cols)


def from_list(components: List, rows: int, cols: int):
    """This is a function that creates a matrix of given dimensions from a list

    :param components: list
    :param rows: int
    :param cols: int
    :return: Matrix
    """
    length_components = len(components)
    size_matrix = rows * cols
    if length_components == size_matrix:
        m = []
        index = 0
        for i in range(rows):
            m.append([])
            for j in range(cols):
                m[i].append(components[index])
                index += 1
        return Matrix(rows, cols, m)
    else:
        if length_components > size_matrix:
            raise DimensionError("There are more components than fit in this matrix")
        else:
            raise DimensionError("There aren't enough components to fill this matrix")


def from_vec(vec):
    """Takes a vector and returns a matrix

    :param vec: Vector
    :return: Matrix
    """
    lst = vec.vec
    return from_list(lst, len(lst), 1)


def rotate(angle: Union[int, float], axis: str = "z", dim: int = 3):
    """This is a function which returns a rotation matrix

    :param angle: real number
    :param axis: 'x', 'y' or 'z'
    :param dim: int (2 or 3)
    :return: Matrix
    """
    c = round(math.cos(angle), 3)
    s = round(math.sin(angle), 3)
    if axis == "z" and dim == 2:
        return from_list([c, s, -s, c], 2, 2)
    elif axis == "x" and dim == 3:
        return from_list([1, 0, 0, 0, c, s, 0, -s, c], 3, 3)
    elif axis == "y" and dim == 3:
        return from_list([c, 0, -s, 0, 1, 0, s, 0, c], 3, 3)
    elif axis == "z" and dim == 3:
        return from_list([c, s, 0, -s, c, 0, 0, 0, 1], 3, 3)


"""Below here is testing"""  # -----------------------------------------------------------------------------------------


def main():
    m = Matrix(6, 6, [[8+2j, 7-1j, 9, 6, 4, 2],
                      [9, 1, 6, 9, 0, 2],
                      [6, 9, 3, 3, 7, 3],
                      [7, 0, 1, 4, 3, 5],
                      [6, 4, 5, 5, 2, 3],
                      [0, 4, 2, 4, 5, 4]])
    print(m.hermitian_transpose())


if __name__ == "__main__":
    main()

# TODO:
#  Add unit tests for new operator definitions (matrices and vectors) and trace
#  Add type/dimension tests in definitions of operators where needed
