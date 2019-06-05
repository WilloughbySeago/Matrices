from Code.errors import *
from Code.vector_class import *


class Matrix:
    """This is a class of matrices that is definetly worse than numpy"""

    def __init__(self, rows, cols, mat=[]):
        """
        Initialise the matrix
        :param rows: int
        :param cols: int
        :param mat: list
        """
        self.rows = rows
        self.cols = cols
        # mat is the list of lists containing the data, matrix is the object
        self.mat = mat
        if self.mat:
            self.rows = len(self.mat)
            self.cols = len(self.mat[0])
        self.create()

    def create(self):
        """
        This is a function that will create and return a matrix
        It will be a zero matrix if no list is given or will be the list if it is given
        :return: Matrix
        """
        if not self.mat:
            mat = []
            for row in range(0, self.rows):
                mat.append([])
                i = 0
                while i < self.cols:
                    mat[row].append(0)
                    i += 1
            return Matrix(self.rows, self.cols, mat)
        else:
            return self

    def show(self):
        """
        This is a function that prints the matrix in a nice to look at format
        :return: None
        """
        print('')
        max_length = 0
        for i in range(self.rows):
            for j in range(self.cols):
                x = len(str(self.mat[i][j]))
                if x > max_length:
                    max_length = x
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                element = str(self.mat[i][j])
                while len(element) < max_length:
                    element = ' ' + element
                row.append(element)
            strng = '['
            for i in range(len(row)):
                if i < len(row) - 1:
                    strng += row[i] + '   '
                else:
                    strng += row[i]
            strng += ']'
            print(strng)
        print('')

    def dim(self):
        """
        This is a function that prints the dimensions of a matrix
        :return: None
        """
        print(f"{self.rows} x {self.cols}")

    def is_square(self):
        """
        This is a function that checks if the matrix is square
        :return: bool
        """
        if self.rows == self.cols:
            return True
        else:
            return False

    def add(self, other):
        """
        This will add a scalar elementwise or add two matrices if the dimensions match
        :param other: Matrix or Scalar
        :return: Matrix
        """
        if type(other) == Matrix:
            if self.rows == other.rows and self.cols == other.cols:
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
                error("Cannot add two matrices of those dimensions")
                raise DimensionError

        elif type(other) in [int, float, complex]:
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
            error("Cannot add that type, try an integer, float, complex number or matrix")
            raise TypeError

    def mult(self, other):
        """
        This will multiply a matrix by a scalar or another matrix
        :param other: Matrix or Scalar
        :return: Matrix
        """
        if type(other) in [int, float, complex]:
            m = []
            for row in range(0, self.rows):
                m.append([])
                for col in range(0, self.cols):
                    m[row].append(0)
            for row in range(0, self.rows):
                for col in range(0, self.cols):
                    m[row][col] = self.mat[row][col] * other
            return Matrix(self.rows, self.cols, m)

        elif type(other) == Matrix:
            if self.cols != other.rows:
                error("Cannot multiply two matrices of those dimensions")
                raise DimensionError
            else:
                m = []
                for i in range(0, self.rows):
                    m.append([])
                    for j in range(0, other.cols):
                        m[i].append(0)
                for i in range(0, self.rows):
                    for j in range(0, other.cols):
                        a = Vector(self.mat[i])
                        b = []
                        for k in range(0, other.rows):
                            b.append(other.mat[k][j])
                        b = Vector(b)
                        print(a.vec)
                        print(b.vec)
                        c = a.dot_prod(b)
                        print(c)
                        m[i][j] = c
                return Matrix(self.rows, other.cols, m)

        else:
            error("""Cannot multiply by that type, 
                  try an integer, float, complex number or matrix""")
            raise TypeError

    def two_x_two_det(self):
        """
        This is a function that will find the determinant of a 2x2 matrix
        :return: Scalar
        """
        if self.rows != 2 or self.cols != 2:
            error("This function can only find the determinant of a 2x2 matrix")
            raise DimensionError
        else:
            return self.mat[0][0] * self.mat[1][1] - self.mat[0][1] * self.mat[1][0]

    def cofactor(self, row, col):
        """
        This is a function that will return the cofactor of a given spot in a matrix
        :param row: int
        :param col: int
        :return: Scalar
        """
        return (-1) ** (col + row) * self.mat[row][col]

    def minor(self, row, col):
        """
        This is a function that will return the minor of a given spot in a matrix
        :param row: int
        :param col: int
        :return: Matrix
        """
        m = []
        for i in range(0, self.rows):
            if i != row:
                m.append([])
                for j in range(0, self.cols):
                    if j != col:
                        if i < row:
                            m[i].append(self.mat[i][j])
                        else:
                            m[i - 1].append(self.mat[i][j])
        m = Matrix(self.rows - 1, self.cols - 1, m)
        return m

    def det(self, d=0):
        """
        This is a function that will find the determinant of an n x n matrix
        :param d: Scalar
        :return: Scalar
        """
        if self.is_square():
            for i in range(0, self.cols):
                m = self.minor(0, i)
                c = self.cofactor(0, i)
                print(d)
                d += c * m.det()
                print(d)
        return d

    def transpose(self):
        """
        This is a function that will return the transpose of a matrix
        :return: Matrix
        """
        m = []
        for i in range(self.cols):
            m.append([])
            for j in range(self.rows):
                m[i].append(self.mat[j][i])
        return Matrix(self.rows, self.cols, m)

    def reshape(self, row, col):
        """
        This is a function that takes a matrix and then reshapes it to the given dimensions
        :param row: int
        :param col: int
        :return: Matrix
        """
        components = self.to_list()
        if len(components) == row * col:
            m = []
            index = 0
            for i in range(row):
                m.append([])
                for j in range(col):
                    m[i].append(components[index])
                    index += 1
            return Matrix(row, col, m)
        else:
            if len(components) > row * col:
                error("There are more components than fit in a matrix of that size")
            else:
                error("There are not enough components to fill a matrix of this size")
            raise DimensionError

    def to_list(self):
        """
        This is a function that returns a list of all components of matrix
        :return: list
        """
        components = []
        for i in range(self.rows):
            for j in range(self.cols):
                components.append(self.mat[i][j])
        return components

    def elementwise(self, function, other=None):
        """
        This is a function that applies a function either to a matrix or between two matrices elementwise
        :param function: function
        :param other: Matrix
        :return: Matrix
        """
        if other is None:
            m = []
            for i in range(self.rows):
                m.append([])
                for j in range(self.cols):
                    m[i].append(function(self.mat[i][j]))
            return Matrix(self.rows, self.cols, m)
        else:
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
                error("These matrices are of different dimensions")
                raise DimensionError

    def mat_mean(self):
        total = self.mat_sum()
        return total / (self.rows * self.cols)

    def mat_sum(self):
        total = 0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.mat[i][j]
        return total

    def to_vec(self):
        """
        Takes a matrix and returns a vector
        :return: Vector
        """
        if self.rows == 1:
            v = self.mat[0]
            return Vector(v)
        elif self.cols == 1:
            v = []
            for i in range(self.rows):
                print(i)
                v.append(self.mat[i][0])
            return Vector(v)
        else:
            error("Cannot make a vector of those dimensions")
            raise DimensionError

    def mat_is_equal(self, other):
        if type(self) != type(other):
            return False
        if type(self) == Matrix:
            if (self.rows, self.cols) == (other.rows, other.cols):
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.mat[i][j] != other.mat[i][j]:
                            return False
                return True
        return self == other


def make(rows, cols, arr=[]):
    """
    This combines two functions to create a matrix and the mat field which is where the actual matrix is stored
    :param rows: int
    :param cols: int
    :param arr: list
    :return: Matrix
    """
    m = Matrix(rows, cols, arr)
    m = m.create()
    return m


def identity(dimension):
    """
    This function returns an identity matrix
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


def from_range(rows, cols, start=0):
    """
    This function returns a matrix of the given dimensions where each element is just the next in range(maximum)
    :param rows: int
    :param cols: int
    :param start: int
    :return: Matrix
    """
    maximum = rows * cols
    components = [x for x in range(start, start + maximum)]
    return from_list(components, rows, cols)


def from_list(components, rows, cols):
    """
    This is a function that creates a matrix of given dimensions from a list
    :param components: list
    :param rows: int
    :param cols: int
    :return: Matrix
    """
    if len(components) == rows * cols:
        m = []
        index = 0
        for i in range(rows):
            m.append([])
            for j in range(cols):
                m[i].append(components[index])
                index += 1
        return Matrix(rows, cols, m)
    else:
        if len(components) > rows * cols:
            error("There are more components than fit in this matrix")
        else:
            error("There aren't enough components to fill this matrix")
        raise DimensionError


def from_vec(vec):
    """
    Takes a vector and returns a matrix
    :param vec: Vector
    :return: Matrix
    """
    lst = vec.vec
    return from_list(lst, len(lst), 1)


def rotate(angle, axis, dim=3):
    """
    This is a function which returns a rotation matrix
    :param angle: real number
    :param axis: 'x', 'y' or 'z'
    :param dim: int (2 or 3)
    :return: Matrix
    """
    c = round(math.cos(angle), 3)
    s = round(math.sin(angle), 3)
    if axis == 'z' and dim == 2:
        return from_list([c, s, -s, c], 2, 2)
    elif axis == 'x' and dim == 3:
        return from_list([1, 0, 0, 0, c, s, 0, -s, c], 3, 3)
    elif axis == 'y' and dim == 3:
        return from_list([c, 0, -s, 0, 1, 0, s, 0, c])
    elif axis == 'z' and dim == 3:
        return from_list([c, s, 0, -s, c, 0, 0, 0, 1])


"""Below here is testing"""  # -----------------------------------------------------------------------------------------

m1 = Matrix(3, 3, [[3, 2, 4], [5, 6, 7], [1, 8, 9]])
m2 = Matrix(3, 3, [[5, 3, 2], [2, 8, 6], [3, 6, 7]])
m3 = from_list([1, 2, 3], 3, 1)
m4 = Matrix(2, 2, [[1, 0], [0, 1]])
m5 = Matrix(2, 2, [[1, 0], [0, 1]])
