from Code.errors import *
import math
import cmath
from typing import Union, List, TypeVar

TNum = TypeVar("TNum", int, float, complex)


class Vector:
    """This is a class of vectors"""

    def __init__(self, vec: List[TNum]) -> None:
        """Initialise the vector

        :param vec: list
        """
        self.vec = vec
        self.length = len(self.vec)

    def __repr__(self):
        return str(self.vec)

    def dot_prod(self, other) -> Union[int, float, complex]:
        """This is a function that returns the dot product of two vectors

        :param other: Vector
        :return: Scalar
        """
        if not isinstance(other, Vector):
            raise TypeError("Other must be a vector")
        a = [i.conjugate() if type(i) == complex else i for i in self.vec]  # complex conjugate of a
        b = other.vec
        if len(a) == len(b):
            a_dot_b = sum([a_i * b_i for a_i, b_i in zip(a, b)])
            return a_dot_b
        else:
            raise DimensionError("Cannot do a dot product with two vectors of these dimensions")

    def cross_prod(self, other):
        """This is a function that returns the cross product of two vectors

        :param other: Vector
        :return: Vector
        """
        if not isinstance(other, Vector):
            raise TypeError("Other must be a 3D vector")
        a = self.vec
        b = other.vec
        if len(a) == 3 and len(b) == 3:
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return Vector([x, y, z])
        else:
            raise DimensionError("Cannot do a cross product with non-3D vectors")

    def vec_add(self, other):
        """This is a function to add two vectors

        :param other: Vector
        :return: Vector
        """
        if not isinstance(other, Vector):
            raise TypeError("This function is for adding vectors, \nto add elementwise use self.elementwise()")
        a = self.vec
        b = other.vec
        if len(a) != len(b):
            raise DimensionError("Cannot add vectors of different dimensions")
        else:
            a_plus_b = [x + y for x, y in zip(a, b)]
            return Vector(a_plus_b)

    def vec_mult(self, scalar: Union[int, float, complex]):
        """This is a function that multiplies a vector by a scalar

        :param scalar: Scalar
        :return: Vector
        """
        if type(scalar) in [int, float, complex]:
            scaled = [a * scalar for a in self.vec]
            return Vector(scaled)
        else:
            raise TypeError("Cannot multiply by a non-scalar")

    def mag(self) -> Union[int, float]:
        """This is a function that returns the magnitude of a vector

        :return: Scalar
        """
        return math.sqrt(sum([x ** 2 for x in self.vec]))

    def unit(self):
        """This is a function that returns a unit vector in the direction of the given vector

        :return: Vector
        """
        mag = self.mag()
        unit = self.vec_mult(1 / mag)
        return unit

    def angle(self, other) -> Union[int, float]:
        """This is a function that calculates the angle between two vectors

        :param other: Vector
        :return: Scalar
        """
        if isinstance(other, Vector):
            a_dot_b = self.dot_prod(other)
            a = self.mag()
            b = other.mag()
            angle = math.acos(a_dot_b / (a * b))
            return angle
        else:
            raise TypeError("Cannot work out angle between non-vectors")

    def elementwise(self, function):
        """This is a function that applies a given function to every element of a vector

        :param function: function
        :return: Vector
        """
        return Vector([function(x) for x in self.vec])

    def vec_mean(self) -> Union[int, float, complex]:
        """This is a function that returns the mean of all values in a vector

        :return: Scalar
        """
        return sum(self.vec) / len(self.vec)

    def vec_sum(self) -> Union[int, float, complex]:
        """This is a function that returns the sum of all elements of a vector

        :return: Scalar
        """
        return sum(self.vec)

    def vec_is_equal(self, other) -> bool:
        """This function returns true if the two vectors are the same

        :return: bool
        """
        if not isinstance(other, Vector):
            return False
        if len(self.vec) != len(other.vec):
            return False
        for i in range(len(self.vec)):
            if self.vec[i] != other.vec[i]:
                return False
        return True

    # Overwrite operators
    def __add__(self, other):
        return self.vec_add(other)

    def __sub__(self, other):
        neg_other = other.vec_mult(-1)
        return self + neg_other

    def __mul__(self, other):
        return self.dot_prod(other)

    def __matmul__(self, other):
        return self.cross_prod(other)

    def __truediv__(self, other):
        if type(other) in [int, float, complex]:
            inverse = 1 / other
            return self.vec_mult(inverse)

    def __eq__(self, other):
        return self.vec_is_equal(other)

    def __ne__(self, other):
        return not self.vec_is_equal(other)

    def __neg__(self):
        return self.vec_mult(-1)

    def __len__(self):
        return len(self.vec)


"""Below here is testing"""  # -----------------------------------------------------------------------------------------


def main():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(-v1)
    print(len(v1))


if __name__ == "__main__":
    main()
