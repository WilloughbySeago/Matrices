from Code.errors import *
import math
from typing import Union, List, Optional, Any, TypeVar

TNum = TypeVar('TNum', int, float, )


class Vector:
    """This is a class of vectors"""

    def __init__(self, vec: List[TNum]) -> None:
        """
        Initialise the vector
        :param vec: list
        """
        self.vec = vec
        self.length = len(self.vec)

    def dot_prod(self, other) -> Union[int, float, complex]:
        """
        This is a function that returns the dot product of two vectors
        :param other: Vector
        :return: Scalar
        """
        a = self.vec
        b = other.vec
        if len(a) == len(b):
            a_dot_b = sum([a_i * b_i for a_i, b_i in zip(a, b)])
            return a_dot_b
        else:
            error("Cannot do a dot product with two vectors of these dimensions")
            raise DimensionError

    def cross_prod(self, other):
        """
        This is a function that returns the cross product of two vectors
        :param other: Vector
        :return: Vector
        """
        a = self.vec
        b = other.vec
        if len(a) == 3 and len(b) == 3:
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return Vector([x, y, z])
        else:
            error("Cannot do a cross product with non-3D vectors")
            raise DimensionError

    def vec_add(self, other):
        """
        This is a function to add two vectors
        :param other: Vector
        :return: Vector
        """

        a = self.vec
        b = other.vec
        if len(a) != len(b):
            error("Cannot add vectors of different dimensions")
            raise DimensionError
        elif type(b) != type(a):
            error("Can only add a vector to a vector")
            raise TypeError
        else:
            a_plus_b = [x + y for x, y in zip(a, b)]
            return Vector(a_plus_b)

    def vec_mult(self, scalar: Union[int, float, complex]):
        """
        This is a function that multiplies a vector by a scalar
        :param scalar: Scalar
        :return: Vector
        """
        if type(scalar) in [int, float, complex]:
            scaled = [a * scalar for a in self.vec]
            return Vector(scaled)
        else:
            error("Cannot multiply by a non-scalar")
            raise TypeError

    def mag(self) -> Union[int, float]:
        """
        This is a function that returns the magnitude of a vector
        :return: Scalar
        """
        return math.sqrt(sum([x ** 2 for x in self.vec]))

    def unit(self):
        """
        This is a function that returns a unit vector in the direction of the given vector
        :return: Vector
        """
        mag = self.mag()
        unit = self.vec_mult(1 / mag)
        return unit

    def angle(self, other) -> Union[int, float]:
        """
        This is a function that calculates the angle between two vectors
        :param other: Vector
        :return: Scalar
        """
        if type(self) == type(other):
            adotb = self.dot_prod(other)
            a = self.mag()
            b = other.mag()
            angle = math.acos(adotb / (a * b))
            return angle
        else:
            error("Cannot work out angle between non-vectors")
            raise TypeError

    def elementwise(self, function):
        """
        This is a function that applies a given function to every element of a vector
        :param function: function
        :return: Vector
        """
        v = []
        for component in self.vec:
            v.append(function(component))
        return Vector(v)

    def vec_mean(self) -> Union[int, float, complex]:
        """
        This is a function that returns the mean of all values in a vector
        :return: Scalar
        """
        return sum(self.vec) / len(self.vec)

    def vec_sum(self) -> Union[int, float, complex]:
        """
        This is a function that returns the sum of all elemts of a vector
        :return: Scalar
        """
        return sum(self.vec)

    def vec_is_equal(self, other) -> bool:
        if type(self) != type(other):
            return False
        if type(self) == Vector:
            if len(self.vec) != len(other.vec):
                return False
            for i in range(len(self.vec)):
                if self.vec[i] != other.vec[i]:
                    return False
        return True


"""Below here is testing"""  # -----------------------------------------------------------------------------------------

v1 = Vector([1, 0, 0])
v2 = Vector([0, 1, 0])
print(v2.angle(v1))
