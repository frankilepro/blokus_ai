from blokus_gym.envs.shapes.shape import Shape


def get_all_shapes():
    return [I1(), I2(), I3(), I4(), I5(), V3(), L4(), Z4(), O4(), L5(),
            T5(), V5(), N(), Z5(), T4(), P(), W(), U(), F(), X(), Y()]


class I1(Shape):
    label = "I1"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y)]
        self.corners = [(x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)]


class I2(Shape):
    label = "I2"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 2), (x - 1, y + 2)]


class I3(Shape):
    label = "I3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 3), (x - 1, y + 3)]


class I4(Shape):
    label = "I4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 4), (x - 1, y + 4)]


class I5(Shape):
    label = "I5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3), (x, y + 4)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 5), (x - 1, y + 5)]


class V3(Shape):
    label = "V3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class L4(Shape):
    label = "L4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class Z4(Shape):
    label = "Z4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y)]
        self.corners = [(x - 2, y - 1), (x + 1, y - 1), (x + 2, y),
                        (x + 2, y + 2), (x - 1, y + 2), (x - 2, y + 1)]


class O4(Shape):
    label = "O4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 2), (x - 1, y + 2)]


class L5(Shape):
    label = "L5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x + 3, y)]
        self.corners = [(x - 1, y - 1), (x + 4, y - 1), (x + 4, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class T5(Shape):
    label = "T5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x - 1, y), (x + 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3),
                        (x - 1, y + 3), (x - 2, y + 1), (x - 2, y - 1)]


class V5(Shape):
    label = "V5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y), (x + 2, y)]
        self.corners = [(x - 1, y - 1), (x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class N(Shape):
    label = "N"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 2, y), (x, y - 1), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 2), (x + 3, y - 1), (x + 3, y + 1),
                        (x - 1, y + 1), (x - 2, y), (x - 2, y - 2)]


class Z5(Shape):
    label = "Z5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 2), (x, y + 2),
                        (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class T4(Shape):
    label = "T4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x - 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]


class P(Shape):
    label = "P"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x, y - 2)]
        self.corners = [(x + 1, y - 3), (x + 2, y - 2), (x + 2, y + 1), (x - 1, y + 1), (x - 1, y - 3)]


class W(Shape):
    label = "W"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 1), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class U(Shape):
    label = "U"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x + 1, y - 1)]
        self.corners = [(x + 2, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2), (x - 1, y - 2)]


class F(Shape):
    label = "F"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class X(Shape):
    label = "X"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class Y(Shape):
    label = "Y"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x - 1, y)]
        self.corners = [(x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]
