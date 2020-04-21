from blokus.envs.shapes.shape import Shape


def get_all_shapes():
    return [I1(), I2(), I3(), I4(), I5(), V3(), L4(), Z4(), O4(), L5(),
            T5(), V5(), N(), Z5(), T4(), P(), W(), U(), F(), X(), Y()]


class I1(Shape):
    ID = "I1"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y)]
        self.corners = [(x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)]


class I2(Shape):
    ID = "I2"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 2), (x - 1, y + 2)]


class I3(Shape):
    ID = "I3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 3), (x - 1, y + 3)]


class I4(Shape):
    ID = "I4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 4), (x - 1, y + 4)]


class I5(Shape):
    ID = "I5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3), (x, y + 4)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 5), (x - 1, y + 5)]


class V3(Shape):
    ID = "V3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class L4(Shape):
    ID = "L4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class Z4(Shape):
    ID = "Z4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y)]
        self.corners = [(x - 2, y - 1), (x + 1, y - 1), (x + 2, y),
                        (x + 2, y + 2), (x - 1, y + 2), (x - 2, y + 1)]


class O4(Shape):
    ID = "O4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 2), (x - 1, y + 2)]


class L5(Shape):
    ID = "L5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x + 3, y)]
        self.corners = [(x - 1, y - 1), (x + 4, y - 1), (x + 4, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class T5(Shape):
    ID = "T5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x - 1, y), (x + 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3),
                        (x - 1, y + 3), (x - 2, y + 1), (x - 2, y - 1)]


class V5(Shape):
    ID = "V5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y), (x + 2, y)]
        self.corners = [(x - 1, y - 1), (x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class N(Shape):
    ID = "N"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 2, y), (x, y - 1), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 2), (x + 3, y - 1), (x + 3, y + 1),
                        (x - 1, y + 1), (x - 2, y), (x - 2, y - 2)]


class Z5(Shape):
    ID = "Z5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 2), (x, y + 2),
                        (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class T4(Shape):
    ID = "T4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x - 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]


class P(Shape):
    ID = "P"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x, y - 2)]
        self.corners = [(x + 1, y - 3), (x + 2, y - 2), (x + 2, y + 1), (x - 1, y + 1), (x - 1, y - 3)]


class W(Shape):
    ID = "W"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 1), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class U(Shape):
    ID = "U"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x + 1, y - 1)]
        self.corners = [(x + 2, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2), (x - 1, y - 2)]


class F(Shape):
    ID = "F"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class X(Shape):
    ID = "X"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class Y(Shape):
    ID = "Y"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x - 1, y)]
        self.corners = [(x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]
