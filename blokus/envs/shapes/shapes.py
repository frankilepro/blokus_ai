from blokus.envs.shapes.shape import Shape


def get_all_shapes():
    return [I1(), I2(), I3(), I4(), I5(), V3(), L4(), Z4(), O4(), L5(),
            T5(), V5(), N(), Z5(), T4(), P(), W(), U(), F(), X(), Y()]


class I1(Shape):
    def __init__(self):
        self.ID = "I1"
        self.size = 1

    def set_points(self, x, y):
        self.points = [(x, y)]
        self.corners = [(x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)]


class I2(Shape):
    def __init__(self):
        self.ID = "I2"
        self.size = 2

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 2), (x - 1, y + 2)]


class I3(Shape):
    def __init__(self):
        self.ID = "I3"
        self.size = 3

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 3), (x - 1, y + 3)]


class I4(Shape):
    def __init__(self):
        self.ID = "I4"
        self.size = 4

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 4), (x - 1, y + 4)]


class I5(Shape):
    def __init__(self):
        self.ID = "I5"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3), (x, y + 4)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 5), (x - 1, y + 5)]


class V3(Shape):
    def __init__(self):
        self.ID = "V3"
        self.size = 3

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class L4(Shape):
    def __init__(self):
        self.ID = "L4"
        self.size = 4

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class Z4(Shape):
    def __init__(self):
        self.ID = "Z4"
        self.size = 4

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y)]
        self.corners = [(x - 2, y - 1), (x + 1, y - 1), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2), (x - 2, y + 1)]


class O4(Shape):
    def __init__(self):
        self.ID = "O4"
        self.size = 4

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 2), (x - 1, y + 2)]


class L5(Shape):
    def __init__(self):
        self.ID = "L5"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x + 3, y)]
        self.corners = [(x - 1, y - 1), (x + 4, y - 1), (x + 4, y + 1), (x + 1, y + 2), (x - 1, y + 2)]


class T5(Shape):
    def __init__(self):
        self.ID = "T5"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x - 1, y), (x + 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 3), (x - 1, y + 3), (x - 2, y + 1), (x - 2, y - 1)]


class V5(Shape):
    def __init__(self):
        self.ID = "V5"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y), (x + 2, y)]
        self.corners = [(x - 1, y - 1), (x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 3), (x - 1, y + 3)]


class N(Shape):
    def __init__(self):
        self.ID = "N"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x + 1, y), (x + 2, y), (x, y - 1), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 2), (x + 3, y - 1), (x + 3, y + 1), (x - 1, y + 1), (x - 2, y), (x - 2, y - 2)]


class Z5(Shape):
    def __init__(self):
        self.ID = "Z5"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x + 1, y), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 2), (x, y + 2), (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class T4(Shape):
    def __init__(self):
        self.ID = "T4"
        self.size = 4

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x - 1, y)]
        self.corners = [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2), (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]


class P(Shape):
    def __init__(self):
        self.ID = "P"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x, y - 2)]
        self.corners = [(x + 1, y - 3), (x + 2, y - 2), (x + 2, y + 1), (x - 1, y + 1), (x - 1, y - 3)]


class W(Shape):
    def __init__(self):
        self.ID = "W"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [(x + 1, y - 1), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 2), (x, y - 2)]


class U(Shape):
    def __init__(self):
        self.ID = "U"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x + 1, y - 1)]
        self.corners = [(x + 2, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2), (x - 1, y - 2)]


class F(Shape):
    def __init__(self):
        self.ID = "F"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y), (x + 2, y + 2), (x - 1, y + 2),
                        (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class X(Shape):
    def __init__(self):
        self.ID = "X"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        self.corners = [(x + 1, y - 2), (x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 2),
                        (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1), (x - 1, y - 2)]


class Y(Shape):
    def __init__(self):
        self.ID = "Y"
        self.size = 5

    def set_points(self, x, y):
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x - 1, y)]
        self.corners = [(x + 3, y - 1), (x + 3, y + 1), (x + 1, y + 2), (x - 1, y + 2), (x - 2, y + 1), (x - 2, y - 1)]
