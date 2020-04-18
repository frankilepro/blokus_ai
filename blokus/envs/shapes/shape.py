import math


def rotatex(xxx_todo_changeme, xxx_todo_changeme1, deg):
    """
    Returns the new x value of a point (x, y)
    rotated about the point (refx, refy) by
    deg degrees clockwise.
    """
    (x, y) = xxx_todo_changeme
    (refx, refy) = xxx_todo_changeme1
    return (math.cos(math.radians(deg))*(x - refx)) + (math.sin(math.radians(deg))*(y - refy)) + refx


def rotatey(xxx_todo_changeme2, xxx_todo_changeme3, deg):
    """
    Returns the new y value of a point (x, y)
    rotated about the point (refx, refy) by
    deg degrees clockwise.
    """
    (x, y) = xxx_todo_changeme2
    (refx, refy) = xxx_todo_changeme3
    return (- math.sin(math.radians(deg))*(x - refx)) + (math.cos(math.radians(deg))*(y - refy)) + refy


def rotatep(p, ref, d):
    """
    Returns the new point as an integer tuple
    of a point p (tuple) rotated about the point
    ref (tuple) by d degrees clockwise.
    """
    return (int(round(rotatex(p, ref, d))), int(round(rotatey(p, ref, d))))


class Shape:
    """
    A class that defines the functions associated
    with a shape.
    """

    def __init__(self):
        self.ID = ""
        self.size = 1
        self.points = []
        self.corners = []
        self.idx = -1

    def create(self, num, pt):
        self.set_points(0, 0)
        pm = self.points
        self.points_map = pm
        self.refpt = pt
        x = pt[0] - self.points_map[num][0]
        y = pt[1] - self.points_map[num][1]
        self.set_points(x, y)

    def set_points(self, x, y):
        self.points = []
        self.corners = []

    def rotate(self, degrees):
        """
        Returns the points that would be covered by a
        shape that is rotated 0, 90, 180, of 270 degrees
        in a clockwise direction.
        """
        assert(self.points is not None)
        assert(degrees in [0, 90, 180, 270])
        self.degrees = degrees

        def rotate_this(p):
            return(rotatep(p, self.refpt, degrees))
        self.points = list(map(rotate_this, self.points))
        self.corners = list(map(rotate_this, self.corners))

    def flip(self, orientation):
        """
        Returns the points that would be covered if the shape
        was flipped horizontally or vertically.
        """
        assert(orientation == "h" or orientation is None)
        assert(self.points is not None)
        self.orientation = orientation

        def flip_h(p):
            x1 = self.refpt[0]
            x2 = p[0]
            x1 = (x1 - (x2 - x1))
            return (x1, p[1])

        def no_flip(p):
            return p
        # flip the piece horizontally
        if orientation == "h":
            self.points = list(map(flip_h, self.points))
            self.corners = list(map(flip_h, self.corners))
        # flip the piece vertically
        elif orientation is None:
            self.points = list(map(no_flip, self.points))
            self.corners = list(map(no_flip, self.corners))
        else:
            raise Exception("Invalid orientation.")

    @staticmethod
    def from_json(obj):
        shape = Shape()
        for key, value in obj.items():
            if key != "refpt" and isinstance(value, list):
                setattr(shape, key, [tuple(l) for l in value])
            else:
                setattr(shape, key, value)

        return shape

    def to_json(self, idx):
        self.idx = idx
        return vars(self)

    def __eq__(self, value):
        # return self.idx == value.idx  # TODO optimize
        return sorted(self.points) == sorted(value.points)

    def __hash__(self):
        # return self.idx  # TODO optimize
        return hash(str(sorted(self.points)))

    def __str__(self):
        return "\n".join([f"Id: {self.ID}", f"Size: {self.size}",
                          f"Orientation: {self.orientation}", f"Degrees: {self.degrees}",
                          f"Points: {sorted(self.points)}"])
