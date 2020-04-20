import math
import numpy as np

# def rotatex(xxx_todo_changeme, xxx_todo_changeme1, deg):
#     """
#     Returns the new x value of a point (x, y)
#     rotated about the point (refx, refy) by
#     deg degrees clockwise.
#     """
#     (x, y) = xxx_todo_changeme
#     (refx, refy) = xxx_todo_changeme1
#     return (math.cos(math.radians(deg))*(x - refx)) + (math.sin(math.radians(deg))*(y - refy)) + refx


# def rotatey(xxx_todo_changeme2, xxx_todo_changeme3, deg):
#     """
#     Returns the new y value of a point (x, y)
#     rotated about the point (refx, refy) by
#     deg degrees clockwise.
#     """
#     (x, y) = xxx_todo_changeme2
#     (refx, refy) = xxx_todo_changeme3
#     return (- math.sin(math.radians(deg))*(x - refx)) + (math.cos(math.radians(deg))*(y - refy)) + refy


# def rotatep(p, ref, d):
#     """
#     Returns the new point as an integer tuple
#     of a point p (tuple) rotated about the point
#     ref (tuple) by d degrees clockwise.
#     """
#     return (int(round(rotatex(p, ref, d))), int(round(rotatey(p, ref, d))))


class Shape:
    """
    A class that defines the functions associated
    with a shape.
    """
    ID = ""
    points = []  # TODO optimize
    corners = []  # TODO optimize
    idx = -1
    rotation_matrix = np.array([(0, -1), (1, 0)])

    @property
    def size(self):
        return len(self.points)
    # def create(self, num, pt):
    #     self.set_points(0, 0)
    #     pm = self.points
    #     self.points_map = pm
    #     self.refpt = pt
    #     x = pt[0] - self.points_map[num][0]
    #     y = pt[1] - self.points_map[num][1]
    #     self.set_points(x, y)

    def set_points(self, x, y):
        raise Exception()

    def rotate(self):
        # """
        # Returns the points that would be covered by a
        # shape that is rotated 0, 90, 180, of 270 degrees
        # in a clockwise direction.
        # """
        # assert(self.points is not None)
        # assert(degrees in [0, 90, 180, 270])
        # self.degrees = degrees

        # def rotate_this(p):
        #     return(rotatep(p, self.refpt, degrees))
        # self.points = list(map(rotate_this, self.points))
        # self.corners = list(map(rotate_this, self.corners))
        np_ref = np.array([self.ref_point])
        np_points = np.array(self.points)
        np_corners = np.array(self.corners)

        np_points = (np_points - self.ref_point) @ self.rotation_matrix + self.ref_point
        self.points = list(map(tuple, np_points))

        np_corners = (np_corners - self.ref_point) @ self.rotation_matrix + self.ref_point
        self.corners = list(map(tuple, np_corners))

    def flip(self):
        # """
        # Returns the points that would be covered if the shape
        # was flipped horizontally or vertically.
        # """
        # assert orientation in ["h", "v"]
        # assert self.points is not None
        # self.orientation = orientation

        # def flip_h(p):
        #     x1 = self.refpt[0]
        #     x2 = p[0]
        #     x1 = (x1 - (x2 - x1))
        #     return (x1, p[1])

        # def no_flip(p):
        #     return p
        # # flip the piece horizontally
        # if orientation == "h":
        #     self.points = list(map(flip_h, self.points))
        #     self.corners = list(map(flip_h, self.corners))
        # # flip the piece vertically
        # elif orientation == "v":
        #     self.points = list(map(no_flip, self.points))
        #     self.corners = list(map(no_flip, self.corners))
        # else:
        #     raise Exception("Invalid orientation.")
        np_ref = np.array([self.ref_point])
        np_points = np.array(self.points)
        np_corners = np.array(self.corners)

        np_points = np_points - np_ref
        np_points[:, 1] = -np_points[:, 1]
        np_points = np_points + np_ref
        self.points = list(map(tuple, np_points))

        np_corners = np_corners - np_ref
        np_corners[:, 1] = -np_corners[:, 1]
        np_corners = np_corners + np_ref
        self.corners = list(map(tuple, np_corners))

    @staticmethod
    def from_json(obj):
        shape = Shape()
        shape.ID = obj["ID"]
        shape.points = list(map(tuple, obj["points"]))  # TODO optimize
        shape.corners = list(map(tuple, obj["corners"]))  # TODO optimize
        shape.idx = obj["idx"]
        # for key, value in obj.items():
        #     if key != "refpt" and isinstance(value, list):
        #         setattr(shape, key, [tuple(l) for l in value])
        #     else:
        #         setattr(shape, key, value)

        return shape

    def to_json(self, idx):
        self.idx = idx
        return {
            'ID': self.ID,
            'points': [(int(x), int(y)) for x, y in self.points],
            'corners': [(int(x), int(y)) for x, y in self.corners],
            'idx': self.idx
        }

    def __eq__(self, value):
        # return self.idx == value.idx  # TODO optimize
        return sorted(self.points) == sorted(value.points)

    def __hash__(self):
        # return self.idx  # TODO optimize
        return hash(str(sorted(self.points)))

    def __str__(self):
        return "\n".join([f"Id: {self.ID}", f"Points: {sorted(self.points)}"])
        # return "\n".join([f"Id: {self.ID}", f"Size: {self.size}",
        #                   f"Orientation: {self.orientation}", f"Degrees: {self.degrees}",
        #                   f"Points: {sorted(self.points)}"])
