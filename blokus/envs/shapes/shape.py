import numpy as np


class Shape:
    """
    A class that defines the functions associated
    with a shape.
    """
    id = ""
    points = []  # TODO optimize, but not this important since max 5 points
    corners = []  # TODO optimize, same
    idx = -1
    rotation_matrix = np.array([(0, -1), (1, 0)])

    @property
    def size(self):
        return len(self.points)

    def set_points(self, x, y):
        raise Exception()

    def rotate(self):
        # """
        # Returns the points that would be covered by a
        # shape that is rotated 0, 90, 180, of 270 degrees
        # in a clockwise direction.
        # """
        np_ref = np.array([self.ref_point])
        np_points = np.array(self.points)
        np_corners = np.array(self.corners)

        np_points = (np_points - np_ref) @ self.rotation_matrix + np_ref
        self.points = list(map(tuple, np_points))

        np_corners = (np_corners - np_ref) @ self.rotation_matrix + np_ref
        self.corners = list(map(tuple, np_corners))

    def flip(self):
        # """
        # Returns the points that would be covered if the shape
        # was flipped horizontally or vertically.
        # """
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
        shape.id = obj["id"]
        shape.points = list(map(tuple, obj["points"]))  # TODO optimize
        shape.corners = list(map(tuple, obj["corners"]))  # TODO optimize
        shape.idx = obj["idx"]

        return shape

    def to_json(self, idx):
        self.idx = idx
        return {
            'id': self.id,
            'points': [(int(x), int(y)) for x, y in self.points],
            'corners': [(int(x), int(y)) for x, y in self.corners],
            'idx': self.idx
        }

    def __eq__(self, value):
        # return self.idx == value.idx  # TODO optimize
        return sorted(self.points) == sorted(value.points)

    def __lt__(self, value):
        return self.idx < value.idx  # TODO optimize
        # return sorted(self.points) == sorted(value.points)

    def __hash__(self):
        # return self.idx  # TODO optimize
        return hash(str(sorted(self.points)))

    def __str__(self):
        return "\n".join([f"Id: {self.id}", f"Points: {sorted(self.points)}"])
