import os
from datetime import datetime
import cv2
import numpy as np
import pickle
import constants as const
import math
from collections import defaultdict
from shapely.geometry import LineString


def pickle_saved_shapes(shapes: list):
    """
    Dump list of tuples [(shape_name, contour),...] to saved shapes pickled file

    :param shapes: List of tuples [(shape_name, contour),...]
    """
    with open(const.SAVED_SHAPES_PICKLED_FILE, 'wb') as handle:
        pickle.dump(shapes, handle)


def load_saved_shapes() -> list:
    """
    Load saved shapes from saved shapes pickled file

    :return: List of tuples [(shape_name, contour),...]
    """
    with open(const.SAVED_SHAPES_PICKLED_FILE, 'rb') as handle:
        return pickle.load(handle)


def point_distance(p1: np.array, p2: np.array) -> float:
    """
    Calculate the distance between two points

    :param p1: First point
    :param p2: Second point
    :return: Distance between the two points
    """
    return np.linalg.norm(p1 - p2)


def get_contour_center(contour: np.array) -> np.array:
    """
    Calculate the center of a contour

    :param contour: Contour to calculate center of
    :return: Center of the contour
    """
    M = cv2.moments(contour)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    return np.array((cX, cY))


def get_midpoint(p1: np.array, p2: np.array) -> np.array:
    """
    Get the middle point between two points

    :param p1: First point
    :param p2: Second point
    :return: Middle point between the two points
    """
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def get_line_slope(line: np.array) -> float:
    """
    Calculates line slope

    :param line: Line to calculate slope of
    :return: float of slope
    """
    if (line[1][0] - line[0][0]) == 0:
        return 1000
    return (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])


def get_lines_angle(line1: np.array, line2: np.array) -> float:
    """
    Calculates the angle between two lines

    :param line1: First line
    :param line2: Second line
    :return: Angle between the two lines
    """
    m1 = get_line_slope(line1)
    m2 = get_line_slope(line2)
    angle = abs((m2 - m1) / (1 + m1 * m2))
    ret = math.atan(angle)
    return (ret * 180) / math.pi


def get_flat_angle_pct(poly_approx: np.array) -> float:
    """
    Runs over poly_approx and returns pct of pairs that had a flat angle between them

    :param poly_approx: Polygon approximation of contour
    :return: float of pct of pairs that had a flat angle between them
    """
    l = len(poly_approx)
    flats = 0
    for idx, p1 in enumerate(poly_approx):
        p1 = p1[0]
        p2 = poly_approx[(idx + 1) % l][0]
        slope = get_line_slope([p1, p2])
        if abs(slope) < const.FLAT_ANGLE_THRESHOLD:
            flats += 1
    return flats / l


def get_line_intersection(line1: np.array, line2: np.array) -> np.array:
    """
    Get intersection point of two lines
    https://stackoverflow.com/a/56791549

    :param line1: First line
    :param line2: Second line
    :return: Intersection point of the two lines
    """
    l1 = LineString(line1)
    l2 = LineString(line2)
    int_point = l1.intersection(l2)
    return np.array([int_point.x, int_point.y], np.int32)


def get_contour_extreme_points(cnt: np.array) -> list:
    """
    Returns the left, right, top and bottom most points of a contour
    https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html

    :param cnt: A given contour
    :return: list of extreme points [left, top, right, bottom]
    """
    left = cnt[cnt[:, :, 0].argmin()][0]
    right = cnt[cnt[:, :, 0].argmax()][0]
    top = cnt[cnt[:, :, 1].argmin()][0]
    bottom = cnt[cnt[:, :, 1].argmax()][0]
    return [left, top, right, bottom]


def get_in_out_stats(cnt: np.array, poly_approx: np.array, jump: int) -> defaultdict[int]:
    """
    Gets stats of contour points inside, outside and on the polygon.
    Connects the poly approx points with a line and checks where the middle point of the line falls.
    The jump parameter is used for line stretching between point i and i + jump

    pointPolygonTest returns +1, -1, or 0 to indicate if a point is inside, outside, or on the contour, respectively.
    :param cnt: Contour to get stats of
    :param poly_approx: Polygon approximation of contour
    :param jump: number of points to jump in order to connect
    """
    l = len(poly_approx)
    in_out = defaultdict(int)
    for idx, p1 in enumerate(poly_approx):
        p1 = p1[0]
        p2 = poly_approx[(idx + jump) % l][0]
        mid = get_midpoint(p1, p2)
        res = cv2.pointPolygonTest(cnt, mid, False)
        in_out[res] += 1

    return in_out


def match_shape_with_dest(cnt: np.array, dest_shape: str, saved_shapes: list) -> bool:
    """
    Checks if a contour is similar to a given shape

    :param cnt: contour to match the shape to
    :param dest_shape: destination shape that is the current estimation for this contour
    :param saved_shapes: List of tuples [(shape_name, contour),...]
    :return: True if the contour is similar to the destination shape
    """
    saved_to_check = [cnt for s, cnt in saved_shapes if s == dest_shape]
    for s_cnt in saved_to_check:
        match = cv2.matchShapes(cnt, s_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if match < const.SHAPE_MATCH_DEST_MAX_THRESHOLD:
            return True

    return False


def match_card(cnt_center: np.array, cards: dict, image_width: int) -> tuple:
    """
    Matches a contour to a card based on X axis distance

    :param cnt_center: Center of contour
    :param cards: dict of cards. key: card center, value: list of shapes
    :param image_width: Width of image
    :return: tuple of card center
    """
    for card_center in cards.keys():
        dist_pct = abs(card_center[0] - cnt_center[0]) / image_width
        if dist_pct < const.CARD_GROUPING_MAX_DIST_AS_IMAGE_WIDTH_PCT:
            return card_center
    return None


def save_image(image, path=None):
    """
    Saves an image to a file

    :param image: Image to save
    :param path: Path to save image to
    """
    if path is None:
        file_name = datetime.now().strftime("%Y-%m-%d %H_%M_%S.%fff") + '.png'
        path = os.path.join(const.IMAGE_LOGS_FOLDER_PATH, file_name)
    cv2.imwrite(path, image)

