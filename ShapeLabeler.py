import cv2
import numpy as np
import constants as const
import helpers


class ShapeLabeler:
    """
    Object for shape labeling a given contour.
    """
    def __init__(self, cnt: np.ndarray, poly_approx: np.ndarray,
                 cnt_center: np.ndarray, cnt_area: float, saved_shapes: list):
        """
        Initializes a ShapeLabeler object

        :param cnt: Contour
        :param poly_approx: Polygon approximation of contour
        :param cnt_center: Center of contour
        :param cnt_area: Area of contour
        :param saved_shapes: List of saved shapes
        """
        self.cnt = cnt
        self.poly_approx = poly_approx
        self.cnt_center = cnt_center
        self.cnt_area = cnt_area
        self.saved_shapes = saved_shapes

    def is_diamond(self) -> bool:
        """
        Checks if a contour is a diamond

        :return: True if contour is a diamond, False otherwise
        """
        extreme_points = helpers.get_contour_extreme_points(self.cnt)
        horz_diag = [extreme_points[0], extreme_points[2]]
        vert_diag = [extreme_points[1], extreme_points[3]]
        angle = helpers.get_lines_angle(vert_diag, horz_diag)
        # Checks angle between the two diagonals
        if abs(90 - angle) > const.DIAMOND_ANGLE_THRESHOLD:
            return False
        area = helpers.point_distance(*horz_diag) * helpers.point_distance(*vert_diag) * 0.5
        # Checks area difference between geometric area and contour area
        if abs(1 - (area / self.cnt_area)) > const.DIAMOND_AREA_DIFF_THRESHOLD:
            return False
        int_point = helpers.get_line_intersection(horz_diag, vert_diag)
        # Checks if the intersection point is close to the center of the contour
        if helpers.point_distance(int_point, self.cnt_center) > const.DIAMOND_CENTER_DIFF_THRESHOLD:
            return False
        _, _, w, h = cv2.boundingRect(self.cnt)
        w_h_ratio = w / h
        # Checks w / h ratio to filter angled shapes
        if w_h_ratio > const.DIAMOND_WH_RATIO_TOP_THRESHOLD or w_h_ratio < const.DIAMOND_WH_RATIO_BOTTOM_THRESHOLD:
            return False
        return True


    def is_wave(self) -> bool:
        """
        Checks if a contour is a wave

        :return: True if contour is a wave, False otherwise
        """
        l = len(self.poly_approx)
        in_out = helpers.get_in_out_stats(self.cnt, self.poly_approx, 2)
        if in_out.get(-1, l) / l > const.WAVE_MID_OUT_PCT_THRESHOLD:
            return False
        _, _, w, h = cv2.boundingRect(self.cnt)
        w_h_ratio = w / h
        # Checks w / h ratio to filter angled shapes
        if w_h_ratio > const.WAVE_WH_RATIO_TOP_THRESHOLD or w_h_ratio < const.WAVE_WH_RATIO_BOTTOM_THRESHOLD:
            return False
        return helpers.match_shape_with_dest(self.cnt, 'wave', self.saved_shapes)

    def is_oval(self) -> bool:
        """
        Checks if a contour is an oval

        :return: True if contour is an oval, False otherwise
        """
        l = len(self.poly_approx)
        in_out = helpers.get_in_out_stats(self.cnt, self.poly_approx, l // 2)
        if in_out.get(1, l) / l < const.OVAL_MID_OUT_PCT_THRESHOLD:
            return False
        _, _, w, h = cv2.boundingRect(self.cnt)
        w_h_ratio = w / h
        # Checks w / h ratio to filter angled shapes
        if w_h_ratio > const.OVAL_WH_RATIO_TOP_THRESHOLD or w_h_ratio < const.OVAL_WH_RATIO_BOTTOM_THRESHOLD:
            return False
        return helpers.match_shape_with_dest(self.cnt, 'oval', self.saved_shapes)

    def detect_shape(self) -> str:
        """
        Detects the shape of a contour

        :return: Shape of contour
        """
        approx_len = len(self.poly_approx)
        shape = None
        if approx_len == 4 and self.is_diamond():
            shape = 'D'
        elif approx_len > 7 and self.is_wave():
            shape = 'W'
        elif approx_len > 7 and self.is_oval():
            shape = 'O'

        return shape
