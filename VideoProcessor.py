from typing import Union, \
                   ValuesView

import cv2
import numpy as np
import threading
from collections import defaultdict
import multiprocessing as mp
from VideoCapture import VideoCapture
from ShapeLabeler import ShapeLabeler
from ColorLabeler import ColorLabeler
import helpers
import constants as const


def get_card_value_worker(args_tuple):
    color_labeler, cnt, shape, shape_details, card_center = args_tuple
    color, fill = color_labeler.get_cnt_color_and_fill(cnt, shape)
    count = len(shape_details)
    return card_center, f"{shape}{color}{count}{fill}"


class VideoProcessor:
    """
    Process video from camera
    """

    def __init__(self, name: Union[str, int] = None):
        """
        Initialize video processor

        :param name: name of the video device. str for video file path, int for webcam device id
        """
        self.saved_shapes = helpers.load_saved_shapes()
        self.kill = False
        if name is not None:
            self.cap = cv2.VideoCapture(name)
        else:
            cap_settings = {
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720
            }
            # kill passed as lambda in order to kill with threading
            self.cap = VideoCapture(0, lambda: self.kill, cap_settings)
        self.mp_pool = mp.pool.ThreadPool(3)  # ThreadPool was faster than Pool on raspberry pi

        self.color_labeler = None
        self.contours = None
        self.bottom_thresh = None
        self.top_thresh = None
        self.contour_areas = None
        self.potential_shapes = None
        self.marked_shapes = None
        self.poly_approxes = None
        self.contour_centers = None

    def is_potential_shape(self, cnt_idx: int) -> bool:
        """
        Check if a contour is a potential shape based on area and thresholds

        :param cnt_idx: Idx of contour to check
        :return: True if contour is a potential shape, False otherwise
        """
        return self.bottom_thresh <= self.contour_areas[cnt_idx] <= self.top_thresh

    def group_to_cards(self, image_width: int) -> dict:
        """
        Groups shapes to cards according to X axis

        :param image_width: width of image
        :return: dict of cards. key: card center, value: list of shapes
        """
        cards = defaultdict(list)
        for idx, cnt_details in self.marked_shapes.items():
            card_center = helpers.match_card(self.contour_centers[idx], cards, image_width)
            card_center = self.contour_centers[idx] if card_center is None else card_center
            cards[tuple(card_center)].append((idx, cnt_details))

        return cards

    def get_card_values(self, cards: dict) -> dict:
        """
        Gets the values of cards in format of DR3F, full docs in game_modules.modes.set

        :param cards: dict of cards. key: card center, value: list of shapes
        :return: dict of cards. key: card center, value: card value
        """
        args_lst = []
        for card_center, shape_details in cards.items():
            cnt_idx, shape = shape_details[0]
            args_lst.append((self.color_labeler,
                             self.potential_shapes[cnt_idx],
                             shape, shape_details, card_center))

        pool_res = self.mp_pool.map(get_card_value_worker, args_lst)

        return {card_center: value for card_center, value in pool_res}

    def check_approx_contour_intersect(self, cnt_idx: int) -> bool:
        """
        Check if the given contour center is within the threshold distance of any other contour center

        :param cnt_idx: Index of contour center to check
        :return: True if an intersecting contour center is found, False otherwise
        """
        for idx2 in self.marked_shapes.keys():
            center1 = self.contour_centers[cnt_idx]
            center2 = self.contour_centers[idx2]
            dist = helpers.point_distance(center1, center2)
            if dist <= const.CONTOUR_CENTER_DISTANCE_INTERSECT_THRESHOLD:
                return True
        return False

    def initial_image_processing(self, image: np.ndarray):
        """
        Initial image processing

        Calculates and saves fields that will be used multiple times
        in order to avoid recalculating on each use

        :param image: image to process
        """
        self.color_labeler = ColorLabeler(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params = const.ADAPTIVE_THRESHOLD_PARAMS
        binary = cv2.adaptiveThreshold(gray, params[0], cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, params[1], params[2])
        self.contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contour_areas = [cv2.contourArea(cnt) for cnt in self.contours]

        self.bottom_thresh = image.shape[0] * image.shape[1] * const.POTENTIAL_SHAPE_SIZE_BOTTOM_THRESHOLD
        self.top_thresh = image.shape[0] * image.shape[1] * const.POTENTIAL_SHAPE_SIZE_TOP_THRESHOLD
        # Filter shapes based on size \ area
        potential_shapes = [(cnt, self.contour_areas[idx]) for idx, cnt in enumerate(self.contours)
                            if self.is_potential_shape(idx)]
        potential_shapes.sort(key=lambda c: c[1], reverse=True)
        self.contour_areas = [c[1] for c in potential_shapes]
        self.potential_shapes = [c[0] for c in potential_shapes]

        self.marked_shapes = {}
        self.poly_approxes = [cv2.approxPolyDP(c, const.POLY_APPROX_FACTOR * cv2.arcLength(c, True), True) for c in
                              self.potential_shapes]
        self.contour_centers = [helpers.get_contour_center(c) for c in self.potential_shapes]

    def draw_cards(self, image: np.array, cards: dict, card_values: dict) -> np.ndarray:
        """
        Draw cards on image

        :param image: image to draw on
        :param cards: dict of cards. key: card center, value: list of shapes
        :param card_values: dict of cards. key: card center, value: card value
        :return: image with cards drawn
        """
        for card, shapes in cards.items():
            card_cnt = np.concatenate([self.potential_shapes[s[0]] for s in shapes])
            x, y, w, h = cv2.boundingRect(card_cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, card_values[card], (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return image

    def get_cards_from_camera(self, return_image: bool = False, log: bool = False) -> Union[None, ValuesView, np.ndarray]:
        """
        Reads the current frame

        :param return_image: True if image should be returned, False otherwise
        :param log: True if image with drawn cards should be logged, False otherwise
        :return: list of cards or image with drawn cards
        """
        _, image = self.cap.read()
        if image is None:
            return

        self.initial_image_processing(image)

        for idx, cnt in enumerate(self.potential_shapes):
            if self.check_approx_contour_intersect(idx) or \
                    helpers.get_flat_angle_pct(self.poly_approxes[idx]) > const.MAX_FLAT_ANGLE_PCT_THRESHOLD:
                continue
            shape_labeler = ShapeLabeler(cnt, self.poly_approxes[idx], self.contour_centers[idx],
                                         self.contour_areas[idx], self.saved_shapes)
            shape = shape_labeler.detect_shape()
            if shape is None:
                continue
            self.marked_shapes[idx] = shape

        cards = self.group_to_cards(image.shape[1])
        card_values = self.get_card_values(cards)
        if return_image:
            image = self.draw_cards(image, cards, card_values)
            if log:
                threading.Thread(target=helpers.save_image, args=(image,), daemon=True).start()
            return self.draw_cards(image, cards, card_values)
        else:
            if log:
                threading.Thread(target=self.draw_and_log_image, args=(image, cards, card_values), daemon=True).start()
            return card_values.values()

    def draw_and_log_image(self, image: np.array, cards: dict, card_values: dict):
        """
        Draw cards on image and log it

        :param image: image to draw on
        :param cards: dict of cards. key: card center, value: list of shapes
        :param card_values: dict of cards. key: card center, value: card value
        """
        image = self.draw_cards(image, cards, card_values)
        helpers.save_image(image)

    def close(self):
        """
        Closes the VideoCapture object thread that is running in the background
        """
        self.kill = True
