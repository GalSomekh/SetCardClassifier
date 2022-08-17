import os
import pathlib

ADAPTIVE_THRESHOLD_PARAMS = (255, 21, 8)
CONTOUR_CENTER_DISTANCE_INTERSECT_THRESHOLD = 10  # %CHECK% Maybe should be pct of image dimensions
DIAMOND_ANGLE_THRESHOLD = 15  # Max angle between diamond diagonals
DIAMOND_AREA_DIFF_THRESHOLD = 0.2  # Max difference in area between contour area and diamond geometric formula area
DIAMOND_CENTER_DIFF_THRESHOLD = 20  # # %CHECK% Maybe should be pct of image dimensions
# Max distance between contour center and diagonal meeting point
DIAMOND_WH_RATIO_BOTTOM_THRESHOLD = 0.575  # Bottom threshold for the ratio of the width to height of a diamond
DIAMOND_WH_RATIO_TOP_THRESHOLD = 0.85  # Top threshold for the ratio of the width to height of a diamond
POTENTIAL_SHAPE_SIZE_BOTTOM_THRESHOLD = 0.001  # Bottom threshold for potential shape size as pct of image dimensions
POTENTIAL_SHAPE_SIZE_TOP_THRESHOLD = 0.02  # Top threshold for potential shape size as pct of image dimensions
WAVE_WH_RATIO_BOTTOM_THRESHOLD = 0.5  # Bottom threshold for the ratio of the width to height of a wave
WAVE_WH_RATIO_TOP_THRESHOLD = 0.8  # Top threshold for the ratio of the width to height of a wave
WAVE_MID_OUT_PCT_THRESHOLD = 0.3  # Pct of the points that are outside the wave contour
OVAL_WH_RATIO_BOTTOM_THRESHOLD = 0.54  # Bottom threshold for the ratio of the width to height of an oval
OVAL_WH_RATIO_TOP_THRESHOLD = 0.8  # Top threshold for the ratio of the width to height of an oval
OVAL_MID_OUT_PCT_THRESHOLD = 0.99  # Pct of the points that are outside the oval contour
POLY_APPROX_FACTOR = 0.015  # Factor for the approximation of the polygonal contour
SHAPE_MATCH_MAX_THRESHOLD = 1.2  # Max distance between contours in general shape matching
SHAPE_MATCH_DEST_MAX_THRESHOLD = 0.5  # Max distance between contours in destination shape matching
CARD_GROUPING_MAX_DIST_AS_IMAGE_WIDTH_PCT = 0.05  # Max distance between cards to be grouped as pct of image dimensions
FLAT_ANGLE_THRESHOLD = 0.5  # Top value to be considered flat
MAX_FLAT_ANGLE_PCT_THRESHOLD = 0.45  # Max pct of the flat angles to be considered flat
CV_FOLDER_PATH = pathlib.Path(__file__).parent.resolve()
SAVED_SHAPES_PICKLED_FILE = os.path.join(CV_FOLDER_PATH, "saved_shapes.pickle")
IMAGE_LOGS_FOLDER_PATH = os.path.join(CV_FOLDER_PATH, "image_logs")

PCT_THRESHOLDS = {
    'O': [(22, 'E'), (62, 'S'), (100, 'F')],
    'W': [(25, 'E'), (45, 'S'), (100, 'F')],
    'D': [(20, 'E'), (45, 'S'), (100, 'F')]
}
MATCHING_HUES = {
    'R': 0,
    'G': 60,
    'B': 115
}