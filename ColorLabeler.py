from collections import Counter
import multiprocessing as mp
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
import constants as const


class ColorLabeler:
    """
    Object for color labeling shapes detected in image

    init color LAB values once for all instances

    code taken from:
    https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
    https://pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/
    """
    white = cv2.cvtColor(np.array([[[255, 255, 255]]], dtype="uint8"), cv2.COLOR_RGB2LAB)[0][0]

    def __init__(self, image: np.ndarray):
        """
        Initialize color labeler

        Saves color classes as LAB, this representation is more accurate than RGB
        for distinguishing colors by euclidean distance

        :param image: image to label, must be in BGR color space, will be saved as LAB on self
        """
        self.lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def label_color(self, lab_trio: np.ndarray) -> str:
        """
        Label color value with closest color class

        :param pixel: pixel to label as np array (in LAB color space)
        :return: name of color class
        """
        rgb_trio = list(cv2.cvtColor(np.array([[lab_trio.astype("uint8")]]), cv2.COLOR_LAB2RGB)[0][0])
        h, _, _ = list(cv2.cvtColor(np.array([[rgb_trio]]), cv2.COLOR_RGB2HSV)[0][0])
        min_dist = (np.inf, None)
        for color, hue in const.MATCHING_HUES.items():
            dist = min(abs(h - hue), 179 - abs(h - hue))
            if dist < min_dist[0]:
                min_dist = (dist, color)

        return min_dist[1]

    def color_label_cluster_duo(self, clusters: list) -> list:
        """
        Label color values of cluster duo with closest color class

        The whiter of the two will be labeled as white, and the other will be labeled with self.label_color

        :param clusters: cluster centers from Kmeans (clt.cluster_centers_)
        :return: list of labels, idxes correspond to clusters
        """
        labels = [None, None]
        white_idx = 1
        c0_white_diff = np.linalg.norm(clusters[0] - self.white)
        c1_white_diff = np.linalg.norm(clusters[1] - self.white)

        if c0_white_diff < c1_white_diff:
            white_idx = 0

        labels[white_idx] = "white"
        other_idx = (white_idx + 1) % 2
        lab_trio = clusters[other_idx]
        labels[other_idx] = self.label_color(lab_trio)

        return labels

    def kmeans_color_percentages(self, cnt: np.ndarray, num_clusters: int = 2) -> list:
        """
        Label colors of contour with Kmeans clustering

        :param cnt: contour to label
        :param num_clusters: number of clusters to use in Kmeans
        :return: list of tuples, each tuple is (label, percentage)
        """
        x, y, w, h = cv2.boundingRect(cnt)
        rect_pixels = self.lab_image[y:y + h, x:x + w]

        # 64 was fastest on raspberry pi
        clt = MiniBatchKMeans(n_clusters=num_clusters, batch_size=64*mp.cpu_count(), tol=1e-3)
        clt.fit(rect_pixels.reshape(-1, 3))

        n_pixels = len(clt.labels_)
        counter = Counter(clt.labels_)
        pcts = [np.round(counter[i] / n_pixels * 100, 2) for i in range(num_clusters)]

        labels = self.color_label_cluster_duo(clt.cluster_centers_)
        return list(zip(labels, pcts))

    def get_cnt_color_and_fill(self, cnt: np.ndarray, shape: str, num_clusters: int = 2) -> (str, str):
        """
        Return color and filling inside of a contour with Kmeans clustering

        :param cnt: contour to check
        :param shape: shape of contour
        :param num_clusters: number of clusters to use in Kmeans
        :return: tuple of (color, fill)
        """
        labels_pcts = self.kmeans_color_percentages(cnt, num_clusters)
        color, pct = [p for p in labels_pcts if p[0] != "white"][0]

        for max_thresh, fill in const.PCT_THRESHOLDS.get(shape, []):
            if pct <= max_thresh:
                return color, fill

        return color, 'U'
