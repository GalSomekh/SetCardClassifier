import cv2
import time
import threading
import queue
import numpy as np


class VideoCapture:
    """
    Bufferless video capture
    taken from https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    """

    def __init__(self, name, kill, cap_settings=None):
        """
        Open the video device and set the given settings

        :param name: name of the video device
        :param kill: function that returns True if the thread should be killed
        :param cap_settings: settings for the video capture, dict - setting: value
        """
        cap_settings = cap_settings or {}
        self.kill = kill
        self.cap = self._open_cap_with_retries(name)
        for setting, val in cap_settings.items():
            self.cap.set(setting, val)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.start()

    def _open_cap_with_retries(self, name, retries: int = 3):
        """
        Open the video capture with retries

        :param name: name of the video device
        :param retries: number of retries
        :return: cv2.VideoCapture object
        """
        for i in range(retries):
            cap = cv2.VideoCapture(name)
            if cap.isOpened():
                return cap
            else:
                # TODO: Logger
                # LOGGER.warning("Could not open camera by index, retrying")
                time.sleep(1)

        raise Exception("Could not open camera")

    def _reader(self):
        """
        Grab frames from capture as soon as they are available
        """
        while not self.kill():
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

        self.cap.release()

    def read(self) -> (bool, np.ndarray):
        """
        Return the most recent frame

        :return: tuple of (bool, frame) to be in cap.read() format
        """
        return True, self.q.get()
