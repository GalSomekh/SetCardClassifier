import cv2
import time
import logging
import helpers
from VideoProcessor import VideoProcessor

logging.basicConfig(level=logging.INFO)
helpers.clear_image_logs()

# Process Video file
vid_path = r"example_videos\old_cards.mp4"
# Process Webcam stream
# vid_path = 0
vp = VideoProcessor(vid_path)
display_processed_image = True
log_results = True
get_next_frame = True

total_processing_time = 0
frame_idx = 0
while get_next_frame:
    start = time.perf_counter()
    processed_image = vp.get_cards_from_camera(return_image=display_processed_image, log=log_results)
    if processed_image is not None:
        frame_processing_time = time.perf_counter() - start
        total_processing_time += frame_processing_time
        frame_idx += 1
        log_message = f"Frame {frame_idx} processed in {frame_processing_time:.4f}," \
                      f"avg processing time: {total_processing_time/frame_idx:.4f}"
        logging.info(log_message)
        if display_processed_image:
            cv2.imshow("Processed Video", processed_image)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("q pressed, quiting")
                exit(0)
    get_next_frame = processed_image is not None

vp.close()

logging.info(f"Done processing {vid_path}")
if log_results:
    logging.info("All processed frames are available in the image_logs folder")
