import cv2
import numpy as np
from ultralytics import YOLO
from tracker.trackableobject import TrackableObject
import logging
import time


# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO("yolov8x.pt")


## Input Video
input_dir = "Input"
default_video = f"{input_dir}/input.mp4"
test_video = default_video

if not cv2.haveImageReader(default_video):
    # Fallback to the first MP4 in Input/ if input.mp4 is missing
    try:
        import os

        candidates = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
        if candidates:
            test_video = f"{input_dir}/{sorted(candidates)[0]}"
    except Exception:
        pass

logger.info("Starting the video: %s", test_video)
cap = cv2.VideoCapture(test_video)

##for camera ip
# camera_ip = "Camera Url"
# logger.info("Starting the live stream..")
# cap = cv2.VideoCapture(camera_ip)
# time.sleep(1.0)


def people_counter():
    """
    Counts the number of people entering and exiting based on object tracking.
    """
    count = 0
    processed_frames = 0

    writer = None
    trackableObjects = {}

    # Initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # Initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in = []

    # Initialize video writer on first frame to match output size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (500, 280))
        if writer is None:
            H, W = frame.shape[:2]
            writer = cv2.VideoWriter("Final_output.mp4", fourcc, output_fps, (W, H), True)
        else:
            H, W = frame.shape[:2]

        line_y = H // 2 - 10
        cv2.line(frame, (0, line_y), (W, line_y), (0, 0, 0), 2)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 255), 1)

                if track_ids is None:
                    continue

                objectID = int(track_ids[i])
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                centroid = (cX, cY)

                to = trackableObjects.get(objectID)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < 0 and centroid[1] < line_y - 10:
                            totalUp += 1
                            move_out.append(totalUp)
                            to.counted = True
                        elif direction > 0 and centroid[1] > line_y + 10:
                            totalDown += 1
                            move_in.append(totalDown)
                            to.counted = True

                            total = []
                            total.append(len(move_in) - len(move_out))

                trackableObjects[objectID] = to

                text = "ID {}".format(objectID)
                cv2.putText(
                    frame,
                    text,
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info_status = [
            ("Enter", totalUp),
            ("Exit ", totalDown),
        ]

        # info_total = [("Total people inside", ', '.join(map(str, total)))]

        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        writer.write(frame)
        cv2.imshow("People Count", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        totalFrames += 1
        processed_frames += 1

        end_time = time.time()
        num_seconds = (end_time - start_time)
        if num_seconds > 28800:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    approx_fps = processed_frames / elapsed if elapsed > 0 else 0.0
    logger.info("Elapsed time: {:.2f}".format(elapsed))
    logger.info("Approx. FPS: {:.2f}".format(approx_fps))


if __name__ == "__main__":
    people_counter()
