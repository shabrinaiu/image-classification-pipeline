import cv2
import numpy as np

FRAME_SKIP = 30  # Skip every n frames (set to 1 to not skip)


def batch_generator(video_path, batch_size):
    cap = cv2.VideoCapture(video_path)
    batch = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        batch.append(frame)

        if len(batch) == batch_size:
            # At this point, batch is ready for processing (e.g., send to model)
            # batch_np = np.stack(batch)  # Shape: (BATCH_SIZE, H, W, 3)
            print(f"Processed batch of {batch_size} frames")
            break
            # batch = []  # Reset batch

        frame_idx += 1

    if batch:
        # batch_np = np.stack(batch)
        print(f"Processed final batch of {len(batch)} frames")

    return batch
