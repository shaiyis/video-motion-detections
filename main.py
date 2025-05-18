import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, shared_memory
from datetime import datetime

def stream(video_path, stream_queue):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fallback to 25 if unknown

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        streamer_detector_shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
        shm_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=streamer_detector_shm.buf)
        np.copyto(shm_frame, frame)

        stream_queue.put((streamer_detector_shm.name, frame.shape))

        time.sleep(1 / fps)
    
    # Signal end of stream
    stream_queue.put(None)
    cap.release()


def detect(stream_queue, detect_queue):
    first_frame = None

    while True:
        item = stream_queue.get()
        
        if item is None:
            detect_queue.put(None)
            break

        shm_name, shape = item
        streamer_detector_shm = shared_memory.SharedMemory(name=shm_name)
        frame = np.ndarray(shape, dtype=np.uint8, buffer=streamer_detector_shm.buf)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray.copy()
            continue

        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            detections.append((x, y, w, h))

        # Copy frame to new shared memory
        detector_displayer_shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
        frame_out = np.ndarray(frame.shape, dtype=frame.dtype, buffer=detector_displayer_shm.buf)
        np.copyto(frame_out, frame)

        detect_queue.put((detector_displayer_shm.name, detections, frame.shape))
        time.sleep(0.02)

        streamer_detector_shm.close()
        streamer_detector_shm.unlink()


def display(detect_queue):
    while True:
        item = detect_queue.get()
        if item is None:
            break

        shm_name, detections, shape = item

        detector_displayer_shm = shared_memory.SharedMemory(name=shm_name)
        frame = np.ndarray(shape, dtype=np.uint8, buffer=detector_displayer_shm.buf)
        
        # Blur each detected region
        for (x, y, w, h) in detections:
            roi = frame[y:y+h, x:x+w]
            # Apply Gaussian blur to the region of interest
            blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)
            frame[y:y+h, x:x+w] = blurred_roi

        # Draw detections
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put timestamp
        now = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        detector_displayer_shm.close()
        detector_displayer_shm.unlink()

    cv2.destroyAllWindows()


def main():
    video_path = 'video.mp4'  # Put your video file in the same directory

    # Queues
    stream_queue = Queue()
    detect_queue = Queue()

    # Start processes
    streamer = Process(target=stream, args=(video_path, stream_queue))
    detector = Process(target=detect, args=(stream_queue, detect_queue))
    displayer = Process(target=display, args=(detect_queue,))
    
    try:
        streamer.start()
        detector.start()
        displayer.start()

        streamer.join()
        detector.join()
        displayer.join()

    finally:
        for p in [streamer, detector, displayer]:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()