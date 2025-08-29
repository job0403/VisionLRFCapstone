from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load models
detect_model = YOLO('yolo11n.pt')
segment_model = YOLO(r"C:\Users\Pongpiched Rakshit\Downloads\best (4).pt").to('cuda')

# Video input
#video_path = r"C:\Users\Pongpiched Rakshit\Downloads\Drivers view Thailand, Wongwian Yai to Maha Chai, Feb 2025 - YouTube - Google Chrome 2025-06-21 14-12-05.mp4"
video_path = r"C:\Users\Pongpiched Rakshit\Documents\Adobe\Premiere Pro\25.0\videoplayback_1_1.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
center_x1 = int(frame_width * 0.4)
center_x2 = int(frame_width * 0.6)
prev_time = time.time()

# Output video writer
output_path = r"C:\Users\Pongpiched Rakshit\Downloads\output_video2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

window_name = "YOLOv8 Combined Danger Detection"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # Segmentation
    seg_results = segment_model.predict(source=frame, device=0, conf=0.5)
    danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    if seg_results[0].masks is not None:
        best_overlap = 0
        best_polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        for polygon in seg_results[0].masks.xy:
            polygon = polygon.astype(int)
            poly_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [polygon], 1)

            overlap_ratio = np.mean(poly_mask[:, center_x1:center_x2])
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_polygon_mask = poly_mask

        if best_overlap > 0.05:
            rows = frame_height
            for y in range(rows - 1, -1, -5):
                max_width = 500
                min_width = -500
                dilation_width = int(min_width + (max_width - min_width) * (y / rows))
                kernel_size = dilation_width if dilation_width % 2 == 1 else dilation_width + 1

                if kernel_size > 1:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                    slice_mask = best_polygon_mask[y:y+5, :]
                    dilated_slice = cv2.dilate(slice_mask, kernel, iterations=1)
                    danger_mask[y:y+5, :] = dilated_slice

            mask_3c = np.repeat(danger_mask[:, :, np.newaxis], 3, axis=2)
            blue_overlay = np.zeros_like(annotated_frame)
            blue_overlay[:, :, 0] = 255
            blended = cv2.addWeighted(blue_overlay, 0.5, annotated_frame, 0.5, 0)
            annotated_frame = np.where(mask_3c == 1, blended, annotated_frame)

    # Object detection
    det_results = detect_model.predict(source=frame, device=0, conf=0.2)
    boxes = det_results[0].boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)

        box_area = (x2 - x1) * (y2 - y1)
        intersect_area = 0

        if box_area > 0:
            obj_mask = danger_mask[y1:y2, x1:x2]
            intersect_area = np.sum(obj_mask > 0)

        overlap_ratio = intersect_area / box_area if box_area > 0 else 0
        color = (0, 0, 255) if overlap_ratio > 0.2 else (0, 255, 0)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show and save
    cv2.imshow(window_name, annotated_frame)
    out_writer.write(annotated_frame)  # Save frame to output video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()
