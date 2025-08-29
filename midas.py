import torch
import cv2
import time
import numpy as np
from midas.model_loader import load_model, default_models

first_execution = True

def process(device, model, model_type, image, input_size, target_size, optimize):
    global first_execution
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("Optimization to half-floats activated.")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution:
        height, width = sample.shape[2:]
        print(f"Input resized to {width}x{height} before encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    return prediction

def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

def crop_center_9_16(frame):
    h, w = frame.shape[:2]
    # Desired output aspect ratio = 9:16 (w:h)
    desired_w = int(h * 9 / 16)
    start_x = (w - desired_w) // 2
    return frame[:, start_x:start_x+desired_w]

def main():
    # Configuration
    model_type = "dpt_next_vit_large_384"
    video_path = r"C:\Users\Pongpiched Rakshit\Documents\Adobe\Premiere Pro\25.0\videoplayback_1_1.mp4"
    optimize = False
    resize_inference = 384  # Reduce resolution for speed

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_weights = default_models[model_type]
    model, transform, net_w, net_h = load_model(
        device, model_weights, model_type, optimize,
        height=resize_inference, square=True
    )

    # Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = 1
    time_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to center 9:16
        cropped = crop_center_9_16(frame)

        # Show cropped original
        cv2.imshow("Original Cropped (9:16)", cropped)

        # Depth estimation
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        image = transform({"image": rgb / 255.0})["image"]

        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), rgb.shape[1::-1], optimize)

        depth_vis = normalize_depth(prediction)
        cv2.imshow("Depth Map", depth_vis)

        # FPS display
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - time_start))
        time_start = now
        print(f"\rFPS: {fps:.2f}", end="")

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    main()
