import cv2
import av          # <-- Added PyAV
import time
import numpy as np
import torch
import serial
import threading
import winsound
import wave        # <-- Added for generating instant memory audio
import io          # <-- Added for memory streams
from ultralytics import YOLO
from midas.model_loader import load_model, default_models
import pygame      # <-- Replaces winsound
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
VIDEO_PATH = "rtsp://admin:Otmadmin1234@10.10.9.120:554/cam/realmonitor?channel=1&subtype=0"
OUTPUT_PATH = r"C:\Users\Pongpiched Rakshit\Downloads\output_with_depth.mp4"
SEGMENT_MODEL = r"C:\Users\Pongpiched Rakshit\Downloads\best (4).pt"

SAVE_OUTPUT = False                 # <- toggle saving
OUTPUT_CODEC = 'mp4v'

DETECT_MODEL = "yolo11n.pt"

MIDAS_MODEL_TYPE = "dpt_next_vit_large_384"
MIDAS_OPTIMIZE = False
MIDAS_HEIGHT = 384
ASSUME_BOTTOM_CENTER_M = 2.0        # bottom-center pixel equals this many meters

DANGER_OVERLAP_THRESH = 0.20        # rails overlap → red bbox

# ----------------------------
# Shared Variables / Kill Switch
# ----------------------------
keep_running = True        
big_lrf_value = "Error"
small_lrf_value = "Error"

# --- Variables for debugging the audio alarm ---
alarm_status_d = float('inf')
alarm_status_dur = 0
alarm_status_cd = 0.0

# LRF serial ports
BIG_LRF_PORT = "COM4"
SMALL_LRF_PORT = "COM3"
LRF_BAUDRATE = 115200

# ----------------------------
# Pygame Zero-Latency Audio Setup
# ----------------------------
# Initialize Pygame mixer with a tiny 512-sample buffer for instant response
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

def create_and_load_beep(filename, freq, duration_ms):
    """Generates a physical wav file and loads it into Pygame's instant mixer."""
    sample_rate = 44100
    samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, samples, False)
    audio_data = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
        
    sound = pygame.mixer.Sound(filename)
    # Optional: clean up the file after loading it into RAM
    try: os.remove(filename) 
    except: pass 
    return sound

print("Initializing Zero-Latency Pygame Audio...")
SND_LVL_1 = create_and_load_beep("temp_beep_1.wav", 800, 200)
SND_LVL_2 = create_and_load_beep("temp_beep_2.wav", 1000, 150)
SND_LVL_3 = create_and_load_beep("temp_beep_3.wav", 1500, 100)
SND_LVL_4 = create_and_load_beep("temp_beep_4.wav", 2000, 50)

# ----------------------------
# MiDaS helper
# ----------------------------
_first_execution = True
def process_midas(device, model, model_type, image_np, target_size, optimize):
    """Forward MiDaS and resize prediction to target_size (w,h)."""
    global _first_execution
    sample = torch.from_numpy(image_np).to(device).unsqueeze(0)

    if optimize and device.type == "cuda":
        if _first_execution:
            print("MiDaS: half precision optimization ON")
        sample = sample.to(memory_format=torch.channels_last).half()

    if _first_execution:
        h, w = sample.shape[2:]
        print(f"MiDaS encoder input: {w}x{h}")
        _first_execution = False

    pred = model.forward(sample)
    pred = (
        torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return pred  

def crop_center_9_16(frame_bgr):
    """Crop to center 9:16 strip using full frame height."""
    h, w = frame_bgr.shape[:2]
    crop_w = int(h * 9 / 16)
    start_x = max((w - crop_w) // 2, 0)
    end_x = min(start_x + crop_w, w)
    return frame_bgr[:, start_x:end_x], start_x, end_x

# ----------------------------
# LRF Readers
# ----------------------------
def lrf_reader(port, which):
    global big_lrf_value, small_lrf_value
    start_cmd = bytes([0xFA,0x01,0xFF,0x04,0x01,0x00,0x00,0x00,0x00,0xFF])
    stop_cmd  = bytes([0xFA,0x01,0xFF,0x04,0x00,0x00,0x00,0x00,0x00,0xFE])
    try:
        ser = serial.Serial(port, LRF_BAUDRATE, timeout=1)
        time.sleep(1)
        ser.write(start_cmd)
        buffer = bytearray()
        while keep_running:
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting)
                while len(buffer) >= 9:
                    if buffer[0] == 0xFB and buffer[1] == 0x03:
                        frame = buffer[:9]
                        buffer = buffer[9:]
                        valid_flag = int.from_bytes(frame[4:6], 'little')
                        distance_dm = int.from_bytes(frame[6:8], 'little')
                        txt = f"{distance_dm/10:.1f} m" if valid_flag == 1 else "Invalid"
                        if which == "BIG":
                            big_lrf_value = txt
                        else:
                            small_lrf_value = txt
                    else:
                        buffer = buffer[1:]
            time.sleep(0.01)
    except Exception:
        pass
    finally:
        try:
            ser.write(stop_cmd)
            ser.close()
        except Exception:
            pass

def get_current_min_distance():
    """Helper function to parse the LRF strings and find the closest distance."""
    distances = []
    for val in [big_lrf_value, small_lrf_value]:
        if isinstance(val, str) and val.endswith(" m"):
            try:
                distances.append(float(val.split()[0]))
            except ValueError:
                pass
    if not distances:
        return float('inf') 
    return min(distances)

# ----------------------------
# Ultra-Fast Async Alarm Thread
# ----------------------------
def alarm_manager():
    """Background thread utilizing Pygame for instant stops."""
    global alarm_status_d, alarm_status_dur, alarm_status_cd
    next_beep_time = 0.0
    
    while keep_running:
        try:
            d = get_current_min_distance()
            alarm_status_d = d
            now = time.time()
            
            if d > 1000:
                alarm_status_dur = 0
                alarm_status_cd = 0.0
                
                # THE MAGIC FIX: Instantly and brutally kills all audio output 
                pygame.mixer.stop() 
                time.sleep(0.05) 
                
            elif 500 < d <= 1000:
                alarm_status_dur = 200
                alarm_status_cd = 1.0
                if now >= next_beep_time:
                    SND_LVL_1.play() # Non-blocking instant play
                    next_beep_time = time.time() + 1.0 
                else:
                    time.sleep(0.02)
                    
            elif 200 < d <= 500:
                alarm_status_dur = 150
                alarm_status_cd = 0.5
                if now >= next_beep_time:
                    SND_LVL_2.play()
                    next_beep_time = time.time() + 0.5
                else:
                    time.sleep(0.02)
                    
            elif 100 <= d <= 200:
                alarm_status_dur = 100
                alarm_status_cd = 0.2
                if now >= next_beep_time:
                    SND_LVL_3.play()
                    next_beep_time = time.time() + 0.2
                else:
                    time.sleep(0.02)
                    
            elif d < 100:
                alarm_status_dur = 50
                alarm_status_cd = 0.1
                if now >= next_beep_time:
                    SND_LVL_4.play()
                    next_beep_time = time.time() + 0.1
                else:
                    time.sleep(0.02)
                    
        except Exception as e:
            print(f"ALARM THREAD ERROR: {e}")
            time.sleep(0.1)
            
    # Cleanup on exit
    pygame.mixer.quit()

# ----------------------------
# Init models / devices
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# MiDaS
midas_weights = default_models[MIDAS_MODEL_TYPE]
midas_model, midas_transform, net_w, net_h = load_model(
    device,
    midas_weights,
    MIDAS_MODEL_TYPE,
    MIDAS_OPTIMIZE,
    height=MIDAS_HEIGHT,
    square=True,
)
if MIDAS_OPTIMIZE and device.type == "cuda":
    midas_model = midas_model.half()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# YOLOs
detect_model = YOLO(DETECT_MODEL).to(0 if device.type == "cuda" else "cpu")
segment_model = YOLO(SEGMENT_MODEL).to(0 if device.type == "cuda" else "cpu")

# Start LRF threads & Alarm Thread
threading.Thread(target=lrf_reader, args=(BIG_LRF_PORT,"BIG"), daemon=True).start()
threading.Thread(target=lrf_reader, args=(SMALL_LRF_PORT,"SMALL"), daemon=True).start()
threading.Thread(target=alarm_manager, daemon=True).start() 

# ----------------------------
# PyAV Video Initialization
# ----------------------------
print("Connecting directly via PyAV for zero-latency stream...")
options = {
    'rtsp_transport': 'udp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'strict': 'experimental'
}

try:
    container = av.open(VIDEO_PATH, options=options)
except Exception as e:
    raise RuntimeError(f"Could not open video stream: {e}")

stream = container.streams.video[0]
stream.thread_type = "NONE" 

frame_w = stream.codec_context.width
frame_h = stream.codec_context.height
fps_in = float(stream.average_rate) if stream.average_rate else 30.0

out_writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, int(fps_in), (frame_w, frame_h))

win = "YOLO + Rails + Depth + LRF (PyAV Zero-Delay)"
cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

# ----------------------------
# Video loop
# ----------------------------
for av_frame in container.decode(stream):
    if not keep_running:
        break
        
    frame = av_frame.to_ndarray(format='bgr24')
    annotated = frame.copy()

    # -------- Segmentation (rails ROI) --------
    rails_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    seg_results = segment_model.predict(source=frame, device=0, conf=0.5, verbose=False)
    if seg_results and seg_results[0].masks is not None:
        center_x1 = int(frame_w * 0.4)
        center_x2 = int(frame_w * 0.6)
        best_overlap = 0.0
        best_polygon_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        for polygon in seg_results[0].masks.xy:
            polygon = polygon.astype(int)
            poly_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [polygon], 1)
            overlap_ratio = np.mean(poly_mask[:, center_x1:center_x2])
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_polygon_mask = poly_mask

        if best_overlap > 0.05:
            rows = frame_h
            for y in range(rows - 1, -1, -5):
                max_width = 500
                min_width = -500
                dilation_width = int(min_width + (max_width - min_width) * (y / rows))
                ksize = dilation_width if dilation_width % 2 == 1 else dilation_width + 1
                if ksize > 1:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
                    slice_mask = best_polygon_mask[y:y+5, :]
                    dilated_slice = cv2.dilate(slice_mask, kernel, iterations=1)
                    rails_mask[y:y+5, :] = dilated_slice

    if rails_mask.any():
        mask_3c = rails_mask[..., None]
        blue = np.zeros_like(annotated); blue[..., 0] = 255
        blended = cv2.addWeighted(blue, 0.5, annotated, 0.5, 0)
        annotated = np.where(mask_3c == 1, blended, annotated)

    # -------- Detection --------
    det_results = detect_model.predict(source=frame, device=0, conf=0.2, verbose=False)
    boxes = det_results[0].boxes.xyxy.cpu().numpy().astype(int) if det_results else []

    # -------- Depth (MiDaS on center 9:16 crop) --------
    cropped, xL, xR = crop_center_9_16(frame) 
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    image_in = midas_transform({"image": rgb / 255.0})["image"]
    with torch.no_grad():
        pred_raw = process_midas(
            device, midas_model, MIDAS_MODEL_TYPE,
            image_in, target_size=rgb.shape[1::-1], optimize=MIDAS_OPTIMIZE
        )

    Hc, Wc = pred_raw.shape
    ref_val = pred_raw[Hc - 1, Wc // 2]
    eps = 1e-8
    scale_k = ASSUME_BOTTOM_CENTER_M * max(ref_val, eps) 
    depth_meters = scale_k / (pred_raw + eps)

    # -------- Annotate detections with danger + depth --------
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)

        box_area = max(1, (x2 - x1) * (y2 - y1))
        overlap = np.sum(rails_mask[y1:y2, x1:x2] > 0) / box_area
        danger = overlap > DANGER_OVERLAP_THRESH
        color = (0, 0, 255) if danger else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if danger:
            xc1 = max(x1, xL); xc2 = min(x2, xR)
            if xc2 > xc1:
                lx1 = max(0, min(Wc - 1, xc1 - xL))
                lx2 = max(0, min(Wc,     xc2 - xL))
                ly1 = max(0, min(Hc - 1, y1))
                ly2 = max(0, min(Hc,     y2))
                if lx2 > lx1 and ly2 > ly1:
                    patch = depth_meters[ly1:ly2, lx1:lx2]
                    if patch.size:
                        closest = float(np.min(patch))
                        cv2.putText(
                            annotated, f"{closest:.2f} m",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                        )

    # -------- Overlay LRF values + FPS --------
    cv2.putText(annotated, f"BIG LRF: {big_lrf_value}", (10, frame_h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(annotated, f"SMALL LRF: {small_lrf_value}", (10, frame_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # show / save
    cv2.imshow(win, annotated)
    if SAVE_OUTPUT and out_writer is not None:
        out_writer.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        keep_running = False
        break

container.close()
if SAVE_OUTPUT and out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()
