import cv2
import numpy as np
import argparse
import math
import time
from collections import deque
from tqdm import tqdm  # pip install tqdm

# ---------- Params (tweak these) ----------
feature_params = dict(maxCorners=300,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

patch_size = 28
max_tracks = 250
redetect_interval = 20
min_track_len = 2
subtitle_block_height = 160
line_thickness = 1
fade_old_lines = True
trail_len = 8

# ---------- Helpers ----------
def make_subtitle_mask(h, w, block_height):
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if block_height > 0:
        mask[h - block_height : h, :] = 0
    return mask

# ---------- Main ----------
def run(input_path, output_path, attach_audio=False, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    num_frames_to_process = end_frame - start_frame

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mask_template = make_subtitle_mask(h, w, subtitle_block_height)

    # Jump to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Empty video or can't read frames.")
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray_prev, mask=mask_template, **feature_params)
    if p0 is None:
        p0 = np.empty((0,1,2), dtype=np.float32)

    tracks = []
    for pt in p0:
        dq = deque(maxlen=trail_len)
        dq.append(tuple(pt.ravel()))
        tracks.append(dq)

    frame_idx = start_frame
    rng = np.random.default_rng(12345)
    colors = [tuple(int(c) for c in rng.integers(50, 255, size=3)) for _ in range(max_tracks)]

    pbar = tqdm(total=num_frames_to_process, unit="frame", desc="Processing", ncols=80)
    start_time = time.time()

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)
            if p1 is None:
                p1 = np.empty_like(p0)
                st = np.zeros((p0.shape[0],1), dtype=np.uint8)

            good_new = p1[st==1].reshape(-1,2)
            new_tracks = []
            gi = 0
            for i, s in enumerate(st):
                if s == 1:
                    x,y = good_new[gi]
                    new_tracks.append(tracks[i])
                    new_tracks[-1].append((float(x), float(y)))
                    gi += 1
            tracks = new_tracks
            p0 = good_new.reshape(-1,1,2).astype(np.float32)
        else:
            p0 = np.empty((0,1,2), dtype=np.float32)

        if frame_idx % redetect_interval == 0 and len(p0) < max_tracks:
            mask = mask_template.copy()
            for t in tracks:
                if len(t) > 0:
                    x,y = map(int, t[-1])
                    cv2.circle(mask, (x,y), 12, 0, -1)
            new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
            if new_pts is not None:
                for pt in new_pts.reshape(-1,2):
                    if len(tracks) >= max_tracks:
                        break
                    dq = deque(maxlen=trail_len)
                    dq.append((float(pt[0]), float(pt[1])))
                    tracks.append(dq)
                all_pts = np.array([list(t[-1]) for t in tracks], dtype=np.float32).reshape(-1,1,2)
                p0 = all_pts

        canvas = frame.copy()

        for idx, t in enumerate(tracks):
            if len(t) >= 2:
                color = colors[idx % len(colors)]
                pts_int = np.array(list(t), dtype=np.int32)
                if len(pts_int) >= 2:
                    cv2.polylines(canvas, [pts_int], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
                if fade_old_lines:
                    alpha_step = 1.0 / max(1, len(t))
                    for j, (x,y) in enumerate(t):
                        alpha = (j+1) * alpha_step
                        r = int(max(1, (1 - alpha) * 6))
                        overlay = canvas.copy()
                        cv2.circle(overlay, (int(x), int(y)), r+2, color, -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.25 * alpha, canvas, 1 - 0.25 * alpha, 0, canvas)

        hs = patch_size // 2
        for idx, t in enumerate(tracks):
            x, y = t[-1]
            xi, yi = int(round(x)), int(round(y))
            x0, y0 = max(0, xi - hs), max(0, yi - hs)
            x1, y1 = min(w, xi + hs), min(h, yi + hs)
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            try:
                patch_small = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            except Exception:
                continue
            cx0, cy0 = max(0, xi - hs), max(0, yi - hs)
            cx1, cy1 = min(w, cx0 + patch_size), min(h, cy0 + patch_size)
            if cx1 - cx0 != patch_size or cy1 - cy0 != patch_size:
                patch_small = patch_small[0:(cy1-cy0), 0:(cx1-cx0)]
            canvas[cy0:cy1, cx0:cx1] = patch_small
            color = colors[idx % len(colors)]
            cv2.rectangle(canvas, (cx0, cy0), (cx1-1, cy1-1), color, 1, cv2.LINE_AA)

        out.write(canvas)
        gray_prev = gray.copy()
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Processing finished in {time.time() - start_time:.1f}s. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output_noaudio.mp4', help='Output video path')
    parser.add_argument('--audio', action='store_true', help='Attach original audio (requires moviepy)')
    parser.add_argument('--start-frame', type=int, default=0, help='First frame to process')
    parser.add_argument('--end-frame', type=int, default=None, help='Last frame to process (exclusive)')
    args = parser.parse_args()
    run(args.input, args.output, attach_audio=args.audio,
        start_frame=args.start_frame, end_frame=args.end_frame)

