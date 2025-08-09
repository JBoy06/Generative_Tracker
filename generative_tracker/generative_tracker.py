import cv2
import numpy as np
import argparse
import math
import time
import logging
from collections import deque
from typing import List, Tuple, Optional, Deque
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Enable OpenCL for GPU acceleration
cv2.ocl.setUseOpenCL(True)

# Constants
DEFAULT_FOURCC = 'mp4v'
FALLBACK_CODECS = ['XVID', 'MJPG', 'X264']
MIN_CIRCLE_RADIUS = 1
MAX_CIRCLE_RADIUS = 6
OVERLAY_ALPHA = 0.25
DETECTION_CIRCLE_RADIUS = 12

class TrackerConfig:
    """Configuration class for tracker parameters."""
    
    def __init__(self):
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Tracking parameters
        self.patch_size = 28
        self.min_patch_size = 14
        self.max_tracks = 250
        self.redetect_interval = 20
        self.min_track_len = 2
        self.trail_len = 8
        
        # Visual parameters
        self.subtitle_block_height = 160
        self.line_thickness = 1
        self.fade_old_lines = True
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.patch_size <= 0 or self.min_patch_size <= 0:
            raise ValueError("Patch sizes must be positive")
        if self.patch_size < self.min_patch_size:
            raise ValueError("patch_size must be >= min_patch_size")
        if self.max_tracks <= 0:
            raise ValueError("max_tracks must be positive")
        if self.trail_len <= 0:
            raise ValueError("trail_len must be positive")

class Track:
    """Represents a single feature track."""
    
    def __init__(self, initial_point: Tuple[float, float], trail_length: int):
        self.points: Deque[Tuple[float, float]] = deque(maxlen=trail_length)
        self.points.append(initial_point)
        self.id = id(self)  # Unique identifier
    
    def add_point(self, point: Tuple[float, float]):
        """Add a new point to the track."""
        self.points.append(point)
    
    def get_latest_point(self) -> Tuple[float, float]:
        """Get the most recent point."""
        return self.points[-1]
    
    def get_length(self) -> int:
        """Get the number of points in the track."""
        return len(self.points)
    
    def is_valid(self, min_length: int) -> bool:
        """Check if track meets minimum length requirement."""
        return len(self.points) >= min_length

class VideoProcessor:
    """Main video processing class for generative tracking."""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: List[Track] = []
        self.colors: List[Tuple[int, int, int]] = []
        self._generate_colors()
        
    def _generate_colors(self):
        """Pre-generate colors for tracks."""
        rng = np.random.default_rng(12345)
        self.colors = [
            tuple(int(c) for c in rng.integers(50, 255, size=3))
            for _ in range(self.config.max_tracks)
        ]
    
    def _create_subtitle_mask(self, height: int, width: int) -> cv2.UMat:
        """Create mask to exclude subtitle area from feature detection."""
        mask_np = np.ones((height, width), dtype=np.uint8) * 255
        if self.config.subtitle_block_height > 0:
            mask_np[height - self.config.subtitle_block_height:, :] = 0
        return cv2.UMat(mask_np)
    
    def _validate_frame_range(self, total_frames: int, start_frame: int, end_frame: Optional[int]) -> Tuple[int, int]:
        """Validate and adjust frame range."""
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if start_frame >= total_frames:
            raise ValueError(f"start_frame ({start_frame}) must be less than total frames ({total_frames})")
        
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        if end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        
        return start_frame, end_frame
    
    def _initialize_tracks(self, gray_frame: cv2.UMat, mask: cv2.UMat) -> np.ndarray:
        """Initialize feature tracks from first frame."""
        points_umat = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.config.feature_params)
        
        if points_umat is not None:
            points = points_umat.get()
            self.tracks = []
            
            # Handle points array - it comes as shape (N, 1, 2)
            points_reshaped = points.reshape(-1, 2)
            for pt in points_reshaped:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    track = Track((x, y), self.config.trail_len)
                    self.tracks.append(track)
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to process point {pt}: {e}")
                    continue
                    
            return points.astype(np.float32)
        else:
            self.tracks = []
            return np.empty((0, 1, 2), dtype=np.float32)
    
    def _update_tracks(self, gray_prev: cv2.UMat, gray_curr: cv2.UMat, points: np.ndarray) -> np.ndarray:
        """Update existing tracks using optical flow."""
        if len(points) == 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        
        try:
            points_umat = cv2.UMat(points)
            new_points_umat, status_umat, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, points_umat, None, **self.config.lk_params
            )
            
            if new_points_umat is not None and status_umat is not None:
                new_points = new_points_umat.get()
                status = status_umat.get()
            else:
                return np.empty((0, 1, 2), dtype=np.float32)
            
        except cv2.error as e:
            logger.warning(f"Optical flow calculation failed: {e}")
            return np.empty((0, 1, 2), dtype=np.float32)
        
        # Filter valid tracks
        valid_tracks = []
        valid_points = []
        
        new_points_reshaped = new_points.reshape(-1, 2) if len(new_points) > 0 else np.empty((0, 2))
        
        for i, (track, is_valid) in enumerate(zip(self.tracks, status.flatten())):
            if is_valid and i < len(new_points_reshaped):
                try:
                    point = new_points_reshaped[i]
                    x, y = float(point[0]), float(point[1])
                    track.add_point((x, y))
                    if track.is_valid(self.config.min_track_len):
                        valid_tracks.append(track)
                        valid_points.append([x, y])
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to process track point: {e}")
                    continue
        
        self.tracks = valid_tracks
        if valid_points:
            return np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            return np.empty((0, 1, 2), dtype=np.float32)
    
    def _detect_new_features(self, gray_frame: cv2.UMat, mask_template: cv2.UMat) -> np.ndarray:
        """Detect new features while avoiding existing tracks."""
        if len(self.tracks) >= self.config.max_tracks:
            return np.array([track.get_latest_point() for track in self.tracks], dtype=np.float32).reshape(-1, 1, 2)
        
        # Create mask excluding existing tracks
        mask_np = mask_template.get().copy()
        for track in self.tracks:
            if track.get_length() > 0:
                x, y = map(int, track.get_latest_point())
                cv2.circle(mask_np, (x, y), DETECTION_CIRCLE_RADIUS, 0, -1)
        
        try:
            mask_umat = cv2.UMat(mask_np)
            new_points_umat = cv2.goodFeaturesToTrack(gray_frame, mask=mask_umat, **self.config.feature_params)
            
            if new_points_umat is not None:
                new_points = new_points_umat.get()
                
                # Add new tracks
                new_points_reshaped = new_points.reshape(-1, 2) if len(new_points) > 0 else np.empty((0, 2))
                for pt in new_points_reshaped:
                    if len(self.tracks) >= self.config.max_tracks:
                        break
                    try:
                        x, y = float(pt[0]), float(pt[1])
                        track = Track((x, y), self.config.trail_len)
                        self.tracks.append(track)
                    except (IndexError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to create track from point {pt}: {e}")
                        continue
        
        except cv2.error as e:
            logger.warning(f"Feature detection failed: {e}")
        
        # Return all current track points
        if self.tracks:
            points_list = []
            for track in self.tracks:
                point = track.get_latest_point()
                points_list.append([point[0], point[1]])
            return np.array(points_list, dtype=np.float32).reshape(-1, 1, 2)
        else:
            return np.empty((0, 1, 2), dtype=np.float32)
    
    def _draw_trails(self, canvas: np.ndarray):
        """Draw motion trails for tracks."""
        for track_idx, track in enumerate(self.tracks):
            if track.get_length() < 2:
                continue
            
            color = self.colors[track_idx % len(self.colors)]
            points_array = np.array(list(track.points), dtype=np.int32)
            
            # Draw polyline trail
            if len(points_array) >= 2:
                cv2.polylines(canvas, [points_array], isClosed=False, 
                            color=color, thickness=self.config.line_thickness, 
                            lineType=cv2.LINE_AA)
            
            # Draw fading circles if enabled
            if self.config.fade_old_lines:
                self._draw_fading_circles(canvas, track.points, color)
    
    def _draw_fading_circles(self, canvas: np.ndarray, points: Deque[Tuple[float, float]], color: Tuple[int, int, int]):
        """Draw fading circles along the track."""
        num_points = len(points)
        if num_points == 0:
            return
        
        alpha_step = 1.0 / max(1, num_points)
        
        for j, (x, y) in enumerate(points):
            try:
                alpha = (j + 1) * alpha_step
                radius = max(MIN_CIRCLE_RADIUS, int((1 - alpha) * MAX_CIRCLE_RADIUS))
                
                overlay = canvas.copy()
                cv2.circle(overlay, (int(float(x)), int(float(y))), radius + 2, color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, OVERLAY_ALPHA * alpha, canvas, 1 - OVERLAY_ALPHA * alpha, 0, canvas)
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(f"Failed to draw fading circle at ({x}, {y}): {e}")
                continue
    
    def _draw_patches(self, canvas: np.ndarray, original_frame: np.ndarray, frame_dims: Tuple[int, int]):
        """Draw resized patches at track locations."""
        height, width = frame_dims
        half_patch = self.config.patch_size // 2
        
        for track_idx, track in enumerate(self.tracks):
            try:
                x, y = track.get_latest_point()
                center_x, center_y = int(round(float(x))), int(round(float(y)))
                
                # Calculate patch bounds
                patch_x0 = max(0, center_x - half_patch)
                patch_y0 = max(0, center_y - half_patch)
                patch_x1 = min(width, center_x + half_patch)
                patch_y1 = min(height, center_y + half_patch)
                
                # Skip if patch would be too small
                if (patch_x1 - patch_x0 < self.config.min_patch_size or 
                    patch_y1 - patch_y0 < self.config.min_patch_size):
                    continue
                
                # Extract and resize patch
                patch = original_frame[patch_y0:patch_y1, patch_x0:patch_x1]
                if patch.size == 0:
                    continue
                
                try:
                    patch_resized = cv2.resize(patch, (self.config.patch_size, self.config.patch_size), 
                                             interpolation=cv2.INTER_AREA)
                except cv2.error:
                    continue
                
                # Calculate canvas placement
                canvas_x0 = max(0, center_x - half_patch)
                canvas_y0 = max(0, center_y - half_patch)
                canvas_x1 = min(width, canvas_x0 + self.config.patch_size)
                canvas_y1 = min(height, canvas_y0 + self.config.patch_size)
                
                # Adjust patch size if it doesn't fit entirely
                if canvas_x1 - canvas_x0 != self.config.patch_size or canvas_y1 - canvas_y0 != self.config.patch_size:
                    patch_resized = patch_resized[0:(canvas_y1 - canvas_y0), 0:(canvas_x1 - canvas_x0)]
                
                # Place patch and draw border
                canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = patch_resized
                color = self.colors[track_idx % len(self.colors)]
                cv2.rectangle(canvas, (canvas_x0, canvas_y0), (canvas_x1 - 1, canvas_y1 - 1), 
                             color, 1, cv2.LINE_AA)
            
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Failed to draw patch for track {track_idx}: {e}")
                continue
    
    def process_video(self, input_path: str, output_path: str, start_frame: int = 0, 
                     end_frame: Optional[int] = None) -> bool:
        """Process video with generative tracking effect."""
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Cannot open input video: {input_path}")
            return False
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate frame range
            start_frame, end_frame = self._validate_frame_range(total_frames, start_frame, end_frame)
            num_frames_to_process = end_frame - start_frame
            
            # Create output directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Remove existing output file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    logger.info(f"Removed existing output file: {output_path}")
                except OSError as e:
                    logger.error(f"Cannot remove existing output file {output_path}: {e}")
                    return False
            
            # Setup output video with fallback codecs
            out = None
            codecs_to_try = [DEFAULT_FOURCC] + FALLBACK_CODECS
            
            for codec in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    logger.info(f"Using codec: {codec}")
                    break
                else:
                    logger.warning(f"Failed to use codec: {codec}")
                    out.release()
            
            if not out or not out.isOpened():
                logger.error(f"Cannot create output video with any codec: {output_path}. Check codec support and file permissions.")
                return False
            
            # Create subtitle mask
            mask_template = self._create_subtitle_mask(height, width)
            
            # Jump to start frame and read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, first_frame = cap.read()
            if not ret:
                logger.error("Cannot read first frame")
                return False
            
            # Initialize tracking
            gray_prev = cv2.cvtColor(cv2.UMat(first_frame), cv2.COLOR_BGR2GRAY)
            current_points = self._initialize_tracks(gray_prev, mask_template)
            
            # Process frames
            frame_idx = start_frame
            start_time = time.time()
            
            with tqdm(total=num_frames_to_process, unit="frame", desc="Processing", ncols=80) as pbar:
                while frame_idx < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Cannot read frame {frame_idx}")
                        break
                    
                    gray_curr = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2GRAY)
                    
                    # Update tracks
                    current_points = self._update_tracks(gray_prev, gray_curr, current_points)
                    
                    # Detect new features periodically
                    if frame_idx % self.config.redetect_interval == 0:
                        current_points = self._detect_new_features(gray_curr, mask_template)
                    
                    # Create output frame
                    canvas = frame.copy()
                    self._draw_trails(canvas)
                    self._draw_patches(canvas, frame, (height, width))
                    
                    # Write frame
                    out.write(canvas)
                    
                    gray_prev = gray_curr
                    frame_idx += 1
                    pbar.update(1)
            
            processing_time = time.time() - start_time
            logger.info(f"Processing finished in {processing_time:.1f}s. Output saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return False
        
        finally:
            cap.release()
            out.release()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generative video tracker with optical flow")
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output_tracked.mp4', help='Output video path')
    parser.add_argument('--start-frame', type=int, default=0, help='First frame to process')
    parser.add_argument('--end-frame', type=int, default=None, help='Last frame to process (exclusive)')
    parser.add_argument('--patch-size', type=int, default=28, help='Target patch size in pixels')
    parser.add_argument('--min-patch-size', type=int, default=14, help='Minimum patch size in pixels')
    parser.add_argument('--max-tracks', type=int, default=250, help='Maximum number of tracks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = TrackerConfig()
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.min_patch_size:
        config.min_patch_size = args.min_patch_size
    if args.max_tracks:
        config.max_tracks = args.max_tracks
    
    # Validate configuration
    try:
        config._validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Process video
    processor = VideoProcessor(config)
    success = processor.process_video(args.input, args.output, args.start_frame, args.end_frame)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
