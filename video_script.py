import cv2
import numpy as np
import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.randsvd_algorithm import randSVD


def load_video_frames(input_path, max_frames=100):
    """Load video frames and return video data and metadata."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    if max_frames is None:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.astype(np.float32))
    else:       
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.astype(np.float32))
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames read from video.")
    
    print(f"Loaded {len(frames)} frames (capped at {max_frames})")
    frames = np.stack(frames, axis=0)  # shape: (T, H, W, 3)
    T, H, W, C = frames.shape
    
    if C != 3:
        raise ValueError("Expected a 3-channel (color) video.")
    
    return frames, fps, width, height


class ProgressMonitor:
    """Monitor progress and print updates every minute."""
    def __init__(self, start_time, interval_seconds=60):
        self.start_time = start_time
        self.interval_seconds = interval_seconds
        self.last_print_minute = 0
        self.running = True
        self.thread = None
    
    def start(self):
        """Start the monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor(self):
        """Monitor elapsed time and print updates."""
        while self.running:
            elapsed = time.time() - self.start_time
            elapsed_minutes = int(elapsed / 60)
            
            if elapsed_minutes > self.last_print_minute:
                self.last_print_minute = elapsed_minutes
                print(f"  [Progress] Standard SVD has been running for {elapsed_minutes} minute(s) ({elapsed/60:.1f} minutes total)")
            
            time.sleep(5)


def save_video_outputs(bg_frames, fg_frames, fps, width, height, bg_output_path, fg_output_path):
    """Save background and foreground frames to video files."""
    bg_frames_clipped = np.clip(bg_frames, 0, 255).astype(np.uint8)
    fg_frames_clipped = np.clip(fg_frames, 0, 255).astype(np.uint8)
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_bg = cv2.VideoWriter(bg_output_path, fourcc, fps, (width, height))
    out_fg = cv2.VideoWriter(fg_output_path, fourcc, fps, (width, height))
    
    T = bg_frames_clipped.shape[0]
    print(f"Saving {T} frames")
    for t in range(T):
        out_bg.write(bg_frames_clipped[t])
        out_fg.write(fg_frames_clipped[t])
    
    out_bg.release()
    out_fg.release()

def separate_foreground_background_randSVD(
    input_path,
    bg_output_path,
    fg_output_path,
    k=5,
    p=5,
    q=1,
    sketch_type='gaussian',
    use_centering=True,
    max_frames=100
):
    """
    Separate foreground and background using Randomized SVD.
    
    input_path:      path to input color MP4
    bg_output_path:  path to write background video (MP4)
    fg_output_path:  path to write foreground video (MP4)
    k:               target rank for randSVD
    p:               oversampling parameter
    q:               number of power iterations
    sketch_type:     sketch type for randSVD (e.g. 'gaussian', 'srht', ...)
    use_centering:   subtract temporal mean frame before SVD
    max_frames:      maximum number of frames to process
    
    Returns:
        elapsed_time: Time taken for SVD computation (seconds)
    """
  
    frames, fps, width, height = load_video_frames(input_path, max_frames)
    T, H, W, C = frames.shape

    mean_frame = 0
    if use_centering:
        mean_frame = frames.mean(axis=0, keepdims=True)
        frames = frames - mean_frame


    bg_centered = np.zeros_like(frames)
    
    svd_start_time = time.time()
    for c in range(3):
        channel_start = time.time()
        print(f"Running randSVD for channel {c}")
        channel_data = frames[:, :, :, c]
        A = channel_data.reshape(T, H * W).T  # (H*W, T)
        

        U, S, Vt = randSVD(A, k, p, q=q, sketch_type=sketch_type)
        print(f"Done with randSVD for channel {c}")

        S_mat = np.diag(S)
        A_bg = U @ S_mat @ Vt
        channel_bg = A_bg.T.reshape(T, H, W)
        bg_centered[:, :, :, c] = channel_bg
        
        channel_time = time.time() - channel_start
        print(f"  Channel {c} completed in {channel_time:.2f} seconds")
    
    svd_elapsed_time = time.time() - svd_start_time


    if use_centering:
        bg_centered = bg_centered + mean_frame
    
    residual = frames - bg_centered
    fg_frames = np.abs(residual)


    save_video_outputs(bg_centered, fg_frames, fps, width, height, bg_output_path, fg_output_path)
    
    return svd_elapsed_time


def separate_foreground_background_standardSVD(
    input_path,
    k=5,
    use_centering=True,
    max_frames=100
):
    """
    Separate foreground and background using standard (non-randomized) SVD.
    
    input_path:      path to input color MP4
    k:               target rank for SVD (number of singular values to keep)
    use_centering:   subtract temporal mean frame before SVD
    max_frames:      maximum number of frames to process
    
    Returns:
        elapsed_time: Time taken for SVD computation (seconds)
    """

    frames, fps, width, height = load_video_frames(input_path, max_frames)
    T, H, W, C = frames.shape

    if use_centering:
        mean_frame = frames.mean(axis=0, keepdims=True)
        frames_centered = frames - mean_frame
    else:
        mean_frame = np.zeros((1, H, W, 3), dtype=np.float32)
        frames_centered = frames.copy()

    bg_centered = np.zeros_like(frames_centered)
    
    svd_start_time = time.time()
    

    progress_monitor = ProgressMonitor(svd_start_time, interval_seconds=60)
    progress_monitor.start()
    
    try:
        for c in range(3):
            channel_start = time.time()
            print(f"Running standard SVD for channel {c}")
            channel_data = frames_centered[:, :, :, c]
            A = channel_data.reshape(T, H * W).T  # (H*W, T)
            
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
        
            S_mat = np.diag(S_k)
            A_bg = U_k @ S_mat @ Vt_k
            channel_bg = A_bg.T.reshape(T, H, W)
            bg_centered[:, :, :, c] = channel_bg
            
            channel_time = time.time() - channel_start
            channel_minutes = channel_time / 60
            print(f"  Channel {c} completed in {channel_time:.2f} seconds ({channel_minutes:.2f} minutes)")
    finally:
        progress_monitor.stop()
    
    svd_elapsed_time = time.time() - svd_start_time
    
    return svd_elapsed_time


def compare_svd_methods(
    input_path,
    k=5,
    p=5,
    q=1,
    sketch_type='gaussian',
    use_centering=True,
    max_frames=100
):
    """
    Compare runtime of Randomized SVD vs Standard SVD for foreground/background separation.
    
    Args:
        input_path: Path to input video
        k: Target rank for SVD
        p: Oversampling parameter for randomized SVD
        q: Number of power iterations for randomized SVD
        sketch_type: Sketch type for randomized SVD
        use_centering: Whether to use temporal mean centering
        max_frames: Maximum number of frames to process
    
    Returns:
        Dictionary with timing results and speedup factor
    """
    base_dir = Path(input_path).parent
    
    print("=" * 70)
    print("Comparing Randomized SVD vs Standard SVD")
    print("=" * 70)
    print(f"Parameters: k={k}, max_frames={max_frames}, sketch_type={sketch_type}")
    print("-" * 70)
    
    print("\n[1/2] Running Randomized SVD...")
    rand_time = separate_foreground_background_randSVD(
        input_path=input_path,
        bg_output_path=str(base_dir / "background_randsvd.mp4"),
        fg_output_path=str(base_dir / "foreground_randsvd.mp4"),
        k=k,
        p=p,
        q=q,
        sketch_type=sketch_type,
        use_centering=use_centering,
        max_frames=max_frames
    )

    
    print("\n[2/2] Running Standard SVD...")
    print("This may take several minutes. Progress updates will be printed every minute.")
    std_time = separate_foreground_background_standardSVD(
        input_path=input_path,
        bg_output_path=str(base_dir / "background_standardsvd.mp4"),
        fg_output_path=str(base_dir / "foreground_standardsvd.mp4"),
        k=k,
        use_centering=use_centering,
        max_frames=max_frames
    )



if __name__ == "__main__":
    current_dir = Path(__file__).parent
    input_video = str(current_dir / "test_video_large.mp4")
    
    compare_svd_methods(
        input_path=input_video,
        k=3,
        p=10,
        q=1,
        sketch_type='gaussian',
        use_centering=False,
        max_frames=200
    )




# Output:

# ======================================================================
# Comparing Randomized SVD vs Standard SVD
# ======================================================================
# Parameters: k=3, max_frames=200, sketch_type=gaussian
# ----------------------------------------------------------------------

# [1/2] Running Randomized SVD...
# Loaded 200 frames (capped at 200)
# Running randSVD for channel 0
# Done with randSVD for channel 0
#   Channel 0 completed in 1031.33 seconds
# Running randSVD for channel 1
# Done with randSVD for channel 1
#   Channel 1 completed in 122.86 seconds
# Running randSVD for channel 2
# Done with randSVD for channel 2
#   Channel 2 completed in 141.09 seconds



# [2/2] Running Standard SVD...
# This may take several minutes. Progress updates will be printed every minute.
# Loaded 200 frames (capped at 200)
# Running standard SVD for channel 0
#   [Progress] Standard SVD has been running for 1 minute(s) (1.0 minutes total)
#   Channel 0 completed in 108.08 seconds (1.80 minutes)
# Running standard SVD for channel 1
#   [Progress] Standard SVD has been running for 2 minute(s) (2.0 minutes total)
#   [Progress] Standard SVD has been running for 3 minute(s) (3.0 minutes total)
#   Channel 1 completed in 118.99 seconds (1.98 minutes)
# Running standard SVD for channel 2
#   [Progress] Standard SVD has been running for 4 minute(s) (4.0 minutes total)
#   [Progress] Standard SVD has been running for 11 minute(s) (11.4 minutes total)
#   [Progress] Standard SVD has been running for 12 minute(s) (12.0 minutes total)
#   [Progress] Standard SVD has been running for 34 minute(s) (34.3 minutes total)
#   [Progress] Standard SVD has been running for 116 minute(s) (116.8 minutes total)
#   [Progress] Standard SVD has been running for 117 minute(s) (117.0 minutes total)
#   [Progress] Standard SVD has been running for 155 minute(s) (155.4 minutes total)
#   Channel 2 completed in 9107.25 seconds (151.79 minutes)
