import csv
import numpy as np
import cv2
import time
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
from scipy.signal import butter, filtfilt, welch

class HeartRateMonitor:
    def __init__(self):
        # Configuration settings
        self.SAVE_BPM = True
        self.SHOW_PLOT = True
        self.CAMERA_INDEX = 1  # Typically 0 for default camera
        self.RUN_DURATION = 60  # in seconds (0 = unlimited)
        
        # Signal processing parameters
        self.BPM_MIN, self.BPM_MAX = 40, 240
        self.BUTTER_ORDER = 4
        self.PYR_LEVELS = 2
        self.ROI_FRAC = 0.18  # forehead height fraction
        self.CHEEK_FRAC_Y = 0.35
        self.CHEEK_FRAC_W, self.CHEEK_FRAC_H = 0.12, 0.16
        
        # Webcam settings
        self.real_width, self.real_height = 640, 480
        self.video_frame_rate = 30  # Estimated frame rate
        
        # Initialize components
        self.webcam = self.initialize_webcam()
        self.face_detector = FaceDetector(minDetectionCon=0.5)
        self.bpm_plot = LivePlot(640, 360, [self.BPM_MIN, self.BPM_MAX], invert=True)
        
        # Buffers for signal processing
        self.buffer_size = 256  # Power of 2 for better FFT
        self.buffer_index = 0
        self.rgb_buffer = np.zeros((self.buffer_size, 3, 3))  # time, ROI, RGB
        
        # BPM calculation
        self.bpm_buffer_size = 10
        self.bpm_buffer = np.zeros(self.bpm_buffer_size)
        self.bpm_index = 0
        self.bpm_count = 0
        
        # Data logging
        self.bpm_log = []
        self.start_time = time.time()
        self.prev_time = 0
        self.frame_count = 0

    def initialize_webcam(self):
        """Initialize and configure the webcam."""
        webcam = cv2.VideoCapture(self.CAMERA_INDEX)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.real_width)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.real_height)
        return webcam

    def butter_bandpass_filter(self):
        """Create a Butterworth bandpass filter for heart rate frequencies."""
        nyquist = 0.5 * self.video_frame_rate
        low = self.BPM_MIN / 60 / nyquist
        high = self.BPM_MAX / 60 / nyquist
        b, a = butter(self.BUTTER_ORDER, [low, high], btype='band')
        return b, a

    def pos_algorithm(self, rgb_signals):
        """Apply Plane-Orthogonal-to-Skin algorithm to RGB signals."""
        # Normalize each color channel
        norm_signals = rgb_signals / (np.mean(rgb_signals, axis=0) + 1e-8) - 1
        
        # POS projection matrix
        H = np.array([[0, 1, -1], [-2, 1, 1]])
        projected = (H @ norm_signals.T).sum(axis=0)
        
        # Remove DC component
        return projected - np.mean(projected)

    def calculate_hr(self, signal):
        """Calculate heart rate from PPG signal using Welch's method."""
        f, p = welch(signal, self.video_frame_rate, nperseg=len(signal))
        
        # Find peak in the frequency range of interest
        mask = (f >= self.BPM_MIN/60) & (f <= self.BPM_MAX/60)
        if not np.any(mask):
            return None, 0
        
        f_masked = f[mask]
        p_masked = p[mask]
        peak_idx = np.argmax(p_masked)
        
        # Parabolic interpolation for more accurate peak
        if 0 < peak_idx < len(f_masked)-1:
            y0, y1, y2 = p_masked[peak_idx-1:peak_idx+2]
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-8)
            peak_freq = f_masked[peak_idx] + delta * (f_masked[1] - f_masked[0])
        else:
            peak_freq = f_masked[peak_idx]
        
        # Calculate SNR
        snr = p_masked[peak_idx] / (np.median(p_masked) + 1e-8)
        return peak_freq * 60, snr  # Convert to BPM

    def get_face_rois(self, x, y, w, h):
        """Generate multiple ROIs (forehead and cheeks) from face bounding box."""
        # Forehead ROI (top portion of face)
        forehead_h = int(h * self.ROI_FRAC)
        forehead = (x, y, w, forehead_h)
        
        # Cheek ROIs (middle portion of face)
        cheek_y = y + int(h * self.CHEEK_FRAC_Y)
        cheek_h = int(h * self.CHEEK_FRAC_H)
        cheek_w = int(w * self.CHEEK_FRAC_W)
        
        left_cheek = (x + int(w * 0.15), cheek_y, cheek_w, cheek_h)
        right_cheek = (x + w - int(w * 0.15) - cheek_w, cheek_y, cheek_w, cheek_h)
        
        return [forehead, left_cheek, right_cheek]

    def process_frame(self, frame):
        """Process a frame to extract heart rate information."""
        # Detect faces
        frame, faces = self.face_detector.findFaces(frame, draw=False)
        
        if not faces:
            return frame, None
        
        # Get primary face
        x, y, w, h = faces[0]['bbox']
        
        # Ensure face is large enough
        if w < 50 or h < 50:
            return frame, None
        
        # Get ROIs
        rois = self.get_face_rois(x, y, w, h)
        
        # Extract RGB signals from each ROI
        current_signals = []
        for rx, ry, rw, rh in rois:
            # Ensure ROI is within frame bounds
            ry = max(0, ry)
            rx = max(0, rx)
            rh = min(rh, frame.shape[0] - ry)
            rw = min(rw, frame.shape[1] - rx)
            
            if rh <= 0 or rw <= 0:
                continue
            
            roi = frame[ry:ry+rh, rx:rx+rw]
            if roi.size == 0:
                continue
            
            # Get mean RGB values (convert BGR to RGB)
            rgb = np.mean(roi.reshape(-1, 3), axis=0)[::-1] / 255.0
            current_signals.append(rgb)
        
        # Need at least 2 ROIs for POS algorithm
        if len(current_signals) < 2:
            return frame, None
        
        # Update circular buffer
        self.rgb_buffer[self.buffer_index] = current_signals[:3]  # Take up to 3 ROIs
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.frame_count += 1
        
        # Wait until we have enough samples
        if self.frame_count < self.buffer_size:
            return frame, None
        
        # Process signals
        bpm_values = []
        snr_values = []
        b, a = self.butter_bandpass_filter()
        
        for roi_idx in range(3):  # Process each ROI separately
            # Get all samples for this ROI
            roi_signals = self.rgb_buffer[:, roi_idx, :]
            
            # Skip if we don't have data for this ROI
            if np.all(roi_signals == 0):
                continue
            
            # Apply POS algorithm
            pos_signal = self.pos_algorithm(roi_signals)
            
            # Bandpass filter
            filtered = filtfilt(b, a, pos_signal)
            
            # Calculate heart rate
            hr, snr = self.calculate_hr(filtered)
            if hr is not None:
                bpm_values.append(hr)
                snr_values.append(snr)
        
        # Combine results from multiple ROIs
        if len(bpm_values) >= 2:
            # Take the two best signals (highest SNR)
            best_indices = np.argsort(snr_values)[-2:]
            current_bpm = np.mean([bpm_values[i] for i in best_indices])
            
            # Update BPM buffer
            self.bpm_buffer[self.bpm_index] = current_bpm
            self.bpm_index = (self.bpm_index + 1) % self.bpm_buffer_size
            self.bpm_count += 1
            
            # Log data
            self.bpm_log.append((time.time() - self.start_time, current_bpm))
            
            # Return smoothed BPM value
            smoothed_bpm = np.mean(self.bpm_buffer) if self.bpm_count > self.bpm_buffer_size else current_bpm
            return frame, smoothed_bpm
        
        return frame, None

    def update_display(self, frame, bpm=None):
        """Update the display with current frame and BPM information."""
        display_frame = frame.copy()
        
        # Add BPM text if available
        if bpm is not None:
            if self.bpm_count > self.bpm_buffer_size:
                cvzone.putTextRect(display_frame, f"BPM: {int(bpm)}", 
                                 (20, 70), scale=2, thickness=2)
            else:
                cvzone.putTextRect(display_frame, "Calculating BPM...", 
                                 (20, 70), scale=2, thickness=2)
        else:
            cvzone.putTextRect(display_frame, "No face detected", 
                             (20, 70), scale=2, thickness=2, colorR=(0, 0, 255))
        
        # Add FPS counter
        fps = 1 / (time.time() - self.prev_time)
        cvzone.putTextRect(display_frame, f"FPS: {int(fps)}", 
                          (20, 120), scale=1, thickness=1)
        
        # Update BPM plot
        if self.SHOW_PLOT:
            if bpm is not None:
                plot_img = self.bpm_plot.update(bpm)
            else:
                plot_img = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # Combine frames
            combined = np.vstack((display_frame, plot_img))
        else:
            combined = display_frame
        
        cv2.imshow("Heart Rate Monitor", combined)

    def run(self):
        """Main execution loop for heart rate monitoring."""
        try:
            while True:
                self.prev_time = time.time()
                
                # Read frame
                success, frame = self.webcam.read()
                if not success:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, bpm = self.process_frame(frame)
                
                # Update display
                self.update_display(processed_frame, bpm)
                
                # Check for exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if self.RUN_DURATION > 0 and time.time() - self.start_time > self.RUN_DURATION:
                    print("Duration finished. Exiting...")
                    break
        
        finally:
            # Cleanup
            self.webcam.release()
            cv2.destroyAllWindows()
            
            # Save BPM data if enabled
            if self.SAVE_BPM and self.bpm_log:
                with open("bpm_log.csv", "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time (s)", "BPM"])
                    writer.writerows(self.bpm_log)
                print(f"Saved {len(self.bpm_log)} BPM measurements to bpm_log.csv")

if __name__ == "__main__":
    print("Starting Heart Rate Monitor...")
    monitor = HeartRateMonitor()
    monitor.run()