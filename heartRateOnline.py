import csv
import numpy as np
import cv2
import time
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone

class HeartRateMonitor:
    def __init__(self, camera_index=0, run_duration=60, save_bpm=True, show_plot=True, , color=(255, 0, 255)):

        self.color = color



        # Settings
        self.SAVE_BPM = save_bpm
        self.SHOW_PLOT = show_plot
        self.CAMERA_INDEX = camera_index
        self.RUN_DURATION = run_duration  # in seconds (0 = unlimited)

        # Webcam settings
        self.real_width, self.real_height = 640, 480
        self.video_width, self.video_height = 160, 120
        self.video_channels = 3
        self.video_frame_rate = 15

        # Initialize webcam
        self.webcam = cv2.VideoCapture(self.CAMERA_INDEX)
        self.webcam.set(3, self.real_width)
        self.webcam.set(4, self.real_height)

        # Face detection
        self.detector = FaceDetector()

        # Visualization
        self.plot_y = LivePlot(320, 240, [60, 120], invert=True, color=(0, 0, 255)) if self.SHOW_PLOT else None
#        self.plot_y = LivePlot(320, 240, [60, 120], invert=True) if self.SHOW_PLOT else None


        # Signal processing parameters
        self.levels = 3
        self.alpha = 170
        self.min_freq, self.max_freq = 1.0, 2.0
        self.buffer_size = 150
        self.buffer_index = 0

        # BPM calculation
        self.bpm_calc_freq = 10
        self.bpm_buffer_size = 10
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))
        self.bpm_index = 0
        self.bpm_count = 0

        # Frequency analysis
        self.frequencies = self.video_frame_rate * np.arange(self.buffer_size) / self.buffer_size
        self.mask = (self.frequencies >= self.min_freq) & (self.frequencies <= self.max_freq)

        # Initialize buffers
        zero_frame = np.zeros((self.video_height, self.video_width, self.video_channels))
        gauss_base = [cv2.pyrDown(zero_frame) for _ in range(self.levels+1)][-1]
        self.video_gauss = np.zeros((self.buffer_size, *gauss_base.shape))
        self.fourier_avg = np.zeros((self.buffer_size))

        # Data logging
        self.bpm_log = []
        self.start_time = time.time()
        self.ptime = 0

    def build_gauss_pyramid(self, frame, levels):
        """Build Gaussian pyramid for the input frame"""
        pyramid = [frame]
        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstruct_frame(self, pyramid, index, levels):
        """Reconstruct frame from Gaussian pyramid"""
        frame = pyramid[index]
        for _ in range(levels):
            frame = cv2.pyrUp(frame)
        return frame[:self.video_height, :self.video_width]

    def process_frame(self, frame):
        """Process a single frame for heart rate detection"""
        frame, faces = self.detector.findFaces(frame, draw=False)
        frame_draw = frame.copy()
        
        # Calculate FPS
        now = time.time()
        fps = 1 / (now - self.ptime) if self.ptime != 0 else 0
        self.ptime = now

        if faces:
            x, y, w, h = faces[0]['bbox']
            if w > 50 and h > 50:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (640, 480))  # Resize for processing

                # Build Gaussian pyramid and store in buffer
                self.video_gauss[self.buffer_index] = self.build_gauss_pyramid(face_roi, self.levels+1)[self.levels]
                
                # Perform FFT analysis
                fft_result = np.fft.fft(self.video_gauss, axis=0)
                fft_result[~self.mask] = 0

                # Calculate BPM periodicallyq
                if self.buffer_index % self.bpm_calc_freq == 0:
                    self.bpm_count += 1
                    for i in range(self.buffer_size):
                        self.fourier_avg[i] = np.real(fft_result[i]).mean()
                    bpm = 60 * self.frequencies[np.argmax(self.fourier_avg)]
                    self.bpm_buffer[self.bpm_index] = bpm
                    self.bpm_index = (self.bpm_index + 1) % self.bpm_buffer_size
                    self.bpm_log.append((int(time.time() - self.start_time), round(bpm, 2)))

                # Reconstruct the amplified signal
                filtered = np.real(np.fft.ifft(fft_result, axis=0)) * self.alpha
                reconstructed = self.reconstruct_frame(filtered, self.buffer_index, self.levels)
                reconstructed_resized = cv2.resize(reconstructed, (640, 480))
                result = face_roi + reconstructed_resized
                result = cv2.convertScaleAbs(result)

                self.buffer_index = (self.buffer_index + 1) % self.buffer_size

                # Prepare visualization
                bpm_value = self.bpm_buffer.mean()
                bpm_plot = self.plot_y.update(float(bpm_value)) if self.SHOW_PLOT else np.zeros((240, 320, 3), dtype=np.uint8)

                small_result = cv2.resize(result, (320, 240))
                bottom = np.hstack((small_result, bpm_plot))
                top = frame_draw

                if self.bpm_count > self.bpm_buffer_size:
                    cvzone.putTextRect(top, f"BPM: {int(bpm_value)}", (20, 30), scale=2,  colorT=(0, 255, 0), colorR=(0, 0, 0) )
                    #cvzone.putTextRect(top, "Calculating BPM...", (20, 30), scale=2)

                else:
                    cvzone.putTextRect(top, "Calculating BPM...", (20, 30), scale=2, colorT=(0, 255, 0), colorR=(0, 0, 0))

                combined = np.vstack((top, bottom))
                cv2.putText(combined, "Press Q to Quit", (10, 70), 0, 1, (255, 255, 255), 2)
                return combined

        # No face detected casw
        cvzone.putTextRect(frame_draw, "No Face Detected", (200, 40), scale=2, colorR=(255, 0, 0))
        bottom = np.zeros((240, 640, 3), dtype=np.uint8)
        combined = np.vstack((frame_draw, bottom))
        return combined

    def run(self):
        """Main loop to run the heart rate monitor"""
        while True:
            success, frame = self.webcam.read()
            if not success:
                break

            output_frame = self.process_frame(frame)
            cv2.imshow("Heart Rate Monitor", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.RUN_DURATION > 0 and time.time() - self.start_time > self.RUN_DURATION:
                print("Duration finished. Exiting...")
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources and save data"""
        self.webcam.release()
        cv2.destroyAllWindows()

        if self.SAVE_BPM and self.bpm_log:
            with open("bpm_log.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "BPM"])
                writer.writerows(self.bpm_log)
            print("BPM log saved to bpm_log.csv")

if __name__ == "__main__":
    monitor = HeartRateMonitor(camera_index=1, run_duration=120, save_bpm=True, show_plot=True)
    monitor.run()