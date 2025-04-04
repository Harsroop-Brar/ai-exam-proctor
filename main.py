import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
import os

class ExamProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Exam Proctor")
        self.root.geometry("800x600")
        
        # Variables
        self.is_monitoring = False
        self.violation_count = 0
        self.last_face_time = time.time()
        self.last_no_face_alert_time = 0  # Track last no-face alert
        self.current_frame = None
        self.debug_mode = True
        
        # Create GUI
        self.create_widgets()
        
        # Camera setup
        self.cap = None
        self.reinitialize_camera()
        
        # Start the video feed
        self.update_video_feed()
    
    def create_widgets(self):
        # Video frame
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(control_frame, text="Reset Camera", command=self.reinitialize_camera)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Violation display
        self.violation_label = ttk.Label(self.root, text="Violations: 0", font=('Helvetica', 12))
        self.violation_label.pack(pady=5)
        
        # Log box
        self.log_text = tk.Text(self.root, height=10, width=80)
        self.log_text.pack(pady=10, padx=10)
        self.log_text.insert(tk.END, "System initialized. Click 'Start Monitoring' to begin.\n")
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def reinitialize_camera(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                self.log(f"Camera initialized with backend: {backend}")
                # Set reasonable resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return True
                
        self.log("Failed to initialize camera!")
        return False
    
    def start_monitoring(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
            
        self.is_monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log("Monitoring started...")
    
    def stop_monitoring(self):
        self.is_monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Monitoring stopped")
    
    def update_video_feed(self):
        start_time = time.time()
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.log("Camera feed interrupted - attempting to reconnect...")
                self.reinitialize_camera()
                # Show black screen
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "RECONNECTING...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                img = Image.fromarray(error_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            else:
                self.current_frame = frame.copy()
                
                if self.is_monitoring:
                    frame = self.detect_violations(frame)
                
                # Convert and display frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                
        except Exception as e:
            self.log(f"Error in video feed: {str(e)}")
            # Show error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "CAMERA ERROR", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img = Image.fromarray(error_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        finally:
            # Adjust delay to maintain ~30 FPS
            processing_time = time.time() - start_time
            delay = max(1, int(15 - (processing_time * 1000)))
            self.root.after(delay, self.update_video_feed)
    
    def detect_violations(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        
        # No face detected
        if len(faces) == 0:
            no_face_time = current_time - self.last_face_time
            if no_face_time > 3:  # 3 seconds threshold before warning
                # Only trigger violation if 5 seconds have passed since last alert
                if current_time - self.last_no_face_alert_time >= 5:
                    cv2.putText(frame, "VIOLATION: No face detected!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.record_violation("No face detected")
                    self.last_no_face_alert_time = current_time
        else:
            self.last_face_time = current_time
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Multiple faces detected (no cooldown for this violation)
            if len(faces) > 1:
                cv2.putText(frame, "VIOLATION: Multiple people!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.record_violation("Multiple faces detected")
        
        return frame
    
    def record_violation(self, violation_type):
        try:
            if self.current_frame is None:
                self.log("Cannot record violation - no frame available")
                return
                
            self.violation_count += 1
            self.violation_label.config(text=f"Violations: {self.violation_count}")
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_message = f"[{timestamp}] {violation_type}\n"
            self.log(log_message)
            
            # Ensure violations directory exists
            os.makedirs("violations", exist_ok=True)
            
            # Save screenshot
            screenshot_path = f"violations/violation_{timestamp}.png"
            cv2.imwrite(screenshot_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            self.log(f"Error recording violation: {str(e)}")
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        if self.debug_mode:
            print(log_entry.strip())
    
    def on_closing(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExamProctorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
