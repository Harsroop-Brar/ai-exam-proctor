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
        
        # Create GUI
        self.create_widgets()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        
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
        
        # Violation display
        self.violation_label = ttk.Label(self.root, text="Violations: 0", font=('Helvetica', 12))
        self.violation_label.pack(pady=5)
        
        # Log box
        self.log_text = tk.Text(self.root, height=10, width=80)
        self.log_text.pack(pady=10, padx=10)
        self.log_text.insert(tk.END, "System ready...\n")
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def start_monitoring(self):
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
        ret, frame = self.cap.read()
        
        if ret:
            if self.is_monitoring:
                frame = self.detect_violations(frame)
            
            # Convert to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            
            # Convert to ImageTk format
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        # Repeat every 15ms
        self.root.after(15, self.update_video_feed)
    
    def detect_violations(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # No face detected
        if len(faces) == 0:
            no_face_time = time.time() - self.last_face_time
            if no_face_time > 3:  # 3 seconds threshold
                cv2.putText(frame, "VIOLATION: No face detected!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.record_violation("No face detected")
        else:
            self.last_face_time = time.time()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Multiple faces detected
            if len(faces) > 1:
                cv2.putText(frame, "VIOLATION: Multiple people!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.record_violation("Multiple faces detected")
        
        return frame
    
    def record_violation(self, violation_type):
        self.violation_count += 1
        self.violation_label.config(text=f"Violations: {self.violation_count}")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {violation_type}\n"
        self.log(log_message)
        
        # Take screenshot as evidence
        if not os.path.exists("violations"):
            os.makedirs("violations")
        screenshot_path = f"violations/violation_{timestamp.replace(':', '-')}.png"
        cv2.imwrite(screenshot_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
    
    def log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
    
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExamProctorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()