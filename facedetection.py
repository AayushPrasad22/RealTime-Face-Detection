import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class RealTimeFaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Face Detection")

        # Create a label to show the video feed
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Create a text box to display the number of detected faces
        self.result_text = tk.Text(root, height=1, width=20)
        self.result_text.pack(pady=10)

        # Load the pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert image for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Update video label with the new frame
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image

            # Update the text box with the number of detected faces
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"Faces: {len(faces)}")

        # Call this method again after 10 milliseconds
        self.root.after(10, self.update_frame)

    def __del__(self):
        # Release the video capture object when the app is closed
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeFaceDetectionApp(root)
    root.mainloop()
