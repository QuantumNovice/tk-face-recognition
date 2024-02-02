import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pickle 
import numpy as np
import face_recognition
import os
from threading import Thread

class FaceRecognition:
    def __init__(self):
        # Threshold for face recognition (adjust as needed)
        self.face_recognition_threshold = 0.6
        
        # Initialize variables
        self.face_id_counter = 0
        self.known_face_encodings = {}
        self.known_face_names = {}
        
        # Initialize variables
        self.face_id_counter, self.known_face_encodings, self.known_face_names = self.load_known_faces()

    # Function to save the known faces information to a Pickle file
    def save_known_faces(self):
        data = {
            "face_id_counter": self.face_id_counter,
            "known_face_encodings": self.known_face_encodings,
            "known_face_names": self.known_face_names
        }
        with open('known_faces.pkl', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    # Function to load the known faces information from a Pickle file
    def load_known_faces(self):
        if os.path.exists('known_faces.pkl'):
            with open('known_faces.pkl', 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                return data.get("face_id_counter", 0), data.get("known_face_encodings", {}), data.get("known_face_names", {})
        else:
            return 0, {}, {}
        
    def find_faces(self, frame):
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # Create a copy of the frame for the current face
            face_image = frame[top:bottom, left:right]

            # Construct a unique filename based on the face encoding
            filename = os.path.join("faces", f"face_{self.face_id_counter}.jpg")
            
            if not os.path.exists(filename):
                # Save the face image
                cv2.imwrite(filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                print("saved image")

            # Check if the face is known based on face recognition
            matches = face_recognition.compare_faces(list(self.known_face_encodings.values()), face_encoding, tolerance=self.face_recognition_threshold)
            face_name = "Unknown"

            # If the face is known, retrieve the ID
            if True in matches:
                matched_index = matches.index(True)
                face_id = list(self.known_face_encodings.keys())[matched_index]
                face_name = self.known_face_names[face_id]

            # If the face is unknown, assign a new ID
            else:
                self.face_id_counter += 1
                face_id = self.face_id_counter
                self.known_face_encodings[face_id] = face_encoding
                self.known_face_names[face_id] = f"Person {face_id}"
                self.save_known_faces()

            if not os.path.exists('detected_faces'):
                # Save the new face as an image file in the "detected_faces" directory
                cv2.imwrite(f"detected_faces/face_{face_id}.jpg", frame[top:bottom, left:right])
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, face_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        

class VideoApp:
    def __init__(self, master):
        self.master = master
        self.cap = cv2.VideoCapture(0)
        self.face_recognition = FaceRecognition()
        self.create_widgets()

    def create_widgets(self):
        self.ret, self.frame = self.cap.read()

        self.image = Image.fromarray(self.frame)
        self.photo = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(self.master, width=self.photo.width(), height=self.photo.height())
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.create_image((0, 0), image=self.photo, anchor='nw')

        self.description = tk.Label(self.master, text="Place for description")
        self.description.pack(side='right')

        self.button =  tk.Button(self.master, text="Show known faces", command=self.show_known_faces)
        self.button.pack()

    def show_known_faces(self):
        # Create a new window to show the list of known faces with a scrollable frame
        window = tk.Toplevel(self.master)
        window.title("Known Faces")

        # Path to the "faces" folder
        faces_folder = "faces"

        # Get a list of image files in the "faces" folder
        image_files = [f for f in os.listdir(faces_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Create a scrollable canvas and a scrollbar
        canvas = tk.Canvas(window)
        scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Function to resize the canvas when the window size changes
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        window.bind("<Configure>", on_configure)

        for image_file in image_files:
            # Load and display each image alongside its filename
            img_path = os.path.join(faces_folder, image_file)
            img = Image.open(img_path)
            img.thumbnail((100, 100))  # Resize the image to fit the window
            img = ImageTk.PhotoImage(img)

            label_image = tk.Label(frame, image=img, text=image_file, compound=tk.TOP)
            label_image.image = img
            label_image.pack()



    def update_frame(self):
        self.ret, self.frame = self.cap.read()

        if self.ret:
            # Pass the frame to the face recognition class
            self.face_recognition.find_faces(self.frame)

            # Update the Tkinter window with the processed frame
            image = Image.fromarray(self.frame)
            self.photo.paste(image)

        self.master.after(10, self.update_frame)  # update it again after 10ms

    def run(self):
        self.update_frame()  # update it first time
        self.master.protocol("WM_DELETE_WINDOW", self.close)  # handle window close event
        self.master.mainloop()  # start program - this loop runs all the time

    def close(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    app.run()
