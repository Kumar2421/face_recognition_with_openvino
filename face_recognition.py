import datetime
import sys
import os
import mysql.connector
from time import perf_counter
import cv2
from openvino.runtime import Core, get_version
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from model_api.performance_metrics import PerformanceMetrics
from datetime import datetime, timedelta, timezone
# Global variables
unique_id_counter = 0
captured_images_dir = "captured_images"
image_path = None  # Define image_path as a global variable

# Connect to the database
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Update with your database password
        database="demo"
    )

# Establish a connection to the database
db_connection = connect_to_database()
db_cursor = db_connection.cursor()

# Connect to the camera
def connect_to_camera(ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"
    vs = cv2.VideoCapture(url)
    return vs

source_1 = connect_to_camera('192.168.29.208', 'admin', 'Libs2000@')
source_2 = connect_to_camera('192.168.29.209', 'admin', 'Libs2000@')

# Model paths
device = "CPU"
faceDETECT = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\face-detection-retail-0005.xml"
faceLANDMARK = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\landmarks-regression-retail-0009.xml"
faceIDENTIFY = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\face-reidentification-retail-0095.xml"

# Frame processor class
class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self):
        self.core = Core()
        self.face_detector = FaceDetector(self.core, faceDETECT, input_size=(0, 0), confidence_threshold=0.6)
        self.landmarks_detector = LandmarksDetector(self.core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(self.core, faceIDENTIFY, match_threshold=0.7, match_algo='HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face_img', self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if len(rois) > self.QUEUE_SIZE:
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return rois, landmarks, face_identities

# Function to draw face detections on the frame
# Function to draw face detections on the frame and capture identified persons
# Function to draw face detections on the frame and capture identified persons
def draw_face_detection(frame, frame_processor, detections):
    size = frame.shape[:2]
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        face_point = (xmin, ymin)
        for point in landmarks:
            x = int(xmin + roi.size[0] * point[0])
            y = int(ymin + roi.size[1] * point[1])
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)
        image_recognizer(frame, text, identity, (xmin, ymin, xmax, ymax), 0.75)  # Pass face region to image_recognizer
    return frame

# Function to capture and save the image of the identified person
# Function to capture and save the image of the identified person
def capture_and_save_image(frame, text, face_rect):
    global unique_id_counter
    global image_path  # Access the global image_path variable
    if not os.path.exists(captured_images_dir):
        try:
            os.makedirs(captured_images_dir)
        except OSError as e:
            print(f"Failed to create directory: {e}")
            return
    face_x, face_y, face_w, face_h = face_rect
    face_image = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    image_name = f"{text}-{unique_id_counter}.jpg"
    image_path = os.path.join(captured_images_dir, image_name)
    cv2.imwrite(image_path, face_image)
    unique_id_counter += 1
    print(f"Image of {text} saved as {image_path}")
    return image_path  # Return the updated image path

# Function to recognize the image and store in the database
def image_recognizer(frame, text, identity, face_rect, threshold):
    xmin, ymin, xmax, ymax = face_rect
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1 - identity.distance) > threshold:
           
            current_time = datetime.now().replace(microsecond=0)  # Remove milliseconds
            entry_date = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).date()
            entry_time = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).time()
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin + textsize[1]), (0 ,0, 0), cv2.FILLED)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0)
            image_path = capture_and_save_image(frame, text, (xmin, ymin, xmax - xmin, ymax - ymin))  # Update image_path
            try:
                # Insert entry record into the database
                query = "INSERT INTO face_recognition (empl_id, camera, entry_date, entry_time, exit_date, exit_time, image_path) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                db_cursor.execute(query, (text, 'unknown', entry_date, entry_time, None, None, image_path))
                db_connection.commit()
                print("Person identified and entry record saved.")
                print(f"Name: {text}, Entry Date: {entry_date}, Entry Time: {entry_time}, Image Path: {image_path}")
            except mysql.connector.Error as err:
                print(f"Error inserting entry record: {err}")
        else:
            textsize = cv2.getTextSize("Unknown", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin + textsize[1]), (255, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Unknown", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

# Function to recognize the image and store in the database
# Initialize frame processors
frame_processor_1 = FrameProcessor()
frame_processor_2 = FrameProcessor()

# Main loop for video processing
while True:
    start_time = perf_counter()
    ret_1, frame_1 = source_1.read()
    ret_2, frame_2 = source_2.read()
    
    if not ret_1 or not ret_2:
        break

    detections_1 = frame_processor_1.face_process(frame_1)
    frame_1 = draw_face_detection(frame_1, frame_processor_1, detections_1)
    
    detections_2 = frame_processor_2.face_process(frame_2)
    frame_2 = draw_face_detection(frame_2, frame_processor_2, detections_2)
    
    cv2.imshow("Camera 1 - Face Recognition Demo", frame_1)
    cv2.imshow("Camera 2 - Face Recognition Demo", frame_2)
    
    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:
        break

source_1.release()
source_2.release()
cv2.destroyAllWindows()

# Close the database connection
db_cursor.close()
db_connection.close()
