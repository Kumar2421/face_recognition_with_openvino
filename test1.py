import datetime
import sys
import os
import time
import numpy as np 
import mysql.connector
from time import perf_counter
import cv2
from openvino.runtime import Core, get_version
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from datetime import datetime, timedelta, timezone
from model_api.performance_metrics import PerformanceMetrics

# Global variables

unique_id_counter = 0
captured_images_dir = "captured_images"
image_path = None  
class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self,fqa_model_path):
        self.core = Core()
        self.face_detector = FaceDetector(self.core, faceDETECT, input_size=(0, 0), confidence_threshold=0.5)
        self.landmarks_detector = LandmarksDetector(self.core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(self.core, faceIDENTIFY, match_threshold=0.5, match_algo='HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face_img1', self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)
        # self.fqa_model = face_quality_assessment(fqa_model_path)

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if len(rois) > self.QUEUE_SIZE:
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return rois, landmarks, face_identities
    def assess_face_quality(self, crop_img):
        fqa_probs = self.fqa_model.detect(crop_img)
        fqa_prob_mean = round(np.mean(fqa_probs), 2)
        return fqa_prob_mean
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

# Connect to the cameras
def connect_to_camera(ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"
    vs = cv2.VideoCapture(url)
    return vs

# Initialize camera sources
source_1 = connect_to_camera('192.168.29.208', 'admin', 'Libs2000@')
source_2 = connect_to_camera('192.168.29.209', 'admin', 'Libs2000@')

# Model paths
device = "CPU"
faceDETECT = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\face-detection-retail-0005.xml"
faceLANDMARK = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\model_2022_3\\landmarks-regression-retail-0009.xml"
faceIDENTIFY = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\model_2022_3\\face-reidentification-retail-0095.xml"

# Initialize face processors for each camera
frame_processor_1 = FrameProcessor("C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face-quality-assessment.onnx")
frame_processor_2 = FrameProcessor("C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face-quality-assessment.onnx")
metrics = PerformanceMetrics()
# Fusion parameters
line_position = 400  # Line position to determine fusion threshold
fqa_threshold = 0.2  # Face quality assessment threshold for fusion

def draw_face_detection(frame, frame_processor, detections, camera_id, fqa_threshold):
    size = frame.shape[:2]
    for roi, landmarks, identity in detections:
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        text = text.upper()
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        # crop_img = frame[ymin:ymax, xmin:xmax] 
        # fqa_score = frame_processor.assess_face_quality(crop_img)
        # cv2.putText(frame, "FQA Score: " + str(fqa_score), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        # if fqa_score >= fqa_threshold:
        image_recognizer(frame, text, identity, (xmin, ymin, xmax, ymax), 0.75, camera_id)  
    return frame



unknown_counter = 0

captured_images_dir = "C:\\xampp1\\htdocs\\demo\\captured_images"

def capture_and_save_image(frame, text, roi):
    global unique_id_counter
    global captured_images_dir  # Access the global captured_images_dir variable
    global unknown_counter
    
    try:
        if not os.path.exists(captured_images_dir):
            os.makedirs(captured_images_dir)
        
        size = frame.shape[:2]
        face_x, face_y, face_w, face_h = roi
        margin = 0.2  # Adjust the margin as needed
        xmin = max(int(face_x - margin * face_w), 0)
        ymin = max(int(face_y - margin * face_h), 0)
        xmax = min(int(face_x + face_w + margin * face_w), size[1])
        ymax = min(int(face_y + face_h + margin * face_h), size[0])
        
        face_image = frame[ymin:ymax, xmin:xmax]
        
        # Save the entire face area
        image_name = f"{text}-{unique_id_counter}.jpg"
        image_path = os.path.join(captured_images_dir, image_name)
        cv2.imwrite(image_path, face_image)
        
        # Update counters
        unique_id_counter += 1
        if text.startswith("Unknown"):
            unknown_counter += 1
        
        print(f"Image of {text} saved as {image_path}")
    except Exception as e:
        print(f"An error occurred while capturing and saving the image: {e}")
    return image_path 


def calculate_clarity_score(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian variance as a measure of image sharpness
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    # Normalize the sharpness score to a range between 0 and 1
    normalized_sharpness_score = min(max(laplacian_var / 1000, 0), 1)
    
    return normalized_sharpness_score
def image_recognizer(frame, text, identity, face_rect, threshold, camera_id):
    xmin, ymin, xmax, ymax = face_rect
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1 - identity.distance) > threshold:
            current_time = datetime.now().replace(microsecond=0)  # Remove milliseconds
            entry_date = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).date()
            entry_time = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).time()
            # Capture and save image
            image_path = capture_and_save_image(frame, text, (xmin, ymin, xmax - xmin, ymax - ymin))
            try:
                # Insert entry record into the database
                if camera_id == 'A':
                    query = "INSERT INTO face_recognition (empl_id, camera, entry_date, entry_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                    image_path_with_backslash = image_path.replace('/', '\\')  # Replace forward slash with backslash
                    db_cursor.execute(query, (text, 'Camera A', entry_date, entry_time.strftime("%I:%M %p"), image_path_with_backslash))
                    print("Person identified and entry record saved for Camera A.")
                elif camera_id == 'B':
                    query = "INSERT INTO face_recognition (empl_id, camera, exit_date, exit_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                    image_path_with_backslash = image_path.replace('/', '\\')  # Replace forward slash with backslash
                    db_cursor.execute(query, (text, 'Camera B', entry_date, entry_time.strftime("%I:%M %p"), image_path_with_backslash))
                    print("Person identified and entry record saved for Camera B.")
                    db_connection.commit()  # Commit the transaction
            except Exception as e:
                db_connection.rollback()  # Rollback in case of an error
                print("Error occurred:", e)
        else:
            # Capture and save image of unknown person
            image_path = capture_and_save_image(frame, f"Unknown", (xmin, ymin, xmax - xmin, ymax - ymin))
            print(f"Unknown person detected. Image saved as {image_path}")
            try:
                # Insert entry record into the database for unknown person
                current_time = datetime.now().replace(microsecond=0)  # Remove milliseconds
                entry_date = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).date()
                entry_time = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).time()
                query = "INSERT INTO face_recognition (empl_id, camera, entry_date, entry_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                db_cursor.execute(query, (f"Unknown", camera_id, entry_date, entry_time.strftime("%I:%M %p"), image_path))
                print("Entry record saved for unknown person.")
                db_connection.commit()  # Commit the transaction
            except Exception as e:
                db_connection.rollback()  # Rollback in case of an error
                print("Error occurred:", e)
    else:
        print("Unknown face detected.")

# Main loop to read frames from both cameras
while True:
    start_time = perf_counter()
    ret_1, frame_1 = source_1.read()
    ret_2, frame_2 = source_2.read()
    
    if not ret_1 or not ret_2:
        break

    # Draw fusion line on frames
    cv2.line(frame_1, (0, line_position), (frame_1.shape[1], line_position), (255, 255, 255), 1)
    cv2.line(frame_2, (0, line_position), (frame_2.shape[1], line_position), (255, 255, 255), 1)

    # Process frames from camera 1
    if frame_1 is not None:
        detections_1 = frame_processor_1.face_process(frame_1)
        for roi, landmarks, identity in zip(*detections_1):
        # Get coordinates of the bounding box
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), frame_1.shape[1])
            ymax = min(int(roi.position[1] + roi.size[1]), frame_1.shape[0])
            if ymax >= line_position:
                # Draw bounding box for faces below the line
                cv2.rectangle(frame_1, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
                # Send for recognition if below the line
                frame_1 = draw_face_detection(frame_1, frame_processor_1, [(roi, landmarks, identity)], camera_id="A", fqa_threshold=fqa_threshold)
  # Adjust threshold as needed
            else:
                # Draw bounding box for faces above the line
                cv2.rectangle(frame_1, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    metrics.update(start_time, frame_1)
    cv2.imshow("Lead Integrated Business Services Pvt Ltd", frame_1)

    # Process frames from camera 2
    
    if frame_2 is not None:
        detections_2 = frame_processor_2.face_process(frame_2)
        for roi, landmarks, identity in zip(*detections_2):
        # Get coordinates of the bounding box
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), frame_2.shape[1])
            ymax = min(int(roi.position[1] + roi.size[1]), frame_2.shape[0])
            if ymax >= line_position:
                # Draw bounding box for faces below the line
                cv2.rectangle(frame_2, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
                # Send for recognition if below the line
                frame_2 = draw_face_detection(frame_2, frame_processor_2, [(roi, landmarks, identity)], camera_id="B", fqa_threshold=fqa_threshold)
  # Adjust threshold as needed
            else:
                # Draw bounding box for faces above the line
                cv2.rectangle(frame_2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    metrics.update(start_time, frame_2)
    cv2.imshow("Lead Integrated Business Services Pvt Ltd.", frame_2)

    # Process keyboard inputs
    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key == ord('w'):  # Move the line up when 'w' is pressed
        line_position -= 10  # Adjust the value according to your preference
    elif key == ord('s'):  # Move the line down when 's' is pressed
        line_position += 10  # Adjust the value according to your preference
    elif key == ord('a'):  # Decrease FQA threshold when 'a' is pressed
        fqa_threshold -= 0.05  # Adjust the decrement value as needed
        print("FQA threshold decreased to:", fqa_threshold)
    elif key == ord('d'):  # Increase FQA threshold when 'd' is pressed
        fqa_threshold += 0.05  # Adjust the increment value as needed

# Release camera resources and close windows
source_1.release()
source_2.release()
cv2.destroyAllWindows()
