import datetime
import sys
import os
import time
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
        self.face_detector = FaceDetector(self.core, faceDETECT, input_size=(0, 0), confidence_threshold=0.8)
        self.landmarks_detector = LandmarksDetector(self.core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(self.core, faceIDENTIFY, match_threshold=0.7, match_algo='HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\senthil', self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if len(rois) > self.QUEUE_SIZE:
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return rois, landmarks, face_identities

def draw_face_detection(frame, frame_processor, detections, camera_id):
    size = frame.shape[:2]
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        # face_point = (xmin, ymin)
        # for point in landmarks:
        #     x = int(xmin + roi.size[0] * point[0])
        #     y = int(ymin + roi.size[1] * point[1])
        #     cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)
        image_recognizer(frame, text, identity, (xmin, ymin, xmax, ymax), 0.75, camera_id)  # Pass camera_id to image_recognizer
    return frame


# Function to capture and save the image of the identified person
# Function to capture and save the image of the identified person
# def capture_and_save_image(frame, text, face_rect):
#     global unique_id_counter
#     global image_path  # Access the global image_path variable
#     if not os.path.exists(captured_images_dir):
#         try:
#             os.makedirs(captured_images_dir)
#         except OSError as e:
#             print(f"Failed to create directory: {e}")
#             return
#     face_x, face_y, face_w, face_h = face_rect
#     face_image = frame[face_y:face_y+face_h, face_x:face_x+face_w]
#     image_name = f"{text}-{unique_id_counter}.jpg"
#     image_path = os.path.join(captured_images_dir, image_name)
#     cv2.imwrite(image_path, face_image)
#     unique_id_counter += 1
#     print(f"Image of {text} saved as {image_path}")
#     return image_path  # Return the updated image path


unknown_counter = 0
# def capture_and_save_image(frame, text, face_rect):
#     global unique_id_counter
#     global captured_images_dir  # Access the global captured_images_dir variable
#     global unknown_counter
    
#     if not os.path.exists(captured_images_dir):
#         try:
#             os.makedirs(captured_images_dir)
#         except OSError as e:
#             print(f"Failed to create directory: {e}")
#             return
    
#     face_x, face_y, face_w, face_h = face_rect
#     face_image = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    
#     # Save the entire face area
#     image_name = f"{text}-{unique_id_counter}.jpg"
#     image_path = os.path.join(captured_images_dir, image_name)
#     cv2.imwrite(image_path, face_image)
    
#     # Update counters
#     unique_id_counter += 1
#     if text.startswith("Unknown"):
#         unknown_counter += 1
    
#     print(f"Image of {text} saved as {image_path}")
#     return image_path  # Return the updated image path
#   # Return the updated image path
#  # Return the updated image path
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
    global unknown_counter
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1 - identity.distance) > threshold:
            current_time = datetime.now().replace(microsecond=0)  # Remove milliseconds
            entry_date = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).date()
            entry_time = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).time()
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

            # Draw text without a rectangle around it
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0)
            image_path = capture_and_save_image(frame, text, (xmin, ymin, xmax - xmin, ymax - ymin))  # Update image_path
            try:
                if camera_id == 'A':
                    # Insert entry record into the database for camera A (entry)
                    query = "INSERT INTO face_recognition (empl_id, camera, entry_date, entry_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                    image_path_with_backslash = image_path.replace('/', '\\')  # Replace forward slash with backslash
                    db_cursor.execute(query, (text, 'Camera A', entry_date, entry_time.strftime("%I:%M %p"), image_path_with_backslash))
                    print("Person identified and entry record saved for Camera A.")
                elif camera_id == 'B':
                    # Insert entry record into the database for camera B (entry)
                    query = "INSERT INTO face_recognition (empl_id, camera, exit_date, exit_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                    image_path_with_backslash = image_path.replace('/', '\\')  # Replace forward slash with backslash
                    db_cursor.execute(query, (text, 'Camera B', entry_date, entry_time.strftime("%I:%M %p"), image_path_with_backslash))
                    print("Person identified and entry record saved for Camera B.")
                    # Commit the transaction
                    db_connection.commit()

                # Add a 1-second delay
                
            except Exception as e:
                # Rollback in case of an error
                db_connection.rollback()
                print("Error occurred:", e)

        else:
            textsize = cv2.getTextSize("Unknown", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            # Draw text without a rectangle around it
            cv2.putText(frame, "Unknown", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Capture and save image of unknown person
            image_path = capture_and_save_image(frame, f"Unknown_{unknown_counter}", (xmin, ymin, xmax - xmin, ymax - ymin))
            print(f"Unknown person detected. Image saved as {image_path}")

            try:
                # Insert entry record into the database for unknown person
                current_time = datetime.now().replace(microsecond=0)  # Remove milliseconds
                entry_date = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).date()
                entry_time = current_time.astimezone(timezone(timedelta(hours=5, minutes=30))).time()
                query = "INSERT INTO face_recognition (empl_id, camera, entry_date, entry_time, image_path) VALUES (%s, %s, %s, %s, %s)"
                db_cursor.execute(query, (f"Unknown_{unknown_counter}", camera_id, entry_date, entry_time.strftime("%I:%M %p"), image_path))
                print("Entry record saved for unknown person.")

                # Commit the transaction
                db_connection.commit()

                # Increment unknown counter
                unknown_counter += 1

                # Add a 1-second delay
               

            except Exception as e:
                # Rollback in case of an error
                db_connection.rollback()
                print("Error occurred:", e)
    else:
        print("Unknown face detected.")




# Function to recognize the image and store in the database
# Initialize frame processors
frame_processor_1 = FrameProcessor()
frame_processor_2 = FrameProcessor()

# Main loop for video processing
# Main loop for video processing
while True:
    start_time = perf_counter()
    ret_1, frame_1 = source_1.read()
    ret_2, frame_2 = source_2.read()
    
    if not ret_1 or not ret_2:
        break

    detections_1 = frame_processor_1.face_process(frame_1)
    frame_1 = draw_face_detection(frame_1, frame_processor_1, detections_1, 'A')  # Pass 'A' for camera 1
    
    detections_2 = frame_processor_2.face_process(frame_2)
    frame_2 = draw_face_detection(frame_2, frame_processor_2, detections_2, 'B')  # Pass 'B' for camera 2
    
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
