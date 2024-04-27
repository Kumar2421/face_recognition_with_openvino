
import logging as log
from time import perf_counter
import cv2
from openvino.runtime import Core, get_version
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from model_api.performance_metrics import PerformanceMetrics

source = 0
device = "CPU"
faceDETECT = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\face-detection-retail-0005.xml"
faceLANDMARK = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\landmarks-regression-retail-0009.xml"
faceIDENTIFY = "C:\\xampp1\\htdocs\demo\\face_recogition_intel\\model_2022_3\\face-reidentification-retail-0095.xml"


class FrameProcessor:
    QUEUE_SIZE = 16
   
    def __init__(self):
        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()
        self.face_detector = FaceDetector(core, faceDETECT, input_size=(0, 0), confidence_threshold=0.6)
        self.landmarks_detector = LandmarksDetector(core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(core, faceIDENTIFY, match_threshold=0.7, match_algo='HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase("C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face_img1", self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))
        self.detections = []
        self.face_tracks = {}  # Dictionary to store face tracks
        self.next_track_id = 0  # Counter for assigning new track IDs

    def update_face_tracks(self, detections):
        # Update face tracks based on new detections
        for detection in detections:
            roi, landmarks, identity = detection
            centroid = (roi.position[0] + roi.size[0] / 2, roi.position[1] + roi.size[1] / 2)
            matched_track_id = None
            min_distance = float('inf')
            for track_id, track_centroid in self.face_tracks.items():
                distance = ((centroid[0] - track_centroid[0]) ** 2 + (centroid[1] - track_centroid[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    matched_track_id = track_id
            if matched_track_id is not None and min_distance < 80:  # Threshold for centroid proximity
                self.face_tracks[matched_track_id] = centroid
                identity.id = matched_track_id
                print(matched_track_id,"matched id")
            else:
                self.next_track_id += 1
                self.face_tracks[self.next_track_id] = centroid
                identity.id = self.next_track_id
                print(self.next_track_id,"next id")


      # Initialize detections attribute

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE > len(rois):
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, _ = self.face_identifier.infer((frame, rois, landmarks))
        self.detections = list(zip(rois, landmarks, face_identities))  # Store detections
        self.update_face_tracks(self.detections)  # Update face tracks
        return self.detections

# Modify the draw_face_detection function to handle multiple detections
def draw_face_detection(frame_processor):
    size = frame_processor.frame.shape[:2]
    for detection in frame_processor.detections:
        roi, landmarks, identity = detection
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        cv2.rectangle(frame_processor.frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        face_point = (xmin, ymin)
        for point in landmarks:
            x = int(xmin + roi.size[0] * point[0])
            y = int(ymin + roi.size[1] * point[1])
            cv2.circle(frame_processor.frame, (x, y), 1, (0, 255, 255), 1)
        image_recognizer(frame_processor.frame, text, identity, face_point, 0.75)
    return frame_processor.frame

def image_recognizer(frame, text, identity, face_point, threshold):
    xmin, ymin = face_point
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1 - identity.distance) > threshold:
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - textsize[1]), (xmin + textsize[0], ymin), (0, 0, 0), cv2.FILLED)  # Draw a filled rectangle as highlighter
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            textsize = cv2.getTextSize("UNKNOWN", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - textsize[1]), (xmin + textsize[0], ymin), (0, 0, 0), cv2.FILLED)  # Draw a filled rectangle as highlighter
            cv2.putText(frame, "UNKNOWN", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cap = cv2.VideoCapture(0)
frame_processor = FrameProcessor()
metrics = PerformanceMetrics()
while True:
    start_time = perf_counter()
    ret, frame_processor.frame = cap.read()
    if not ret:
        break
    detections = frame_processor.face_process(frame_processor.frame)
    frame_processor.frame = draw_face_detection(frame_processor)
    metrics.update(start_time, frame_processor.frame)
    cv2.imshow("face recognition demo", frame_processor.frame)
    key = cv2.waitKey(1)
    if key in (ord('q'), ord('Q'), 27):
            break

