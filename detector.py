import sys
import logging as log
from time import perf_counter
import cv2
import numpy as np
from openvino.runtime import Core, get_version
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from model_api.performance_metrics import PerformanceMetrics
from collections import Counter
from deepface import DeepFace

source = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\pexels_videos_1721303 (1080p).mp4"
device = "CPU"
faceDETECT = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\model_2022_3\\face-detection-retail-0005.xml"
faceLANDMARK = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\model_2022_3\\landmarks-regression-retail-0009.xml"
faceIDENTIFY = "C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\model_2022_3\\face-reidentification-retail-0095.xml"
class face_quality_assessment():
    def __init__(self, model_path, device='GPU'):
        self.model = cv2.dnn.readNet(model_path)
        self.input_width = 128
        self.input_height = 128
        self.device = device
        if self.device != 'GPU':
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(input_img)
        self.model.setInput(blob)
        outputs = self.model.forward()
        return outputs[0].reshape(-1)
class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self,fqa_model_path):
        self.core = Core()
        self.face_detector = FaceDetector(self.core, faceDETECT, input_size=(0, 0), confidence_threshold=0.6)
        self.landmarks_detector = LandmarksDetector(self.core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(self.core, faceIDENTIFY, match_threshold=0.8, match_algo='HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('C:\\xampp1\\htdocs\\demo\\captured_faces', self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)
        self.fqa_model = face_quality_assessment(fqa_model_path)

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if len(rois) > self.QUEUE_SIZE:
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return rois, landmarks, face_identities

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
        image_recognizer(frame, text, identity, face_point, 0.75)
    return frame

def image_recognizer(frame, text, identity, face_point, threshold):
    xmin, ymin = face_point
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1 - identity.distance) > threshold:
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            # Ensure text position is within frame boundaries
            text_x = max(xmin, 0)
            text_y = max(ymin - 10, 0)  # Adjusted to place text above the face rectangle
            cv2.rectangle(frame, (text_x, text_y), (text_x + textsize[0], text_y + textsize[1]), (0 ,0, 0), cv2.FILLED)
            cv2.putText(frame, text, (text_x, text_y + textsize[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            textsize = cv2.getTextSize("Unknown", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            # Ensure text position is within frame boundaries
            text_x = max(xmin, 0)
            text_y = max(ymin - 10, 0)  # Adjusted to place text above the face rectangle
            cv2.rectangle(frame, (text_x, text_y), (text_x + textsize[0], text_y + textsize[1]), (255, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Unknown", (text_x, text_y + textsize[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

def apply_ensemble_methods(face_identities_ensemble, unknowns_ensemble):
    # Example of simple voting
    num_frames = len(face_identities_ensemble)  # Use len() to get the length of the list
    final_identities = []

    for i in range(num_frames):
        identities = face_identities_ensemble[i]
        unknowns = unknowns_ensemble[i]

        if len(identities) > 0:
            final_identities.extend(identities)
        else:
            final_identities.extend([None] * len(unknowns))

    # Check if the final_identities list is empty
    if final_identities:
        # Apply simple voting only if the list is not empty
        final_result = Counter(final_identities).most_common(1)[0][0]
    else:
        # Handle the case where the list is empty
        final_result = None

    return final_result

cap = cv2.VideoCapture(0)  # Use video source

frame_processor = FrameProcessor("C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face-quality-assessment.onnx")
metrics = PerformanceMetrics()

# Initialize lists for ensemble methods and voting
face_identities_ensemble = []
unknowns_ensemble = []

while True:
    start_time = perf_counter()
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break
    
    # Check if the frame is None
    if frame is None:
        print("Error: No frame captured from the video source.")
        break

    detections = frame_processor.face_process(frame)
    frame = draw_face_detection(frame, frame_processor, detections)
    
    # Aggregate face identities and unknowns for ensemble methods
    face_identities, unknowns = detections[2], detections[3]
    face_identities_ensemble.extend(face_identities)
    unknowns_ensemble.extend(unknowns)

    metrics.update(start_time, frame)
    cv2.imshow("face recognition demo", frame)
    key = cv2.waitKey(1)
    if key in [ord('q'), ord('Q'), 27]:
        cap.release()
        cv2.destroyAllWindows()
        break

# Convert lists to numpy arrays
face_identities_ensemble = np.array(face_identities_ensemble)
unknowns_ensemble = np.array(unknowns_ensemble)

# Apply ensemble methods and voting for the final result
final_result = apply_ensemble_methods(face_identities_ensemble, unknowns_ensemble)
print("Final Result:", final_result)
