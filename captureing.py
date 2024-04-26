import cv2
import os
import numpy as np

def enhance_image(image):
    # Convert the image to RGB (if it's grayscale)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply histogram equalization to enhance contrast
    enhanced = cv2.equalizeHist(image[:,:,0])
    return cv2.merge([enhanced, enhanced, enhanced])

def is_image_quality_good(image):
    if image is None or image.size == 0:
        return False
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of Laplacian to measure image quality
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Check if variance of Laplacian is above a threshold
    return laplacian_var > 150

def capture_face(vs, name):
    # Load the deep learning based face detector
    prototxt_path = r'C:\xampp1\htdocs\demo\face_recogition_intel\model_2022_3\deploy.prototxt'
    model_path = r'C:\xampp1\htdocs\demo\face_recogition_intel\model_2022_3\res10_300x300_ssd_iter_140000_fp16.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Directory to save captured face images
    output_dir = 'captured_faces'
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter to track the number of captured images
    image_count = 0
    
    # Margin around the detected faces
    margin = 30

    while True:
        # Read frame from the camera
        ret, frame = vs.read()

        if not ret:
            print("Error: Unable to read frame from camera")
            break

        # Enhance the captured image
        enhanced_frame = enhance_image(frame)

        # Convert the frame to a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(enhanced_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network to detect faces
        net.setInput(blob)
        detections = net.forward()

        # Process each detected face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the confidence level is above a threshold
            if confidence > 0.7:
                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Add margin to the bounding box
                startX = max(0, startX - margin)
                startY = max(0, startY - margin)
                endX = min(frame.shape[1], endX + margin)
                endY = min(frame.shape[0], endY + margin)
                
                # Crop the detected face region
                face_roi = frame[startY:endY, startX:endX]
                
                # Check if the image quality is good
                if is_image_quality_good(face_roi):
                    # Save the cropped face image with the person's name
                    image_path = os.path.join(output_dir, f'{name}_{image_count}.jpg')
                    cv2.imwrite(image_path, face_roi)
                    image_count += 1
                    print(f"Image {image_count} saved as {image_path}")
                else:
                    print("Warning: Low quality image. Skipping.")
                
                # Draw the bounding box around the face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display the original frame with face detection
        cv2.imshow('Original Frame', frame)

        # Exit loop if 'q' is pressed or capture 50 images
        if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= 50:
            break

    # Release the camera and close the window
    vs.release()
    cv2.destroyAllWindows()

def connect_to_camera(ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"
    vs = cv2.VideoCapture(url)
    return vs

if __name__ == "__main__":
    name = input("Enter your name: ")
    source_1 = connect_to_camera('192.168.29.208', 'admin', 'Libs2000@')
    capture_face(source_1, name)
