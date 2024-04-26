import os

import cv2
import cv2

def connect_to_camera(ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"
    vs = cv2.VideoCapture(url)
    return vs

def capture_images(source, output_dir, interval=100, num_images=10):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    while count < num_images:
        ret, frame = source.read()
        if not ret:
            print("Failed to retrieve frame from the camera.")
            break
        
        # Save the frame as an image
        image_path = os.path.join(output_dir, f"image_{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")

        # Display the frame
        cv2.imshow('IP Camera', frame)
        
        count += 1
        if cv2.waitKey(interval) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    source.release()
    cv2.destroyAllWindows()

# Connect to the camera
source_2 = connect_to_camera('192.168.29.208', 'admin', 'Libs2000@')

# Check if the camera connection is successful
if source_2 is not None:
    print("Connected to the camera successfully.")
    
    # Capture and save images from the camera stream
    capture_images(source_2, 'captured_images', interval=1000, num_images=10)
    
else:
    print("Failed to connect to the camera.")
