import cv2
import os
import time

student_id = input("Enter Student ID: ")

folder = os.path.join("dataset", student_id)
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open camera")
    exit()

print("Look at the camera...")
print("Capturing 5 images automatically...")

time.sleep(3)  # Gives you 3 seconds to get ready

for i in range(1, 6):

    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    filename = os.path.join(folder, f"{i}.jpg")
    cv2.imwrite(filename, frame)

    print(f"Image {i} saved")

    cv2.imshow("Capture Face", frame)
    cv2.waitKey(500)   # Show image for 0.5 seconds

    time.sleep(1)      # Wait 1 second before next photo

camera.release()
cv2.destroyAllWindows()

print("Face capture completed!")