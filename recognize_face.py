import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from database import get_connection
from datetime import datetime

print("Loading embeddings...")

with open("embeddings/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print("Loading FaceNet...")

detector = MTCNN()
embedder = FaceNet()


def mark_attendance(student_id):

    student_id = int(student_id)

    try:
        print("Recognized student_id:", student_id)

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DATABASE()")
        print("Current Database:", cursor.fetchone())

        cursor.execute("SELECT COUNT(*) FROM attendance")
        print("Attendance Count:", cursor.fetchone())

        today = datetime.now().date()
        current_time = datetime.now().time()

        print("Checking attendance for:", student_id)

        cursor.execute(
            """
            SELECT * FROM attendance
            WHERE student_id=%s AND attendance_date=%s
            """,
            (student_id, today)
        )

        result = cursor.fetchone()

        if result is None:

            print("No previous attendance found.")

            cursor.execute(
                """
                INSERT INTO attendance
                (student_id, attendance_date, attendance_time, status)
                VALUES (%s,%s,%s,%s)
                """,
                (
                    student_id,
                    today,
                    current_time,
                    "Present"
                )
            )

            conn.commit()

            print("Commit Successful")
            print("Rows Inserted:", cursor.rowcount)

            cursor.execute("SELECT * FROM attendance")
            print(cursor.fetchall())

            print("✅ Attendance marked successfully for", student_id)

        else:
            print("Attendance already exists for", student_id)

        cursor.close()
        conn.close()

    except Exception as e:
        print("❌ DATABASE ERROR:", e)
        
print("Opening Camera...")

# Try the default backend first
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Trying DirectShow...")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Trying Media Foundation...")
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not camera.isOpened():
    print("Cannot open camera")
    exit()

THRESHOLD = 0.75

marked_students = set()

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    for face in faces:

        x, y, w, h = face["box"]

        x = max(0, x)
        y = max(0, y)

        crop = rgb[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (160, 160))

        embedding = embedder.embeddings([crop])[0]

        best_name = "Unknown"
        best_distance = float("inf")

        for student, saved_embedding in embeddings.items():

            distance = np.linalg.norm(saved_embedding - embedding)

            print(f"Student {student} -> Distance = {distance:.4f}")

            if distance < best_distance:
                best_distance = distance
                best_name = student

        print("----------------------")
        print("Best Match:", best_name)
        print("Best Distance:", best_distance)
        print("----------------------")

        if best_distance > THRESHOLD:
            best_name = "Unknown"

        if best_name != "Unknown":
            print("Recognized:", best_name)

            if best_name not in marked_students:
                mark_attendance(best_name)
                marked_students.add(best_name)
                print("Attendance Marked:", best_name)

        if best_name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            best_name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    # Close on Q key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Closing...")
        break

    # Close if X button is clicked
    if cv2.getWindowProperty("Face Recognition Attendance", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed.")
        break

camera.release()
cv2.destroyAllWindows()