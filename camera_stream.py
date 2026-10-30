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

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

THRESHOLD = 0.95

marked_students = set()
# -----------------------------
# Function to Mark Attendance
# -----------------------------
def mark_attendance(student_id):

    try:

        conn = get_connection()
        cursor = conn.cursor()

        today = datetime.now().date()
        current_time = datetime.now().time()

        cursor.execute(
            """
            SELECT *
            FROM attendance
            WHERE student_id=%s
            AND attendance_date=%s
            """,
            (student_id, today)
        )

        result = cursor.fetchone()

        if result is None:

            cursor.execute(
                """
                INSERT INTO attendance
                (
                    student_id,
                    attendance_date,
                    attendance_time,
                    status
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    student_id,
                    today,
                    current_time,
                    "Present"
                )
            )

            conn.commit()

            print(f"Attendance marked for {student_id}")

        cursor.close()
        conn.close()

    except Exception as e:

        print("Database Error:", e)
        