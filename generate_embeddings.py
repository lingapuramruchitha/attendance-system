import os
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

dataset_path = "dataset"
embeddings = {}

for student in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student)

    if not os.path.isdir(student_folder):
        continue

    student_embeddings = []

    for image_name in os.listdir(student_folder):

        if not image_name.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(student_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(image)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]["box"]

        x = max(0, x)
        y = max(0, y)

        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (160, 160))

        embedding = embedder.embeddings([face])[0]

        student_embeddings.append(embedding)

    if len(student_embeddings) > 0:
        embeddings[student] = np.mean(student_embeddings, axis=0)
        print(f"{student} embeddings generated using {len(student_embeddings)} images")

os.makedirs("embeddings", exist_ok=True)

with open("embeddings/embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)

print("Embeddings saved successfully!")