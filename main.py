import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# ---------------------- GOOGLE SHEETS SETUP ----------------------
SERVICE_ACCOUNT_FILE = r"C:\Users\Rishabh\Downloads\aiattendencesystem-d13ae1e5a8e4.json"  # Replace with your JSON key path
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open("Attendence Sheet").sheet1  # Replace with your Google Sheet name

# ---------------------- CREATE/LOAD DATASET ----------------------
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Capture new student images
student_name = input("Enter Student Name: ")
roll_no = input("Enter Roll Number: ")
student_folder = os.path.join(dataset_dir, f"{roll_no}_{student_name}")
if not os.path.exists(student_folder):
    os.makedirs(student_folder)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

print("Capturing 30 images. Look at the camera...")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{student_folder}/{roll_no}_{student_name}_{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset created successfully!")

# ---------------------- LOAD DATASET AND COMPUTE ORB FEATURES ----------------------
student_images = {}
orb = cv2.ORB_create()

for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if "_" not in folder_name:
        print(f"Skipping invalid folder: {folder_name}")
        continue
    roll_no_folder, student_name_folder = folder_name.split("_", 1)
    images = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        images.append((img, des))
    student_images[f"{roll_no_folder}_{student_name_folder}"] = images

# ---------------------- INITIALIZE ATTENDANCE ----------------------
attendance = pd.DataFrame(columns=["RollNo", "StudentName", "Time"])

# Clear Google Sheet before starting
sheet.clear()
sheet.append_row(["RollNo", "StudentName", "Time"])

# ---------------------- START REAL-TIME ATTENDANCE ----------------------
cap = cv2.VideoCapture(0)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("Starting attendance system. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        kp2, des2 = orb.detectAndCompute(face_roi, None)
        best_match = "Unknown"
        max_matches = 0

        for key, imgs in student_images.items():
            for img, des1 in imgs:
                if des1 is not None and des2 is not None:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = [m for m in matches if m.distance < 70]
                    if len(good_matches) > max_matches:
                        max_matches = len(good_matches)
                        best_match = key

        if best_match != "Unknown":
            roll_no_detected, student_name_detected = best_match.split("_", 1)
            cv2.putText(frame, f"{roll_no_detected} {student_name_detected}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # MARK ATTENDANCE LOCALLY
            now = datetime.now().strftime("%H:%M:%S")
            if not ((attendance["RollNo"] == roll_no_detected) & (attendance["StudentName"] == student_name_detected)).any():
                attendance = pd.concat([attendance, pd.DataFrame([[roll_no_detected, student_name_detected, now]],
                                                                 columns=["RollNo", "StudentName", "Time"])], ignore_index=True)
                # MARK ATTENDANCE IN GOOGLE SHEET
                sheet.append_row([roll_no_detected, student_name_detected, now])
                print(f"Attendance marked for {roll_no_detected} {student_name_detected}")
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow("Smart Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance session ended.")








