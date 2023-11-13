# Real-time-Drowsiness-Detection-System-using-Facial-Landmarks

This project aims to create a Drowsiness Detection System using Convolutional Neural Networks (CNNs) to monitor and alert individuals about their drowsy state during activities such as driving. The system utilizes facial feature analysis, particularly focusing on the eyes, to detect signs of drowsiness.

## Features

- Real-time monitoring of facial features.
- Eye state classification as "Open" or "Closed" using a trained CNN model.
- Continuous tracking of the user's drowsiness score.
- Alarm activation when the drowsiness score surpasses a predefined threshold.
- Simple and intuitive user interface.

## Requirements

- Python
- OpenCV
- Keras
- TensorFlow
- Matplotlib
- Pygame
## Architecture Diagram/Flow

![image](https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks/assets/93427303/7b211d1a-ebb1-4422-8086-6e1cb291356e)


## Installation

1. Clone the repository: git clone https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks.git

2. Install dependencies: pip install -r requirements.txt

3. Download the pre-trained model file (cnnCat2.h5) or train a new model using the provided dataset.

4. Ensure the required Haar cascade files (haarcascade_frontalface_alt.xml, haarcascade_lefteye_2splits.xml, haarcascade_righteye_2splits.xml) are available in the specified paths.

5. Run the Drowsiness Detection.py script: python "Drowsiness Detection.py"

## Usage

1. Execute the script, and the webcam feed will open.

2. The system will continuously monitor eye states and update the drowsiness score.

3. A visual and audible alarm will be triggered if the drowsiness score exceeds the threshold.

4. Close the application by pressing 'q' on the keyboard.

## Program:

```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


lang = st.text_input("Language")
singer = st.text_input("singer")
music_player = st.selectbox("Music Player", ["YouTube", "Spotify"])


if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        if music_player == "YouTube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        elif music_player == "Spotify":
            webbrowser.open(f"https://open.spotify.com/search/{lang} {emotion} song {singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
```
## Output:

![output1](output1.png)
![output2](output2.png)

## Result:

The system provides real-time monitoring of drowsiness levels, alerting users when they show signs of fatigue, especially in situations like driving where staying alert is crucial for safety. The visual and audible alarms aim to prompt users to take breaks or corrective actions to prevent potential accidents due to drowsiness.
