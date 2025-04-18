import cvzone
import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from google import genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
st.image('MathGestures.jpg')

col1, col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run',value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer:")
    output_text_area = st.subheader("")


client = genai.Client(api_key="AIzaSyAlJbnXbnibwW5YBdUb9Z3QjOep3cm9Oq0")


# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(propId=3,value=1000)
cap.set(propId=4,value=600)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=False)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        print(fingers)
        return fingers,lmList
    else:
        return None
    
def draw(info,prev_pos,canvas):
    fingers,lmlist = info
    current_pos = None
    #it's actually if fingers == [1,0,0,0,0] but it my system below statement works properly and if the code is working then dont change it.
    if fingers == [1,1,0,0,0]:
        current_pos = tuple(lmlist[8][0:2]) #[8] = hand landmark and we need 0,1 = x & y axis
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas,current_pos,prev_pos,color=(255,255,255),thickness=10)

    elif fingers == [0,1,1,1,1]: #Erase whole Canvas here also, elif fingers == [1,1,1,1,1] is correct
        canvas = np.zeros_like(img)

    elif fingers == [1, 0, 0, 0, 0]:  # Eraser mode (Fist), elif fingers == [0, 0, 0, 0, 0] is correct
        current_pos = tuple(lmlist[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, color=(0, 0, 0), thickness=30)  # Larger thickness for erasing


    return current_pos,canvas

def sendToAI(client,canvas,fingers):
    if fingers == [0,1,1,1,0]:
        pil_Image = Image.fromarray(canvas)
        response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["Show me the answer of Math Problem",pil_Image]
            )
        return response.text

prev_pos = None
canvas = None
image_merge = None
output_text = ""

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)
        image_merge = img.copy()


    info = getHandInfo(img)
    if info:
        fingers,lmlist = info
        prev_pos,canvas = draw(info,prev_pos,canvas)
        output_text = sendToAI(client,canvas,fingers)

    image_merge = cv2.addWeighted(img,0.7,canvas,0.3,0)
    FRAME_WINDOW.image(image_merge,channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    
    # Display the image in a window before deploying your code on Streamlit
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("Combined Image", image_merge)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)