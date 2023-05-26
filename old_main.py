import cv2
import time
import os
import HandTrackingModule as htm
import json
import requests

url = 'https://api.github.com/some/endpoint'


# def initialize_camera(height, width):
#     cap = cv2.VideoCapture(1)
#     cap.set(3, height)
#     cap.set(4, width)


folderPath = "fingers"  # name of the folder, where there are images of fingers
searchPath = "search"  # name of the folder, where there are images of fingers
fingerList = os.listdir(folderPath)  # list of image titles in 'fingers' folder
overlayList = []

for imgPath in fingerList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75, maxHands=1)
totalFingers = 0

img = cv2.imread(f'{searchPath}/5.jpg')
img = cv2.flip(img, 1)
img = detector.findHands(img)
lm_list, bbox = detector.findPosition(img, draw=False)

if lm_list:
    fingersUp = detector.fingersUp()
    totalFingers = fingersUp.count(1)

    h, w, c = overlayList[totalFingers].shape
    img[0:h, 0:w] = overlayList[totalFingers]

cTime = time.time()
fps = 1 / (cTime - pTime)
pTime = cTime

cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

cv2.imshow("Image", img)
cv2.waitKey(1)


def res():
    payload = {'answer': totalFingers}
    r = requests.post(url, data=json.dumps(payload))
    print(r)
    print(totalFingers)
    return totalFingers


res()
