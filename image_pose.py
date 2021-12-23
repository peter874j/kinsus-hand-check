import cv2
import mediapipe as mp
import time
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


imagePath = r'.\images'

# 列出指定路徑底下所有檔案(包含資料夾)

allFileList = os.listdir(imagePath)

for i in allFileList:
    print(i)
    file_name = os.path.join(imagePath, i)

    img = cv2.imread(file_name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                hand_id = [4, 8, 12, 16, 20]
                if id in hand_id:
                    cv2.circle(img, (cx, cy), 10
                           , (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        #             (255, 0, 255), 3)
    output_name = 'image_output/' + i
    cv2.imwrite(output_name, img)

print('done...')
# cv2.imshow("TEST Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



