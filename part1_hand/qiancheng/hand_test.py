import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  # 用于存储食指顶端手指节的历史位置
  index_finger_tips = []
  recording = False

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

        # 获取手指节的坐标
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # 判断是否只伸出了食指
        if (index_finger_tip.y < index_finger_pip.y and
            middle_finger_tip.y > index_finger_pip.y and
            ring_finger_tip.y > index_finger_pip.y and
            pinky_tip.y > index_finger_pip.y):
          h, w, _ = image.shape
          cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
          if not recording:
            index_finger_tips.append(None)  # 插入一个None以表示新的开始
            recording = True
          index_finger_tips.append((cx, cy))

          # 在图像上绘制食指顶端手指节的当前位置
          cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        else:
          recording = False

    # 在图像上绘制食指顶端手指节的历史位置，并按时间顺序连线
    for i in range(1, len(index_finger_tips)):
      if index_finger_tips[i] is not None and index_finger_tips[i-1] is not None:
        cv2.line(image, index_finger_tips[i-1], index_finger_tips[i], (0, 0, 255), 2)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()