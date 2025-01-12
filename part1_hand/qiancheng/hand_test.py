import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    print("请检查：")
    print("1. 是否已连接摄像头")
    print("2. 摄像头是否被其他程序占用")
    print("3. 是否有摄像头访问权限")
    exit()

# 打印摄像头信息
print(f"摄像头分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"摄像头帧率: {int(cap.get(cv2.CAP_PROP_FPS))}")

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  # 用于存储食指顶端手指节的历史位置
  index_finger_tips = []
  recording = False
  eraser_mode = False
  eraser_radius = 20  # 橡皮擦范围半径

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

        # 获取所有手指的关键点
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        h, w, _ = image.shape
        # 转换所有手指尖点的坐标
        all_finger_tips = [
            (int(thumb_tip.x * w), int(thumb_tip.y * h)),
            (int(index_finger_tip.x * w), int(index_finger_tip.y * h)),
            (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)),
            (int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)),
            (int(pinky_tip.x * w), int(pinky_tip.y * h))
        ]

        # 检查是否所有手指都伸展（橡皮擦模式）
        if (all(tip.y < index_finger_pip.y for tip in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip])):
            eraser_mode = True
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            # 绘制橡皮擦范围指示器
            cv2.circle(image, (cx, cy), eraser_radius, (255, 0, 0), 2)
            
            # 擦除范围内的轨迹点
            i = 0
            while i < len(index_finger_tips):
                if index_finger_tips[i] is not None:
                    if calculate_distance((cx, cy), index_finger_tips[i]) < eraser_radius:
                        index_finger_tips[i] = None
                i += 1
            recording = False
        
        # 检查是否只伸出食指（绘画模式）
        elif (index_finger_tip.y < index_finger_pip.y and
            middle_finger_tip.y > index_finger_pip.y and
            ring_finger_tip.y > index_finger_pip.y and
            pinky_tip.y > index_finger_pip.y):
            eraser_mode = False
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            if not recording:
                index_finger_tips.append(None)  # 插入一个None以表示新的开始
                recording = True
            index_finger_tips.append((cx, cy))
            # 在图像上绘制食指顶端手指节的当前位置
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        else:
            recording = False
            eraser_mode = False

    # 在图像上绘制食指顶端手指节的历史位置，并按时间顺序连线
    for i in range(1, len(index_finger_tips)):
        if index_finger_tips[i] is not None and index_finger_tips[i-1] is not None:
            cv2.line(image, index_finger_tips[i-1], index_finger_tips[i], (0, 0, 255), 2)

    # 显示当前模式
    mode_text = "橡皮擦模式" if eraser_mode else "绘画模式"
    cv2.putText(image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()