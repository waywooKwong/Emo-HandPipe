import cv2
import mediapipe as mp
import numpy as np
import math
import os
from datetime import datetime
from collections import deque
import pytesseract
from PIL import Image
"""
目前基本实现手写字符识别：简单字识别准确率高，复杂字识别准确率低
主要改动如下：
1. 在钱哥的基础上，稍微降低了灵敏度
2. 结合手写字符识别模型（目前只有中文）
3. 写完字以后按下空格清屏然后截屏识别（函数已经写好，后序只需将函数的触发改为手势）
4. 目前截屏只截字，没有将人脸截进去，后续完善
5. 模型安装教程在.md文件中
"""
# 初始化MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(image):
    """简单的图像预处理"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh

def recognize_text(image):
    """识别单个字符"""
    # 预处理图像
    processed_image = preprocess_for_ocr(image)
    
    # 保存预处理后的图像
    cv2.imwrite('preprocessed_image.png', processed_image)
    
    # 使用PIL打开图像
    pil_image = Image.fromarray(processed_image)
    
    try:
        # 使用基本的OCR配置
        text = pytesseract.image_to_string(
            pil_image,
            lang='chi_sim',
            config='--oem 3 --psm 10'
        )
        text = text.strip()
        if text:
            first_char = text[0] if len(text) > 0 else ''
            if first_char:
                print("识别结果:", first_char)
                return first_char
            else:
                print("无法识别字符")
                return None
        else:
            print("无法识别字符")
            return None
    except Exception as e:
        print(f"OCR识别错误: {str(e)}")
        return None

def save_screenshot(image):
    """保存截图并识别字符"""
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'screenshots/handwriting_{timestamp}.png'
    
    # 保存原始图像
    cv2.imwrite(filename, image)
    print(f'Screenshot saved as {filename}')
    
    # 对保存的图像进行文字识别
    recognized_char = recognize_text(image)
    
    return filename, recognized_char

def calculate_distance(p1, p2):
    """计算两点之间的距离"""
    if p1 is None or p2 is None:
        return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def smooth_point(point, points_buffer):
    """平滑处理单个点"""
    if len(points_buffer) == 0:
        return point
    
    x = sum(p[0] for p in points_buffer) + point[0]
    y = sum(p[1] for p in points_buffer) + point[1]
    n = len(points_buffer) + 1
    return (x // n, y // n)

def interpolate_points(p1, p2, num_points=5):
    """在两点之间插值生成平滑的过渡点"""
    if p1 is None or p2 is None:
        return []
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(p1[0] * (1 - t) + p2[0] * t)
        y = int(p1[1] * (1 - t) + p2[1] * t)
        points.append((x, y))
    return points

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
  # 初始化变量
  index_finger_tips = []
  recording = False
  frame_count = 0
  last_point = None
  last_drawn_point = None
  MIN_DISTANCE = 8    # 减小距离阈值
  SAMPLING_INTERVAL = 2
  SMOOTH_BUFFER_SIZE = 4
  INTERPOLATION_POINTS = 6  # 减少插值点数量
  LINE_THICKNESS = 5  # 增加线条粗细
  
  # 添加状态计数器，用于防止意外断触
  lost_frames = 0
  MAX_LOST_FRAMES = 10
  last_valid_hand = None
  
  # 创建画布
  _, first_frame = cap.read()
  canvas = np.zeros_like(first_frame)
  
  # 用于实时平滑的点缓冲区
  points_buffer = deque(maxlen=SMOOTH_BUFFER_SIZE)

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    frame_count += 1
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    display_image = image.copy()
    
    # 检测是否有手势
    hand_detected = False
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            display_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

        # 获取所有手指关节的坐标
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

        # 调整手势判断阈值，使其更容易识别
        is_index_up = index_finger_tip.y < index_finger_pip.y - 0.02  # 降低食指伸直的要求
        is_middle_down = middle_finger_tip.y > middle_finger_pip.y  # 中指只要低于关节点即可
        is_ring_down = ring_finger_tip.y > ring_finger_pip.y  # 无名指只要低于关节点即可
        is_pinky_down = pinky_tip.y > pinky_pip.y  # 小指只要低于关节点即可
        is_thumb_in = True  # 移除拇指的限制

        # 放宽判断条件：只要食指伸直且其他手指大致弯曲即可
        if is_index_up and (is_middle_down or is_ring_down or is_pinky_down):
            hand_detected = True
            last_valid_hand = hand_landmarks
            lost_frames = 0  # 重置丢失帧计数
            
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            current_point = (cx, cy)
            
            # 实时平滑处理
            points_buffer.append(current_point)
            smoothed_point = smooth_point(current_point, points_buffer)

            # 只在采样间隔帧数到达时记录点
            if frame_count % SAMPLING_INTERVAL == 0:
                if not recording:
                    index_finger_tips.append(None)
                    recording = True
                    last_point = smoothed_point
                    last_drawn_point = smoothed_point
                    index_finger_tips.append(smoothed_point)
                    cv2.circle(canvas, smoothed_point, LINE_THICKNESS, (0, 0, 255), -1)
                elif calculate_distance(smoothed_point, last_point) >= MIN_DISTANCE:
                    interp_points = interpolate_points(last_drawn_point, smoothed_point, INTERPOLATION_POINTS)
                    for i in range(1, len(interp_points)):
                        cv2.line(canvas, interp_points[i-1], interp_points[i], (0, 0, 255), LINE_THICKNESS)
                    
                    index_finger_tips.append(smoothed_point)
                    last_point = smoothed_point
                    last_drawn_point = smoothed_point

            # 在显示图像上绘制当前点
            cv2.circle(display_image, smoothed_point, LINE_THICKNESS+2, (0, 255, 0), -1)
    
    # 处理手势丢失的情况
    if not hand_detected:
        lost_frames += 1
        if lost_frames >= MAX_LOST_FRAMES:  # 只有连续丢失足够多帧才重置状态
            recording = False
            points_buffer.clear()
            last_drawn_point = None
            last_valid_hand = None
        elif last_valid_hand and recording:  # 如果只是短暂丢失，使用最后一个有效的手势位置
            index_finger_tip = last_valid_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            current_point = (cx, cy)
            points_buffer.append(current_point)

    # 将canvas叠加到display_image上
    display_image = cv2.addWeighted(display_image, 1, canvas, 0.7, 0)

    # 显示镜像图像
    cv2.imshow('MediaPipe Hands', cv2.flip(display_image, 1))
    
    # 检测按键
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC键退出
        break
    elif key == 32:  # 空格键清屏并保存截图
        if len(index_finger_tips) > 0:
            filename, text = save_screenshot(cv2.flip(canvas, 1))
            if text:
                print(f"保存图片: {filename}")
                print(f"识别文字: {text}")
        canvas = np.zeros_like(first_frame)
        index_finger_tips = []
        points_buffer.clear()
        last_drawn_point = None
        recording = False
        lost_frames = 0
        last_valid_hand = None

cap.release()
cv2.destroyAllWindows()