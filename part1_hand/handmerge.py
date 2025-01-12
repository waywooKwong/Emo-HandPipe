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

def calculate_angle(p1, p2):
    """计算两点之间的线段与水平线的夹角（弧度）"""
    if p1 is None or p2 is None:
        return 0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def is_near_horizontal(angle, threshold=0.15):
    """判断是否接近水平线（允许小偏差）"""
    return abs(math.sin(angle)) < threshold

def is_near_vertical(angle, threshold=0.15):
    """判断是否接近垂直线（允许小偏差）"""
    return abs(math.cos(angle)) < threshold

def smart_smooth_points(points, window_size=5):
    """智能平滑处理一组点
    
    特性：
    1. 对接近水平/垂直的线条进行矫正
    2. 保持斜线和曲线的原有特征
    3. 使用滑动窗口进行局部平滑
    4. 通过分析点的分布特征来判断是否需要矫正
    """
    if len(points) < window_size:
        return points
    
    smoothed = []
    half_window = window_size // 2
    
    # 计算整体的角度分布
    angles = []
    for i in range(len(points) - 1):
        angle = calculate_angle(points[i], points[i + 1])
        angles.append(angle)
    
    # 判断是否是一个连续的笔画（通过分析角度变化）
    angle_changes = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
    avg_angle_change = sum(angle_changes) / len(angle_changes) if angle_changes else 0
    is_stroke_straight = avg_angle_change < 0.3  # 如果角度变化小，说明是直线
    
    for i in range(len(points)):
        # 获取当前窗口内的点
        start_idx = max(0, i - half_window)
        end_idx = min(len(points), i + half_window + 1)
        window = points[start_idx:end_idx]
        
        if len(window) < 2:
            smoothed.append(points[i])
            continue
        
        # 计算窗口内点的平均位置
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        
        # 计算窗口首尾点的角度
        angle = calculate_angle(window[0], window[-1])
        
        # 获取当前点
        current = points[i]
        
        # 根据线条特征决定是否需要矫正
        if is_stroke_straight:
            if is_near_horizontal(angle):
                # 接近水平线，保持x坐标变化，y坐标使用平均值
                smoothed.append((current[0], int(avg_y)))
            elif is_near_vertical(angle):
                # 接近垂直线，保持y坐标变化，x坐标使用平均值
                smoothed.append((int(avg_x), current[1]))
            else:
                # 斜线，使用较小的平滑处理
                weight_current = 0.8  # 增加当前点的权重
                weight_avg = 0.2     # 减少平均位置的权重
                smooth_x = int(current[0] * weight_current + avg_x * weight_avg)
                smooth_y = int(current[1] * weight_current + avg_y * weight_avg)
                smoothed.append((smooth_x, smooth_y))
        else:
            # 对于非直线笔画（如撇、捺），使用更轻微的平滑处理
            weight_current = 0.9  # 进一步增加当前点的权重
            weight_avg = 0.1     # 进一步减少平均位置的权重
            smooth_x = int(current[0] * weight_current + avg_x * weight_avg)
            smooth_y = int(current[1] * weight_current + avg_y * weight_avg)
            smoothed.append((smooth_x, smooth_y))
    
    return smoothed

def interpolate_points(p1, p2, num_points=5):
    """在两点之间插值生成平滑的过渡点"""
    if p1 is None or p2 is None:
        return []
    
    # 计算两点之间的角度
    angle = calculate_angle(p1, p2)
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(p1[0] * (1 - t) + p2[0] * t)
        y = int(p1[1] * (1 - t) + p2[1] * t)
        points.append((x, y))
    
    # 根据线条特征选择不同的平滑处理
    if is_near_horizontal(angle) or is_near_vertical(angle):
        # 对于接近水平或垂直的线条，使用较强的平滑
        return smart_smooth_points(points, window_size=3)
    else:
        # 对于斜线，使用较弱的平滑
        return points

# For webcam input:
cap = cv2.VideoCapture(0)

# 设置摄像头帧率为60fps
cap.set(cv2.CAP_PROP_FPS, 60)
# 设置高分辨率以获得更好的效果
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    print("请检查：")
    print("1. 是否已连接摄像头")
    print("2. 摄像头是否被其他程序占用")
    print("3. 是否有摄像头访问权限")
    exit()

# 打印摄像头实际设置
actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头实际帧率: {actual_fps}")
print(f"摄像头实际分辨率: {actual_width}x{actual_height}")

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
  MIN_DISTANCE = 5    # 减小距离阈值，使点的记录更频繁
  SAMPLING_INTERVAL = 1  # 减小采样间隔，使采样更频繁
  SMOOTH_BUFFER_SIZE = 3  # 减小平滑缓冲区大小，使响应更快
  INTERPOLATION_POINTS = 8  # 增加插值点数量，使线条更平滑
  LINE_THICKNESS = 5  # 保持线条粗细不变
  
  # 添加状态计数器，用于防止意外断触
  lost_frames = 0
  MAX_LOST_FRAMES = 5  # 减小最大丢失帧数，使恢复更快
  last_valid_hand = None
  
  # 添加比耶手势检测相关变量
  peace_gesture_frames = 0
  PEACE_GESTURE_THRESHOLD = 15  # 保持比耶手势阈值不变
  
  # 添加竖大拇指手势检测相关变量
  thumbs_up_frames = 0
  THUMBS_UP_THRESHOLD = 15  # 竖大拇指手势阈值
  
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

        # 检测比耶手势
        is_index_straight = index_finger_tip.y < index_finger_pip.y - 0.04
        is_middle_straight = middle_finger_tip.y < middle_finger_pip.y - 0.04
        is_ring_bent = ring_finger_tip.y > ring_finger_pip.y
        is_pinky_bent = pinky_tip.y > pinky_pip.y
        
        is_peace_gesture = is_index_straight and is_middle_straight and is_ring_bent and is_pinky_bent
        
        # 检测竖大拇指手势
        is_thumb_up = thumb_tip.y < thumb_ip.y - 0.1  # 大拇指明显竖直
        is_other_fingers_down = (
            index_finger_tip.y > index_finger_pip.y and  # 食指弯曲
            middle_finger_tip.y > middle_finger_pip.y and  # 中指弯曲
            ring_finger_tip.y > ring_finger_pip.y and  # 无名指弯曲
            pinky_tip.y > pinky_pip.y  # 小指弯曲
        )
        
        is_thumbs_up_gesture = is_thumb_up and is_other_fingers_down
        
        if is_thumbs_up_gesture:
            thumbs_up_frames += 1
            # 在大拇指位置绘制确认进度圈
            if thumbs_up_frames < THUMBS_UP_THRESHOLD:
                # 获取图像尺寸
                h, w, _ = display_image.shape
                # 计算进度（0到360度）
                progress = int((thumbs_up_frames / THUMBS_UP_THRESHOLD) * 360)
                # 在大拇指指尖位置绘制黄色进度圈
                center = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                radius = 30
                # 绘制完整的圆形背景
                cv2.circle(display_image, center, radius, (0, 255, 255), 2)
                # 绘制进度弧
                cv2.ellipse(display_image, center, (radius, radius), -90, 
                           0, progress, (0, 255, 255), 4)
                # 在圆圈中心绘制一个点
                cv2.circle(display_image, center, 4, (0, 255, 255), -1)
                # 显示截图提示
                cv2.putText(display_image, "Taking photo...", 
                           (center[0] - 60, center[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if thumbs_up_frames >= THUMBS_UP_THRESHOLD:
                # 保存当前帧（不包含轨迹）
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if not os.path.exists('part1_hand/pic/emo'):
                    os.makedirs('part1_hand/pic/emo')
                filename = f'part1_hand/pic/emo/thumbs_up_{timestamp}.png'
                # 保存镜像后的图像
                cv2.imwrite(filename, cv2.flip(image, 1))
                print(f"表情照片已保存: {filename}")
                
                # 显示退出消息
                h, w, _ = display_image.shape
                text = "Goodbye!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                
                # 创建一个临时图像用于绘制镜像文字
                temp_image = np.zeros_like(display_image)
                cv2.putText(temp_image, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                # 水平翻转临时图像
                temp_image = cv2.flip(temp_image, 1)
                # 将临时图像叠加到原始图像上
                display_image = cv2.addWeighted(display_image, 1, temp_image, 1, 0)
                
                # 显示最后一帧
                cv2.imshow('MediaPipe Hands', cv2.flip(display_image, 1))
                cv2.waitKey(1500)  # 显示1.5秒告别消息
                
                # 释放摄像头并关闭所有窗口
                cap.release()
                cv2.destroyAllWindows()
                break
            continue  # 如果是竖大拇指手势，跳过后续的绘制逻辑
        else:
            thumbs_up_frames = 0  # 如果不是竖大拇指手势，重置计数器
        
        if is_peace_gesture:
            peace_gesture_frames += 1
            # 在手指位置绘制确认进度圈
            if peace_gesture_frames < PEACE_GESTURE_THRESHOLD:
                # 获取图像尺寸
                h, w, _ = display_image.shape
                # 计算进度（0到360度）
                progress = int((peace_gesture_frames / PEACE_GESTURE_THRESHOLD) * 360)
                # 在食指指尖位置绘制绿色进度圈
                center = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                radius = 30
                cv2.ellipse(display_image, center, (radius, radius), -90, 
                           0, progress, (0, 255, 0), 2)
                # 显示保存提示
                cv2.putText(display_image, "Saving...", 
                           (center[0] - 40, center[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if peace_gesture_frames >= PEACE_GESTURE_THRESHOLD:
                # 触发截屏和清屏
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
                peace_gesture_frames = 0  # 重置计数器
            continue  # 如果是比耶手势，跳过后续的绘制逻辑
        else:
            peace_gesture_frames = 0  # 如果不是比耶手势，重置计数器

        # 调整手势判断阈值，适当放宽但保持一定准确性
        is_index_up = index_finger_tip.y < index_finger_pip.y - 0.03  # 食指伸直要求适中
        is_middle_down = middle_finger_tip.y > middle_finger_pip.y  # 中指只要弯曲即可
        is_ring_down = ring_finger_tip.y > ring_finger_pip.y  # 无名指只要弯曲即可
        is_pinky_down = pinky_tip.y > pinky_pip.y  # 小指只要弯曲即可
        is_thumb_down = thumb_tip.y > thumb_ip.y - 0.01  # 拇指稍微放宽要求

        # 要求：食指必须伸直，其他手指弯曲（但要求没那么严格）
        if is_index_up and (is_middle_down and (is_ring_down or is_pinky_down)):
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
                    # 增加插值点的处理
                    interp_points = interpolate_points(last_drawn_point, smoothed_point, INTERPOLATION_POINTS)
                    if len(interp_points) > 1:
                        # 对插值点进行智能平滑
                        smooth_points = smart_smooth_points(interp_points)
                        
                        # 绘制平滑后的线段
                        for i in range(1, len(smooth_points)):
                            cv2.line(canvas, smooth_points[i-1], smooth_points[i], 
                                   (0, 0, 255), LINE_THICKNESS)
                    
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

cap.release()
cv2.destroyAllWindows()