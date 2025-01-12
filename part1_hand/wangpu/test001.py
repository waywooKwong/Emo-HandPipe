import cv2
import numpy as np
import mediapipe as mp
import time
import os
import pytesseract
from collections import deque
import math

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

    try:
        # 使用基本的OCR配置
        text = pytesseract.image_to_string(
            processed_image,
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

def create_smooth_curve(points, smoothing=10):
    """使用贝塞尔曲线创建平滑曲线"""
    if len(points) < 3:
        return points

    # 创建平滑曲线
    points = np.array(points)

    # 先对原始点进行轻微平滑处理，减少抖动
    kernel_size = 5  # 增加kernel size以获得更平滑的效果
    kernel = np.ones(kernel_size) / kernel_size
    x = points[:, 0]
    y = points[:, 1]
    if len(x) >= kernel_size:
        x = np.convolve(x, kernel, mode='valid')
        y = np.convolve(y, kernel, mode='valid')
        points = np.column_stack((x, y))

    # 生成更多的点以实现平滑效果
    t = np.linspace(0, 1, len(points) * smoothing)
    t_original = np.linspace(0, 1, len(points))

    # 使用三次样条插值，比线性插值更平滑
    smooth_x = np.interp(t, t_original, points[:, 0])
    smooth_y = np.interp(t, t_original, points[:, 1])

    # 对插值后的点再次进行平滑
    window = 7  # 增加窗口大小以获得更平滑的效果
    smooth_x = np.convolve(smooth_x, np.ones(window)/window, mode='valid')
    smooth_y = np.convolve(smooth_y, np.ones(window)/window, mode='valid')

    # 返回平滑后的点
    return list(zip(smooth_x.astype(int), smooth_y.astype(int)))

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

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # 初始化变量
    index_finger_tips = []
    recording = False
    saved = False
    current_stroke = []
    all_strokes = []
    finger_up_start = 0
    finger_detected = False
    DELAY_THRESHOLD = 0.5
    save_start_time = 0
    save_delay = 1.0
    save_confirmed = False
    eraser_mode = False
    eraser_radius = 60
    exit_start_time = 0
    exit_delay = 1.0
    exit_confirmed = False
    
    # 新增变量
    frame_count = 0
    last_point = None
    last_drawn_point = None
    MIN_DISTANCE = 8
    SAMPLING_INTERVAL = 2
    SMOOTH_BUFFER_SIZE = 4
    INTERPOLATION_POINTS = 6
    LINE_THICKNESS = 5
    points_buffer = deque(maxlen=SMOOTH_BUFFER_SIZE)
    lost_frames = 0
    MAX_LOST_FRAMES = 10
    last_valid_hand = None

    # 创建画布
    _, first_frame = cap.read()
    canvas = np.zeros_like(first_frame)

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
        
        # 检测是否有手势
        hand_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # 获取所有手指的坐标
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # 获取对应的 PIP 关节
                index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

                h, w, _ = image.shape

                # 修改手掌检测的条件
                if (thumb_tip.y < thumb_tip.x and  # 拇指伸开
                    all(finger.y < pip.y - 0.04 for finger, pip in [  # 降低判定阈值，使更容易检测到手掌打开
                        (index_finger_tip, index_finger_pip),
                        (middle_finger_tip, middle_finger_pip),
                        (ring_finger_tip, ring_finger_pip),
                        (pinky_tip, pinky_pip)])):

                    eraser_mode = True
                    # 获取食指位置作为橡皮擦的位置
                    eraser_x, eraser_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    # 绘制橡皮擦的视觉提示（红色圆圈，更明显）
                    cv2.circle(image, (eraser_x, eraser_y), eraser_radius, (0, 0, 255), 2)
                    cv2.circle(image, (eraser_x, eraser_y), 3, (0, 0, 255), -1)  # 中心点

                    # 擦除该范围内的所有笔画
                    for stroke in all_strokes[:]:  # 使用切片创建副本进行迭代
                        new_stroke = []
                        for point in stroke:
                            # 使用更大的判定范围
                            if abs(point[0] - eraser_x) > eraser_radius * 0.7 and abs(point[1] - eraser_y) > eraser_radius * 0.7:
                                new_stroke.append(point)

                        if len(new_stroke) > 0:
                            stroke[:] = new_stroke
                        else:
                            all_strokes.remove(stroke)

                    # 同样处理当前正在绘制的笔画
                    if recording:
                        new_current_stroke = []
                        for point in current_stroke:
                            if abs(point[0] - eraser_x) > eraser_radius * 0.7 and abs(point[1] - eraser_y) > eraser_radius * 0.7:
                                new_current_stroke.append(point)
                        current_stroke[:] = new_current_stroke
                        if not current_stroke:  # 如果当前笔画被完全擦除
                            recording = False

                    # 擦除历史轨迹
                    new_finger_tips = []
                    for point in index_finger_tips:
                        if point is None or (abs(point[0] - eraser_x) > eraser_radius * 0.7 and abs(point[1] - eraser_y) > eraser_radius * 0.7):
                            new_finger_tips.append(point)
                    index_finger_tips[:] = new_finger_tips

                    continue  # 跳过正常的绘制逻辑
                else:
                    eraser_mode = False

                # 判断是否只伸出了食指
                if (index_finger_tip.y < index_finger_pip.y and
                    middle_finger_tip.y > index_finger_pip.y and
                    ring_finger_tip.y > index_finger_pip.y and
                    pinky_tip.y > index_finger_pip.y):

                    hand_detected = True
                    last_valid_hand = hand_landmarks
                    lost_frames = 0

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
                    cv2.circle(image, smoothed_point, LINE_THICKNESS+2, (0, 255, 0), -1)

                else:
                    if recording:
                        recording = False
                        points_buffer.clear()
                        last_drawn_point = None

                # 判断是否食指和中指都伸直（识别手势）
                if (index_finger_tip.y < index_finger_pip.y and
                    middle_finger_tip.y < middle_finger_pip.y and
                    ring_finger_tip.y > ring_finger_pip.y and
                    pinky_tip.y > pinky_pip.y):

                    # 获取更多关节点来判断方向
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                    # 计算食指和中指的方向向量
                    index_direction = index_finger_tip.y - index_mcp.y
                    middle_direction = middle_finger_tip.y - middle_mcp.y

                    # 使用多个点的平均方向，提高准确性
                    finger_direction = (index_direction + middle_direction) / 2

                    # 手指朝上，触发识别操作
                    if finger_direction < -0.08:
                        # 添加确认过程的视觉反馈
                        if not save_confirmed:
                            save_start_time = time.time()
                            save_confirmed = True

                        # 在手指位置绘制确认进度圈
                        elapsed_time = time.time() - save_start_time
                        if elapsed_time < save_delay:
                            # 计算进度（0到360度）
                            progress = int((elapsed_time / save_delay) * 360)
                            # 在食指指尖位置绘制绿色进度圈
                            center = (w - int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                            radius = 30
                            cv2.ellipse(image, center, (radius, radius), -90, 
                                      0, progress, (0, 255, 0), 2)
                            # 显示识别提示
                            temp_image = np.zeros_like(image)
                            center = (image.shape[1] // 2, image.shape[0] // 2)
                            cv2.putText(temp_image, "Recognizing...", (center[0] - 40, center[1] - 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            temp_image = cv2.flip(temp_image, 1)
                            image = cv2.addWeighted(image, 1, temp_image, 1, 0)

                        # 当达到延迟时间后执行识别操作
                        elif not saved:
                            # 创建一个白色背景的图像
                            white_image = np.ones_like(image) * 255
                            # 将canvas中的内容绘制到白色图像上（镜像处理）
                            mirrored_canvas = cv2.flip(canvas.copy(), 1)
                            white_image = cv2.bitwise_and(white_image, white_image, mask=cv2.bitwise_not(mirrored_canvas))

                            # 执行文字识别
                            recognized_char = recognize_text(white_image)
                            if recognized_char:
                                # 在右上角显示识别结果
                                cv2.putText(image, f"Recognized: {recognized_char}", (30, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            saved = True
                            # 清空所有点和连线
                            index_finger_tips.clear()
                            all_strokes.clear()
                            # 清空画布
                            canvas = np.zeros_like(first_frame)
                            # 重置绘制状态
                            recording = False
                            points_buffer.clear()
                            last_drawn_point = None

                # 检测大拇指点赞手势来结束程序
                elif (thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y - 0.15 and
                      abs(thumb_tip.x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x) < 0.1 and
                      index_finger_tip.y > index_finger_pip.y and
                      middle_finger_tip.y > middle_finger_pip.y and
                      ring_finger_tip.y > ring_finger_pip.y and
                      pinky_tip.y > pinky_pip.y):

                    # 添加退出确认过程的视觉反馈
                    if not exit_confirmed:
                        exit_start_time = time.time()
                        exit_confirmed = True
                        save_confirmed = False

                    # 在大拇指位置绘制退出确认进度圈
                    elapsed_time = time.time() - exit_start_time
                    if elapsed_time < exit_delay:
                        # 计算进度（0到360度）
                        progress = int((elapsed_time / exit_delay) * 360)
                        center = (w - int(thumb_tip.x * w), int(thumb_tip.y * h))
                        radius = 30
                        cv2.circle(image, center, radius, (0, 255, 255), 2)
                        cv2.ellipse(image, center, (radius, radius), -90, 
                                  0, progress, (0, 255, 255), 4)
                        cv2.circle(image, center, 4, (0, 255, 255), -1)

                        temp_image = np.zeros_like(image)
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        cv2.putText(temp_image, "Exiting...", (center[0] - 40, center[1] - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        temp_image = cv2.flip(temp_image, 1)
                        image = cv2.addWeighted(image, 1, temp_image, 1, 0)

                    # 当达到延迟时间后退出程序
                    elif elapsed_time >= exit_delay:
                        text = "Goodbye!"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = (w - text_size[0]) // 2
                        text_y = (h + text_size[1]) // 2

                        temp_image = np.zeros_like(image)
                        cv2.putText(temp_image, text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        temp_image = cv2.flip(temp_image, 1)
                        image = cv2.addWeighted(image, 1, temp_image, 1, 0)

                        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                        cv2.waitKey(1000)

                        cap.release()
                        cv2.destroyAllWindows()
                        break

                else:
                    saved = False
                    save_confirmed = False
                    exit_confirmed = False

        # 处理手势丢失的情况
        if not hand_detected:
            lost_frames += 1
            if lost_frames >= MAX_LOST_FRAMES:
                recording = False
                points_buffer.clear()
                last_drawn_point = None
                last_valid_hand = None
            elif last_valid_hand and recording:
                index_finger_tip = last_valid_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = image.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                current_point = (cx, cy)
                points_buffer.append(current_point)

        # 将canvas叠加到display_image上
        image = cv2.addWeighted(image, 1, canvas, 0.7, 0)

        # 显示所有笔画
        for stroke in all_strokes:
            for i in range(1, len(stroke)):
                cv2.line(image, stroke[i-1], stroke[i], (0, 0, 255), 2)

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()