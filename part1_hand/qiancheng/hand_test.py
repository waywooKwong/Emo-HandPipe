import cv2
import numpy as np
import mediapipe as mp
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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
  saved = False
  image_count = 0
  current_stroke = []    # 当前正在绘制的笔画
  all_strokes = []      # 所有完成的笔画
  finger_up_start = 0   # 记录手指抬起的开始时间
  finger_detected = False  # 是否检测到手指抬起
  DELAY_THRESHOLD = 0.5  # 延迟阈值（秒）
  save_start_time = 0    # 记录开始保存的时间
  save_delay = 1.0      # 保存操作的延迟时间（秒）
  save_confirmed = False # 是否确认保存
  eraser_mode = False
  eraser_radius = 60  # 增大橡皮擦的半径，使擦除范围更大
  delete_start_time = 0    # 记录开始删除的时间
  delete_delay = 1.0      # 删除操作的延迟时间（秒）
  delete_confirmed = False # 是否确认删除
  exit_confirmed = False  # 是否确认退出
  exit_start_time = 0    # 记录开始退出的时间
  exit_delay = 1.0       # 退出操作的延迟时间（秒）

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

        # 获取所有手指的坐标
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

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

          # 如果是第一次检测到手指抬起
          if not finger_detected:
            finger_up_start = time.time()
            finger_detected = True

          # 检查是否已经过了延迟时间
          if time.time() - finger_up_start >= DELAY_THRESHOLD:
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            if not recording:
              current_stroke = []  # 开始新的笔画
              recording = True
            current_stroke.append((cx, cy))

            # 在图像上绘制食指顶端手指节的当前位置
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # 如果点数足够，实时显示平滑后的线条
            if len(current_stroke) >= 3:
              smooth_current = create_smooth_curve(current_stroke, smoothing=15)  # 增加smoothing参数
              for i in range(1, len(smooth_current)):
                cv2.line(image, smooth_current[i-1], smooth_current[i], (0, 0, 255), 2)
            else:
              # 如果点数不够，显示原始线条
              for i in range(1, len(current_stroke)):
                cv2.line(image, current_stroke[i-1], current_stroke[i], (0, 0, 255), 2)
        else:
          finger_detected = False  # 重置检测状态
          if recording:
            # 当笔画结束时，创建平滑曲线并保存
            if len(current_stroke) >= 3:
              smooth_stroke = create_smooth_curve(current_stroke, smoothing=15)  # 增加smoothing参数
              all_strokes.append(smooth_stroke)
              # 将平滑后的点添加到历史轨迹中
              index_finger_tips.extend([None] + smooth_stroke)
            else:
              all_strokes.append(current_stroke)
              index_finger_tips.extend([None] + current_stroke)
          recording = False

        # 判断是否食指和中指都伸直（保存手势）
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

            # 手指朝上，保存操作
            if finger_direction < -0.08:
                # 添加确认过程的视觉反馈
                if not save_confirmed:
                    save_start_time = time.time()
                    save_confirmed = True
                    delete_confirmed = False  # 确保删除状态被重置

                # 在手指位置绘制确认进度圈
                elapsed_time = time.time() - save_start_time
                if elapsed_time < save_delay:
                    # 计算进度（0到360度）
                    progress = int((elapsed_time / save_delay) * 360)
                    # 在食指指尖位置绘制绿色进度圈
                    center = (w - int(index_finger_tip.x * w), int(index_finger_tip.y * h))  # 镜像x坐标
                    radius = 30
                    cv2.ellipse(image, center, (radius, radius), -90, 
                               0, progress, (0, 255, 0), 2)
                    # 显示保存提示（文字位置镜像）
                    # 创建一个临时图像用于绘制镜像文字
                    temp_image = np.zeros_like(image)
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    cv2.putText(temp_image, "Saving...", (center[0] - 40, center[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # 水平翻转临时图像
                    temp_image = cv2.flip(temp_image, 1)
                    # 将临时图像叠加到原始图像上
                    image = cv2.addWeighted(image, 1, temp_image, 1, 0)

                # 当达到延迟时间后执行保存操作
                elif not saved:
                    # 创建一个白色背景的图像
                    white_image = np.ones_like(image) * 255
                    # 只绘制平滑后的笔画（镜像处理）
                    for stroke in all_strokes:
                        mirrored_stroke = [(w - x, y) for x, y in stroke]
                        for i in range(1, len(mirrored_stroke)):
                            cv2.line(white_image, mirrored_stroke[i-1], mirrored_stroke[i], (0, 0, 255), 2)
                    if not os.path.exists('part1_hand/pic/handwriting'):
                        os.makedirs('part1_hand/pic/handwriting')
                    cv2.imwrite(f'part1_hand/pic/handwriting/hand_landmarks_{image_count}.png', white_image)
                    image_count += 1
                    saved = True


                    # 在右上角显示保存成功信息
                    cv2.putText(image, "Saved!", (30, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # 清空所有点和连线
                    index_finger_tips.clear()
                    all_strokes.clear()

        # 判断是否食指、中指和无名指都伸直（删除手势）
        elif (index_finger_tip.y < index_finger_pip.y and
              middle_finger_tip.y < middle_finger_pip.y and
              ring_finger_tip.y < ring_finger_pip.y and
              pinky_tip.y > pinky_pip.y):

            # 添加删除确认过程的视觉反馈
            if not delete_confirmed:
                delete_start_time = time.time()
                delete_confirmed = True
                save_confirmed = False  # 确保保存状态被重置

            # 在手指位置绘制删除确认进度圈
            elapsed_time = time.time() - delete_start_time
            if elapsed_time < delete_delay:
                # 计算进度（0到360度）
                progress = int((elapsed_time / delete_delay) * 360)
                # 在食指指尖位置绘制红色进度圈
                center = (w - int(index_finger_tip.x * w), int(index_finger_tip.y * h))  # 镜像x坐标
                radius = 30
                cv2.ellipse(image, center, (radius, radius), -90, 
                           0, progress, (0, 0, 255), 2)
                # 显示删除提示（文字位置镜像）
                # 创建一个临时图像用于绘制镜像文字
                temp_image = np.zeros_like(image)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                cv2.putText(temp_image, "Deleting...", (center[0] - 40, center[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 水平翻转临时图像
                temp_image = cv2.flip(temp_image, 1)
                # 将临时图像叠加到原始图像上
                image = cv2.addWeighted(image, 1, temp_image, 1, 0)

            # 当达到延迟时间后执行删除操作
            elif delete_confirmed:
                files = sorted([f for f in os.listdir('part1_hand/pic/handwriting') if f.startswith('hand_landmarks_')])
                if files:
                    last_file = os.path.join('part1_hand/pic/handwriting', files[-1])
                    try:
                        os.remove(last_file)
                        print(f"Deleted file: {last_file}")
                        image_count = max(0, image_count - 1)
                        # 在右上角显示删除成功信息
                        cv2.putText(image, "Deleted!", (30, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                delete_confirmed = False

        # 检测大拇指点赞手势来结束程序
        elif (# 大拇指竖直向上的条件
              thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y - 0.15 and  # 大拇指明显竖直
              abs(thumb_tip.x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x) < 0.1 and  # 大拇指基本垂直

              # 其他手指弯曲的条件
              index_finger_tip.y > index_finger_pip.y and  # 食指弯曲
              middle_finger_tip.y > middle_finger_pip.y and # 中指弯曲
              ring_finger_tip.y > ring_finger_pip.y and     # 无名指弯曲
              pinky_tip.y > pinky_pip.y):                   # 小指弯曲

            # 添加退出确认过程的视觉反馈
            if not exit_confirmed:
                exit_start_time = time.time()
                exit_confirmed = True
                save_confirmed = False  # 确保其他状态被重置
                delete_confirmed = False

            # 在大拇指位置绘制退出确认进度圈
            elapsed_time = time.time() - exit_start_time
            exit_delay = 1.0  # 设置1秒的确认时间

            if elapsed_time < exit_delay:
                # 计算进度（0到360度）
                progress = int((elapsed_time / exit_delay) * 360)
                # 在大拇指指尖位置绘制黄色进度圈
                center = (w - int(thumb_tip.x * w), int(thumb_tip.y * h))  # 镜像x坐标
                radius = 30
                # 绘制完整的圆形背景
                cv2.circle(image, center, radius, (0, 255, 255), 2)
                # 绘制进度弧
                cv2.ellipse(image, center, (radius, radius), -90, 
                           0, progress, (0, 255, 255), 4)
                # 在圆圈中心绘制一个点
                cv2.circle(image, center, 4, (0, 255, 255), -1)
                # 显示退出提示（文字位置镜像）              
                # 创建一个临时图像用于绘制镜像文字
                temp_image = np.zeros_like(image)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                cv2.putText(temp_image, "Exiting...", (center[0] - 40, center[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # 水平翻转临时图像
                temp_image = cv2.flip(temp_image, 1)
                # 将临时图像叠加到原始图像上
                image = cv2.addWeighted(image, 1, temp_image, 1, 0)

            # 当达到延迟时间后退出程序
            elif elapsed_time >= exit_delay:
                # 在退出前保存一张png相机图片
                if not os.path.exists('part1_hand/pic/emo'):
                    os.makedirs('part1_hand/pic/emo')
                cv2.imwrite(f'part1_hand/pic/emo/exit_emo_image.png', image)

                # 显示退出消息（在镜像后的正确位置）
                text = "Goodbye!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2

                # 创建一个临时图像用于绘制镜像文字
                temp_image = np.zeros_like(image)
                cv2.putText(temp_image, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # 水平翻转临时图像
                temp_image = cv2.flip(temp_image, 1)
                # 将临时图像叠加到原始图像上
                image = cv2.addWeighted(image, 1, temp_image, 1, 0)

                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                cv2.waitKey(1000)  # 显示1秒告别消息

                # 释放摄像头并关闭所有窗口
                cap.release()
                cv2.destroyAllWindows()
                break

        else:
            saved = False
            save_confirmed = False
            delete_confirmed = False
            exit_confirmed = False  # 重置退出确认状态

    # 只显示平滑后的笔画
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(image, stroke[i-1], stroke[i], (0, 0, 255), 2)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()