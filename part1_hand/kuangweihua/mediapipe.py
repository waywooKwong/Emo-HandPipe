import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class HandGestureRecognizer:
    def __init__(self):
        # 更新 MediaPipe 导入方式
        import mediapipe.tasks.python.vision.hand_landmarker as mp_hands
        self.hands = mp_hands.HandLandmarker.create(
            base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """处理每一帧图像并返回处理后的图像和手部关键点"""
        # 更新处理方式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.hands.detect(mp_image)
        
        # 存储所有检测到的手的关键点
        all_hands_landmarks = []
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # 绘制手部关键点和连接线
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # 收集关键点坐标
                landmarks = []
                for lm in hand_landmarks:
                    h, w, _ = frame.shape
                    landmarks.append({
                        'x': int(lm.x * w),
                        'y': int(lm.y * h),
                        'z': lm.z
                    })
                all_hands_landmarks.append(landmarks)
        
        return frame, all_hands_landmarks

    def recognize_gesture(self, landmarks: List) -> str:
        """基于手部关键点识别手势
        这里可以添加更多的手势识别逻辑
        """
        if not landmarks:
            return "No hand detected"
        
        # 这里添加简单的手势识别逻辑
        # 后续可以扩展更多的手势
        return "Hand detected"

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # 处理帧
        processed_frame, hand_landmarks = recognizer.process_frame(frame)
        
        # 识别手势
        gesture = recognizer.recognize_gesture(hand_landmarks)
        
        # 显示手势文本
        cv2.putText(
            processed_frame,
            f"Gesture: {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 显示结果
        cv2.imshow('Hand Gesture Recognition', processed_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
