from deepface import DeepFace
import cv2
import json
import time

def analyze_emotion():
    """
    使用DeepFace分析图片中的人脸属性，返回年龄、主要性别和主要情绪。

    Args:
        img_path (str): 图片文件的路径

    Returns:
        dict or str: 如果分析成功，返回包含age、dominant_gender和dominant_emotion的字典；
                    如果失败，返回"natural"
    """
    attributes = ["age", "gender", "emotion"]
    img_path = "part1_hand\pic\emo\user_emo.png"

    # 添加重试机制
    max_retries = 3
    retry_delay = 1  # 延迟1秒

    for attempt in range(max_retries):
        try:
            # 使用cv2读取图片检查文件是否可访问
            img = cv2.imread(img_path)
            if img is None:
                print(f"尝试 {attempt + 1}/{max_retries}: 无法读取图片，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                continue

            emo_info = DeepFace.analyze(img_path, actions=attributes)
            # 提取需要的信息
            result = {
                'age': emo_info[0]['age'],
                'gender': emo_info[0]['dominant_gender'],
                'emotion': emo_info[0]['dominant_emotion']
            }
            print(f"成功分析情绪信息: {result}")
            return result

        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries}: 分析图片时发生错误, 检查您的光线与摄像头配置情况, 确保光线充足: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                img_path = "part1_hand\pic\emo\default_user_emo.png"
                emo_info = DeepFace.analyze(img_path, actions=attributes)
                # 提取需要的信息
                result = {
                    'age': emo_info[0]['age'],
                    'gender': emo_info[0]['dominant_gender'],
                    'emotion': emo_info[0]['dominant_emotion']
                }
                return result

if __name__ == "__main__": 
    # 测试代码
    result = analyze_emotion()
    print(f"图片中人物情绪信息为：{result}")