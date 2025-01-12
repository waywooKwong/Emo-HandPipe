from deepface import DeepFace
import cv2
import json

def analyze_emotion(img_path):
    """
    使用DeepFace分析图片中的人脸属性，包括年龄、性别和情绪。

    Args:
        img_path (str): 图片文件的路径

    Returns:
        dict or str: 如果分析成功，返回包含分析结果的字典；如果失败，返回"natural"
    """
    attributes = ["age", "gender", "emotion"]
    
    # 这里输出格式的信息还需要处理
    try:
        emo_info = DeepFace.analyze(img_path, actions=attributes)
        return emo_info
    except Exception as e:
        print(f"分析图片时发生错误: {str(e)}")
        return "natural"

if __name__ == "__main__":
    # 测试代码
    test_img_path = "emo\kuangweihua\\test.jpg"
    result = analyze_emotion(test_img_path)
    print(f"图片中人物情绪信息为：{result}")