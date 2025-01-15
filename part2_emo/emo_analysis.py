from deepface import DeepFace
import cv2
import json

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
    img_path = "part1_hand/pic/emo/user_emo.png"
    try:
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
        print(f"分析图片时发生错误(返回默认值 natural): {str(e)}")
        return "natural"

if __name__ == "__main__": 
    # 测试代码
    result = analyze_emotion()
    print(f"图片中人物情绪信息为：{result}")