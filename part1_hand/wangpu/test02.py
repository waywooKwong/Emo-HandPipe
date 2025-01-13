import pytesseract
from PIL import Image
import cv2
import numpy as np
import os

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 保存预处理后的图片
    output_path = 'preprocessed_image.png'
    cv2.imwrite(output_path, thresh)
    return output_path

def recognize_text(image_path):
    try:
        # 预处理图像
        processed_image_path = preprocess_image(image_path)
        
        # 识别文字
        image = Image.open(processed_image_path)
        text = pytesseract.image_to_string(image, 
                                         lang='chi_sim', 
                                         config='--oem 3 --psm 6')
        
        if not text.strip():
            print("未能识别到任何文字")
            return None
            
        return text.strip()
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 指定输入图片路径
    input_image = "screenshots\handwriting_20250112_230808.png"  # 替换为你的实际图片路径
    
    print("开始文字识别...")
    result = recognize_text(input_image)
    
    if result:
        print("识别结果:")
        print("-" * 50)
        print(result)
        print("-" * 50)