import requests
import time
import hashlib
import base64
import json
import cv2
import numpy as np
from typing import Dict, Optional

# 讯飞API配置
API_URL = "https://webapi.xfyun.cn/v1/service/v1/ocr/handwriting"
APPID = "faa9c394"
API_KEY = "c58daacd25f9424f836b4d4cd09a63b7"

def get_header() -> Dict[str, str]:
    """生成API请求头，增加错误处理和参数配置"""
    cur_time = str(int(time.time()))
    param = {
        "language": "cn|en",  # 支持中英文混合识别
        "location": "false",   # 不需要返回位置信息
        "content": "all",      # 识别所有内容
        "character": "false"   # 不限制单字识别
    }
    
    try:
        param_base64 = base64.b64encode(json.dumps(param).encode('utf-8')).decode('utf-8')
        checksum = hashlib.md5((API_KEY + cur_time + param_base64).encode('utf-8')).hexdigest()
        
        return {
            "X-Appid": APPID,
            "X-CurTime": cur_time,
            "X-Param": param_base64,
            "X-CheckSum": checksum,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"
        }
    except Exception as e:
        print(f"生成请求头时出错: {str(e)}")
        return {}

def preprocess_image(image):
    """图像预处理优化"""
    try:
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整图像大小，确保合适的尺寸
        height, width = gray.shape
        if height > 1000 or width > 1000:
            scale = min(1000/height, 1000/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # 修改为普通二值化
            11,
            2
        )
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)  # 增大kernel大小
        # 闭运算连接断开的笔画
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # 开运算去除小噪点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # 转回BGR格式
        processed = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        
        # 保存处理后的图像用于调试
        cv2.imwrite('preprocessed_image.png', processed)
        
        return processed
    except Exception as e:
        print(f"图像预处理出错: {str(e)}")
        return image

def recognize_single_char(image_path: str) -> Optional[str]:
    """识别汉字"""
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 图像预处理
        processed_image = preprocess_image(image)
        
        # 转换图片为base64，使用PNG格式以保持质量
        _, img_encoded = cv2.imencode('.png', processed_image)
        image_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # 最多重试3次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 发送请求
                response = requests.post(
                    API_URL,
                    headers=get_header(),
                    data={"image": image_base64},
                    timeout=10
                )
                
                # 打印完整的响应内容用于调试
                print(f"API响应: {response.text}")
                
                # 解析结果
                result = response.json()
                if result.get('code') == '0':
                    try:
                        blocks = result['data']['block']
                        if not blocks:
                            continue
                            
                        # 获取所有识别到的文字
                        all_text = ""
                        for block in blocks:
                            for line in block['line']:
                                for word in line['word']:
                                    all_text += word['content']
                        
                        if all_text:
                            # 返回第一个字符
                            return all_text[0]
                            
                    except (KeyError, IndexError) as e:
                        print(f"解析识别结果时出错: {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                else:
                    print(f"识别失败 (尝试 {attempt + 1}/{max_retries}): {result.get('desc')}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    
            except requests.exceptions.RequestException as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                
        return None
        
    except Exception as e:
        print(f"识别过程发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试识别
    char = recognize_single_char("screenshots\handwriting_20250113_172706.png")
    if char:
        print("识别结果:", char)
    else:
        print("未识别到字符")