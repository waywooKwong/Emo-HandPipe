import pytesseract
from PIL import Image
import cv2
import numpy as np

# 设置 Tesseract 路径（根据实际情况，推荐默认安装，反正加上所有语言包一共700m）
pytesseract.pytesseract.tesseract_cmd = r"E:\Software\Tesseract\tesseract.exe"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite('preprocessed_image.png', thresh)
    return 'preprocessed_image.png'

# 预处理图像
processed_image_path = preprocess_image('preprocessed_image.png')

# 识别文字
image = Image.open(processed_image_path)
text = pytesseract.image_to_string(image, lang='chi_sim', config='--oem 3 --psm 10')#第二个参数指定语言包，这里是中文
print(text.strip())