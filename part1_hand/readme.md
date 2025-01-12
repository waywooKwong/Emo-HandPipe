# 手势识别

由 钱程、王璞、邝伟华 负责实现，

钱程和邝伟华实现手势动作，王璞实现手写字符识别

## 一、手写字符识别模型

1. https://github.com/UB-Mannheim/tesseract/wiki

访问 github 地址，下载最新模型：

<img src="readme_pic/image.png" width="500" alt="Installation screenshot">

2. 打开安装包：

按照安装向导完成安装，记下安装路径（默认路径通常是 C:\Program Files\Tesseract-OCR\tesseract.exe

注意这里要把最后两个包勾选上，不然没有中文识别

<img src="readme_pic/image2.png" width="500" alt="Installation screenshot">

3. 安装 python 库

pip install pytesseract opencv-python pillow numpy

4. 简单测试

```Python
import pytesseract
from PIL import Image
import cv2
import numpy as np

# 设置 Tesseract 路径（根据实际情况，推荐默认安装，反正加上所有语言包一共700m）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite('preprocessed_image.png', thresh)
    return 'preprocessed_image.png'

# 预处理图像
processed_image_path = preprocess_image('image.png')

# 识别文字
image = Image.open(processed_image_path)
text = pytesseract.image_to_string(image, lang='chi_sim', config='--oem 3 --psm 10')#第二个参数指定语言包，这里是中文
print(text.strip())
```
