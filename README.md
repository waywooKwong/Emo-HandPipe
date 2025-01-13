# Emo-HandPipe
24Fall-Python Final Project

## 测试须知

### 部分测试

测试文件分为三部分

1. part1_hand 手势识别和书写识别

    - 注意需要根据 part1_hand\readme.md 的介绍先安装 tesseract 环境

    - 测试代码 part1_hand\hand_merge.py 

    ```
    line 29 下述参数要设置正确：
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

2. part2_emo 表情识别

    - 注意先安装 deepface 环境

    ```python
    pip install deepface
    ```

    - 测试代码 part2_emo\emo_merge.py

3. part3_llm 大语言模型内容输入

    - 先启动 ollama, 且确保拉取了 qwen2.5

    ```
    ollama serve
    (如果没安装) ollama pull qwen2.5
    ```

    - 测试代码 part3_llm\llm_merge.py   

### 整体接口逻辑

整体接口逻辑代码 main.py
