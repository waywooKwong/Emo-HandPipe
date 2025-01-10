from deepface import DeepFace
import cv2
import json

img_path = "E:\Emo-HandPipe\emo\kuangweihua\\test.jpg"

attributes = ["emotion", "age", "gender", "race"]
emo_info = DeepFace.analyze(img_path, actions=attributes)

print(type(emo_info))
print(emo_info)
