# 导入part2_emo中的emo_analysis和llm_generate
from part1_hand import hand_merge
from part2_emo import emo_analysis
from part3_llm import llm_generate
from deepface import DeepFace
if __name__ == "__main__":
    # 1. Part1_hand 手势识别和书写识别 -> 获取手写文字
    hand_info = hand_merge.handwriting_recognition()

    # 2. Part2_emo 表情识别 -> 获取图片中人物情绪信息
    emo_info = emo_analysis.analyze_emotion()

    # 3. Part3_llm 生成行为预测结果
    result = llm_generate.model_generate(hand_info, emo_info)
    print("预测结果:", result)