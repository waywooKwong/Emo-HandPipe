# 导入part2_emo中的emo_analysis和llm_generate
from part1_hand import hand_merge
from part2_emo import emo_analysis
from part3_llm import llm_generate

if __name__ == "__main__":
    # 获取手写文字
    hand_info = hand_merge.handwriting_recognition()
    # 获取图片中人物情绪信息 
    emo_img_path = "part1_hand/pic/emo/user_emo.png"
    emo_info = emo_analysis.analyze_emotion(emo_img_path)
    print("图片中人物情绪信息:", emo_info)
    # 生成行为预测结果
    result = llm_generate.model_generate("ok的手势", emo_info)
    print("预测结果:", result)