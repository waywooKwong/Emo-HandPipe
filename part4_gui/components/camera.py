import streamlit as st
from part1_hand.hand_merge import handwriting_recognition
from part2_emo.emo_analysis import analyze_emotion
from part3_llm.llm_generate import model_generate
import os

class CameraComponent:
    def __init__(self):
        if "hand_info" not in st.session_state:
            st.session_state.hand_info = ""
        if "emo_info" not in st.session_state:
            st.session_state.emo_info = "natural"
        if "processing_stage" not in st.session_state:
            st.session_state.processing_stage = "等待手写输入..."
        
    def show(self):
        # 1. Part1_hand 手势识别和书写识别 -> 获取手写文字
        st.session_state.processing_stage = "正在进行手写识别..."
        hand_info = handwriting_recognition()
        
        # 如果有手写内容，继续处理
        if hand_info:
            st.session_state.hand_info = hand_info
            
            # 2. Part2_emo 表情识别 -> 获取图片中人物情绪信息
            st.session_state.processing_stage = "正在分析表情..."
            emo_img_path = "part1_hand/pic/emo/user_emo.png"
            try:
                emotion_result = analyze_emotion(emo_img_path)
                if emotion_result:
                    st.session_state.emo_info = emotion_result
            except:
                emotion_result = "natural"
                st.session_state.emo_info = emotion_result
            
            # 3. Part3_llm 生成行为预测结果
            st.session_state.processing_stage = "正在生成AI回答..."
            result = model_generate(hand_info, emotion_result)
            if result:
                st.session_state.response = result
                st.session_state.processing_stage = "处理完成"
        
        return st.session_state.hand_info, st.session_state.emo_info 