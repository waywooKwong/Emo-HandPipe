import streamlit as st
from part1_hand.hand_merge import handwriting_recognition
from part2_emo.emo_analysis import analyze_emotion
from part3_llm.llm_generate import model_generate
import os
import time

class CameraComponent:
    def __init__(self):
        if "hand_info" not in st.session_state:
            st.session_state.hand_info = ""
        if "emo_info" not in st.session_state:
            st.session_state.emo_info = "natural"
        if "error_info" not in st.session_state:
            st.session_state.error_info = None
        if "processing_stage" not in st.session_state:
            st.session_state.processing_stage = "等待手写输入..."
        
    def show(self):
        # 显示上一次的错误信息（如果有）
        if st.session_state.error_info:
            st.error(st.session_state.error_info)
            st.session_state.error_info = None
        
        try:
            # 运行手写识别（使用原始的处理窗口）
            st.session_state.processing_stage = "正在进行手写识别..."
            hand_info = handwriting_recognition()
            
            # 如果有手写内容，更新状态
            if hand_info:
                st.session_state.hand_info = hand_info
                
            # 当part1返回时（摄像头已关闭），开始后续处理
            st.session_state.processing_stage = "正在分析表情..."
            try:
                emotion_result = analyze_emotion("part1_hand/pic/emo/user_emo.png")
                if emotion_result:
                    st.session_state.emo_info = emotion_result
                    
                    st.session_state.processing_stage = "正在生成AI回答..."
                    response = model_generate(hand_info, emotion_result)
                    if response:
                        st.session_state.response = response
                        st.session_state.processing_stage = "处理完成"
                    else:
                        st.session_state.error_info = "❌ 大模型未能生成有效回答"
                else:
                    st.session_state.error_info = "❌ 表情识别结果为空"
            except Exception as e:
                st.session_state.error_info = f"❌ 后续处理出错: {str(e)}"
            
        except Exception as e:
            st.session_state.error_info = f"❌ 手写识别出错: {str(e)}"
            st.session_state.processing_stage = "等待手写输入..."
        
        return st.session_state.hand_info, st.session_state.emo_info 