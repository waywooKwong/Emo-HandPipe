import streamlit as st
from components.camera import CameraComponent
import time

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="Emo-HandPipe",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 添加CSS样式
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            font-size: 3rem !important;
            text-align: center;
            color: #1E88E5;
            margin-bottom: 2rem;
        }
        .stMarkdown {
            font-size: 1.2rem;
        }
        .stSuccess {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stInfo {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .demo-text {
            font-size: 1.1rem;
            color: #666;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 标题和描述
    st.title("✨ Emo-HandPipe: 情感手写交互系统")
    st.markdown("---")
    
    # 简介和演示视频
    st.markdown("### 🎯 系统简介")
    st.markdown("""
    Emo-HandPipe 是一个创新的人机交互系统，它结合了：
    - 📝 空中手写识别
    - 😊 实时表情分析
    - 🤖 智能对话生成
    
    通过这些技术的融合，为用户提供了一种全新的交互体验。
    """)
    
    # 演示部分
    st.markdown("### 🎬 系统演示")
    demo_col1, demo_col2 = st.columns([1, 1])
    
    with demo_col1:
        st.markdown("#### 演示视频")
        # 这里使用一个示例视频
        video_placeholder = st.empty()
        video_placeholder.markdown("""
        <div style='border: 2px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;'>
        [演示视频播放区域]<br>
        模拟手写"你好"并竖起大拇指的过程
        </div>
        """, unsafe_allow_html=True)
        
        # 模拟视频播放进度
        progress_bar = st.progress(0)
        if "demo_started" not in st.session_state:
            st.session_state.demo_started = False
        
        if st.button("▶️ 播放演示"):
            st.session_state.demo_started = True
            
    with demo_col2:
        st.markdown("#### 分析结果")
        result_placeholder = st.empty()
        
        # 如果开始演示，显示模拟的分析过程
        if st.session_state.get("demo_started", False):
            # 模拟视频播放和分析过程
            for i in range(101):
                progress_bar.progress(i)
                if i == 30:
                    result_placeholder.info("✍️ 手写识别中...")
                elif i == 60:
                    result_placeholder.info("""
                    ✍️ 手写内容：\n\n**"你好"**\n
                    😊 情绪状态：\n\n**"happy"**
                    """)
                elif i == 90:
                    result_placeholder.success("""
                    🤖 AI回答：\n\n**"看起来你心情不错！你用开心的表情写下了问候，让我们开始愉快的对话吧！"**
                    """)
                time.sleep(0.02)
    
    st.markdown("---")
    
    # # 使用说明
    # with st.expander("📖 使用说明", expanded=True):
    #     st.markdown("""
    #     ### 操作指南：
        
    #     1. 🖐️ **手写输入**
    #        - 使用食指在空中书写
    #        - 保持手势清晰可见
        
    #     2. ✌️ **保存内容**
    #        - 做出比耶手势
    #        - 系统会保存当前内容并清空
        
    #     3. 👍 **完成输入**
    #        - 竖起大拇指
    #        - 系统会保存内容和表情
    #        - 自动进行分析并生成回答
    #     """)
    
    # # 实际交互区域
    # st.markdown("### 💫 开始体验")
    # interact_col1, interact_col2 = st.columns([2, 1])
    
    # with interact_col1:
    #     st.subheader("🎥 交互区域")
    #     camera = CameraComponent()
    #     hand_result, emotion_result = camera.show()
        
    #     if st.session_state.get("processing_stage"):
    #         st.info(f"🔄 当前状态: {st.session_state.processing_stage}")
    
    # with interact_col2:
    #     st.subheader("📊 识别结果")
    #     if hand_result:
    #         st.info(f"✍️ 手写内容：\n\n**{hand_result}**")
    #     if emotion_result:
    #         st.info(f"😊 情绪状态：\n\n**{emotion_result}**")
    #     if "response" in st.session_state:
    #         st.success(f"🤖 AI回答：\n\n**{st.session_state.response.get('answer_part', '')}**")
    
    # 添加页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Emo-HandPipe © 2024 | 基于手势识别和情感分析的智能交互系统
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 