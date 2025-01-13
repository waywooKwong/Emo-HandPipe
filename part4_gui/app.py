import streamlit as st
from components.camera import CameraComponent

def main():
    st.set_page_config(
        page_title="Emo-HandPipe",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("Emo-HandPipe: 情感手写交互系统")
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        1. 食指在空中书写
        2. 比耶手势(✌️)保存当前内容并清空
        3. 竖起大拇指(👍)保存内容和表情，并退出摄像头
        """)
    
    # 状态显示区域
    status_container = st.empty()
    
    # 摄像头组件会自动处理手写识别、表情识别和生成回答
    camera = CameraComponent()
    hand_result, emotion_result = camera.show()
    
    # 显示当前状态
    if st.session_state.get("processing_stage"):
        status_container.info(f"🔄 当前状态: {st.session_state.processing_stage}")
    
    # 显示识别结果
    if hand_result or emotion_result:
        col1, col2 = st.columns(2)
        with col1:
            if hand_result:
                st.info(f"✍️ 手写识别结果: {hand_result}")
        with col2:
            if emotion_result:
                st.info(f"😊 情绪状态: {emotion_result}")
    
    # 显示大模型回答
    if "response" in st.session_state:
        st.success(f"🤖 AI回答: {st.session_state.response.get('answer_part', '')}")

if __name__ == "__main__":
    main() 