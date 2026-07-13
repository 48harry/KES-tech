import streamlit as st
import tempfile
import os
from agent import run_agent, setup_pdf_database, generate_automatic_report
import logging

st.set_page_config(page_title="KEA Tech-GPT", page_icon="🤖")
st.title("🤖 KEA Tech-GPT (동적 문서 분석)")

# --- [로거(Logger) 세팅] ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_usage.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 🔥 [핵심 수정 포인트] 앱이 시작되자마자 가장 먼저 빈 대화 바구니를 만들어줍니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. 사이드바: 파일 업로드 UI ---
with st.sidebar:
    st.header("📄 분석할 문서 첨부")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if uploaded_file is not None:
        file_sig = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("processed_file_sig") != file_sig:
            if uploaded_file.size > 5 * 1024 * 1024:
                st.error("🚨 파일이 너무 큽니다!")
            else:
                with st.spinner("문서를 분석하고 DB를 굽는 중입니다..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    st.session_state["retriever"] = setup_pdf_database(tmp_file_path)
                    os.remove(tmp_file_path)  # 임시 파일 정리
                st.session_state["processed_file_sig"] = file_sig
                st.success("✅ 문서 분석 준비 완료!")
        else:
            st.success("✅ 문서 분석 준비 완료!")

# --- 기존 사이드바 코드 아래에 추가 ---
with st.sidebar:
    st.divider() # 구분선
    st.header("📊 자동 보고서 생성")
    
    # 대화 내용이 있을 때만 버튼 활성화
    if len(st.session_state.messages) > 1:
        if st.button("📝 대화 내용으로 보고서 만들기"):
            st.session_state["report_content"] = generate_automatic_report(st.session_state.messages)

        if st.session_state.get("report_content"):
            with st.expander("👀 보고서 미리보기"):
                st.markdown(st.session_state["report_content"])
            st.download_button("📥 보고서 파일(.md) 다운로드",
                data=st.session_state["report_content"],
                file_name="Tech_Analysis_Report.md", mime="text/markdown")
    else:
        st.info("채팅을 시작하면 보고서 생성 기능이 활성화됩니다.")

logger = logging.getLogger(__name__)
logger.info("🚀 KEA Tech-GPT 앱이 성공적으로 시작되었습니다.")

# --- 2. 기존 채팅 UI (동일함) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리 (logging+except처리)

if prompt := st.chat_input("질문을 입력해주세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI가 생각 중입니다..."):
            MAX_MESSAGES = 6
            optimized_messages = st.session_state.messages[-MAX_MESSAGES:]
            try:
                response = run_agent(optimized_messages, retriever=st.session_state.get("retriever"))
                logger.info(f"[AI 답변 완료] ...")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                logger.error(f"[치명적 오류 발생] AI 응답 생성 중 에러: {e}")
                response = "죄송합니다, 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
                st.error(response)
                st.session_state.messages.append({"role": "assistant", "content": response})