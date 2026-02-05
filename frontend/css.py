import html
import streamlit as st

def render_message(role: str, content: str):
    """
    Render a chat message as a styled bubble.
    - User messages: right-aligned, light blue background
    - Assistant messages: left-aligned, light gray background
    """
    # Escape HTML to prevent injection
    safe_content = html.escape(content)
    # Preserve line breaks
    safe_content = safe_content.replace("\n", "<br>")
    
    if role == "user":
        # User message: right-aligned, red-accent bubble
        st.markdown(
            f"""
            <div class="chat-wrapper">
                <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                    <div style="
                        background-color: #ff3b3b;
                        color: #ffffff;
                        padding: 12px 16px;
                        border-radius: 18px 18px 4px 18px;
                        max-width: 70%;
                        word-wrap: break-word;
                        border: 1px solid #ff5c5c;
                        font-size: 14px;
                        line-height: 1.5;
                    ">
                        {safe_content}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Assistant message: left-aligned, dark neutral bubble
        st.markdown(
            f"""
            <div class="chat-wrapper">
                <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                    <div style="
                        background-color: #2b2b2b;
                        color: #f1f1f1;
                        padding: 12px 16px;
                        border-radius: 18px 18px 18px 4px;
                        max-width: 70%;
                        word-wrap: break-word;
                        border: 1px solid #3d3d3d;
                        font-size: 14px;
                        line-height: 1.5;
                    ">
                        {safe_content}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_mode_banner():
    if st.session_state.agent_enabled:
        st.markdown(
            """
            <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
                🤖 <b>Agent Mode</b> — Advanced research capabilities enabled
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.document_id:
        st.markdown(
            """
            <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
                📄 <b>RAG Mode</b> — Querying uploaded document
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
                💬 <b>Chat Mode</b> — General conversation
            </div>
            """,
            unsafe_allow_html=True,
        )

# Styled document processing status (replace default spinner/success/error UI)

def render_processing_banner(placeholder, text):
    placeholder.markdown(
        f"""
        <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
            ⏳ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_success_banner(placeholder, text):
    placeholder.markdown(
        f"""
        <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
            ✅ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_error_banner(placeholder, text):
    placeholder.markdown(
        f"""
        <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
            ❌ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_generating_banner(placeholder, text):
    placeholder.markdown(
        f"""
       <div style="
                max-width: 1000px;
                margin: 0.5rem 11rem 1rem 11rem;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
            ">
            ✨ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

