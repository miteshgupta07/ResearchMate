import html
import streamlit as st

import streamlit as st

import streamlit as st

def render_message(role: str, content: str):
    if role == "user":
        justify = "flex-end"
        bubble_bg = "#ff3b3b"
        bubble_border = "#ff5c5c"
        bubble_radius = "18px 18px 4px 18px"
        text_color = "#ffffff"
        side_padding = "48px 165px 0 155px"   # top right bottom left
    else:
        justify = "flex-start"
        bubble_bg = "#2b2b2b"
        bubble_border = "#3d3d3d"
        bubble_radius = "18px 18px 18px 4px"
        text_color = "#f1f1f1"
        side_padding = "48px 165px 0 155px"

    st.markdown(
        f"""
        <div style="
            width: 100%;
            display: flex;
            justify-content: {justify};
            padding: {side_padding};
            box-sizing: border-box;
        ">
            <div style="
                background-color: {bubble_bg};
                color: {text_color};
                padding: 15px 16px;
                border-radius: {bubble_radius};
                max-width: 100%;
                border: 1px solid {bubble_border};
                font-size: 14px;
                line-height: 2.5;
                word-wrap: break-word;
            ">
                {content}
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
                margin: 0.5rem auto 1rem auto;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
                text-align: center;
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
                margin: 0.5rem auto 1rem auto;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
                text-align: center;
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
                margin: 0.5rem auto 1rem auto;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                background: #1a1a1a;
                border: 1px solid #1a1a1a;
                color: #f1f1f1;
                font-size: 0.9rem;
                text-align: center;
            ">
                💬 <b>Chat Mode</b> — General conversation
            </div>
            """,
            unsafe_allow_html=True,
        )


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

