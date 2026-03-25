import html
import streamlit as st


def load_responsive_css():
    """
    Inject responsive CSS styles once at app startup.
    Call this function at the beginning of your Streamlit app.
    """
    st.markdown(
        """
        <style>
        /* ============================================
           RESPONSIVE CSS FOR RESEARCHMATE
           Breakpoints:
           - Desktop: default (>1024px)
           - Tablet: max-width 1024px
           - Mobile: max-width 600px
        ============================================ */

        /* CSS Variables */
        :root {
            --chat-max-width: 900px;
            --banner-max-width: 900px;
            --bubble-max-width: 75%;
            --chat-input-width: 65%;

            /* Desktop spacing */
            --chat-padding-x: 100px;
            --banner-margin-x: auto;

            /* Colors */
            --user-bubble-bg: #ff3b3b;
            --user-bubble-border: #ff5c5c;
            --user-text-color: #ffffff;
            --assistant-bubble-bg: #2b2b2b;
            --assistant-bubble-border: #3d3d3d;
            --assistant-text-color: #f1f1f1;
            --banner-bg: #1a1a1a;
            --banner-border: #1a1a1a;
            --banner-text: #f1f1f1;
            --input-bg: #262626;
            --input-border: #404040;
            --input-focus-border: #ff3b3b;
        }

        /* ============================================
           CHAT CONTAINER
        ============================================ */
        .chat-container {
            width: 100%;
            max-width: 900px;
            margin: 0 auto;   /* THIS IS MISSING */
            display: flex;
            box-sizing: border-box;
            padding: 24px 16px 0 16px;
        }
        .chat-container.user {
            justify-content: flex-end;
        }

        .chat-container.assistant {
            justify-content: flex-start;
        }

        /* ============================================
           CHAT BUBBLE
        ============================================ */
        .chat-bubble {
            padding: 15px 16px;
            max-width: 65%;
            font-size: 14px;
            line-height: 1.8;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .chat-bubble.user {
            background-color: var(--user-bubble-bg);
            color: var(--user-text-color);
            border: 1px solid var(--user-bubble-border);
            border-radius: 18px 18px 4px 18px;
        }

        .chat-bubble.assistant {
            background-color: var(--assistant-bubble-bg);
            color: var(--assistant-text-color);
            border: 1px solid var(--assistant-bubble-border);
            border-radius: 18px 18px 18px 4px;
        }

        /* ============================================
           BANNER (Mode, Processing, Success, Error, Generating)
        ============================================ */
        .banner {
            max-width: var(--banner-max-width);
            margin: 0.5rem auto 1rem auto;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            background: var(--banner-bg);
            border: 1px solid var(--banner-border);
            color: var(--banner-text);
            font-size: 0.9rem;
            text-align: center;
            box-sizing: border-box;
        }

        .banner.status {
            max-width: var(--banner-max-width);
            margin: 0.5rem auto 1rem auto;   /* center align */
            text-align: left;
        }

        /* ============================================
           TABLET BREAKPOINT (max-width: 1024px)
        ============================================ */
        @media screen and (max-width: 1024px) {
            :root {
                --chat-padding-x: 40px;
                --bubble-max-width: 80%;
                --chat-input-width: 85%;
            }

            .chat-container {
                padding: 20px 40px 0 40px;
            }

            .chat-bubble {
                max-width: 80%;
            }

            .banner {
                max-width: 90%;
                margin-left: auto;
                margin-right: auto;
            }

            .banner.status {
                margin-left: 40px;
                margin-right: 40px;
            }
        }

        /* ============================================
           MOBILE BREAKPOINT (max-width: 600px)
        ============================================ */
        @media screen and (max-width: 600px) {
            :root {
                --chat-padding-x: 12px;
                --bubble-max-width: 92%;
                --chat-input-width: 96%;
            }

            .chat-container {
                padding: 12px 12px 0 12px;
            }

            .chat-bubble {
                max-width: 92%;
                padding: 12px 14px;
                font-size: 13px;
                line-height: 1.7;
            }

            .banner {
                max-width: 96%;
                margin-left: auto;
                margin-right: auto;
                padding: 0.5rem 0.8rem;
                font-size: 0.85rem;
            }

            .banner.status {
                margin-left: 12px;
                margin-right: 12px;
            }
        }

        /* ============================================
           EXTRA SMALL DEVICES (max-width: 400px)
        ============================================ */
        @media screen and (max-width: 400px) {
            :root {
                --chat-input-width: 98%;
            }

            .chat-container {
                padding: 8px 8px 0 8px;
            }

            .chat-bubble {
                max-width: 95%;
                padding: 10px 12px;
                font-size: 12px;
            }

            .banner {
                max-width: 98%;
                padding: 0.4rem 0.6rem;
                font-size: 0.8rem;
            }

            .banner.status {
                margin-left: 8px;
                margin-right: 8px;
            }
        }

        /* ============================================
           STREAMLIT OVERRIDES FOR RESPONSIVENESS
        ============================================ */
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        @media screen and (max-width: 1024px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }

        @media screen and (max-width: 600px) {
            .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def load_chat_input_css():
    """
    Inject CSS styles for the Streamlit chat input component.
    Makes the input box centered, wider, and visually balanced.
    Call this function once at app startup.
    """
    st.markdown(
        """
        <style>
        /* ============================================
           CHAT INPUT STYLING
           Targets Streamlit's chat input components
        ============================================ */

        /* Bottom block container - full width base */
        div[data-testid="stBottomBlockContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            background: transparent !important;
        }

        /* Main chat input wrapper */
        div[data-testid="stChatInput"] {
            width: 65% !important;
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 0 !important;
        }

        /* Chat input form container */
        div[data-testid="stChatInput"] > div {
            width: 100% !important;
            background-color: #262626 !important;
            border: 1px solid #404040 !important;
            border-radius: 24px !important;
            padding: 4px 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }

        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #ff3b3b !important;
            box-shadow: 0 2px 12px rgba(255, 59, 59, 0.2) !important;
        }

        /* Textarea styling */
        textarea[data-testid="stChatInputTextArea"] {
            width: 100% !important;
            min-height: 44px !important;
            max-height: 200px !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
            background-color: transparent !important;
            border: none !important;
            color: #f1f1f1 !important;
            resize: none !important;
            outline: none !important;
        }

        textarea[data-testid="stChatInputTextArea"]::placeholder {
            color: #888888 !important;
            opacity: 1 !important;
        }

        textarea[data-testid="stChatInputTextArea"]:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Send button container */
        div[data-testid="stChatInput"] button {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 40px !important;
            min-height: 40px !important;
            width: 40px !important;
            height: 40px !important;
            border-radius: 50% !important;
            background-color: #ff3b3b !important;
            border: none !important;
            cursor: pointer !important;
            transition: background-color 0.2s ease, transform 0.1s ease !important;
            margin: 2px !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
        }

        div[data-testid="stChatInput"] button:hover {
            background-color: #ff5252 !important;
            transform: scale(1.05) !important;
        }

        div[data-testid="stChatInput"] button:active {
            transform: scale(0.95) !important;
        }

        div[data-testid="stChatInput"] button svg {
            width: 20px !important;
            height: 20px !important;
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        /* Disabled state */
        div[data-testid="stChatInput"] button:disabled {
            background-color: #404040 !important;
            cursor: not-allowed !important;
            transform: none !important;
        }

        /* Inner flex container alignment */
        div[data-testid="stChatInput"] > div > div {
            display: flex !important;
            align-items: flex-end !important;
            gap: 8px !important;
        }

        /* ============================================
           TABLET BREAKPOINT (max-width: 1024px)
        ============================================ */
        @media screen and (max-width: 1024px) {
            div[data-testid="stChatInput"] {
                width: 85% !important;
                max-width: 700px !important;
            }

            textarea[data-testid="stChatInputTextArea"] {
                padding: 10px 14px !important;
                font-size: 14px !important;
            }

            div[data-testid="stChatInput"] button {
                min-width: 38px !important;
                min-height: 38px !important;
                width: 38px !important;
                height: 38px !important;
            }

            div[data-testid="stChatInput"] button svg {
                width: 18px !important;
                height: 18px !important;
            }
        }

        /* ============================================
           MOBILE BREAKPOINT (max-width: 600px)
        ============================================ */
        @media screen and (max-width: 600px) {
            div[data-testid="stBottomBlockContainer"] {
                padding-left: 8px !important;
                padding-right: 8px !important;
            }

            div[data-testid="stChatInput"] {
                width: 96% !important;
                max-width: 100% !important;
            }

            div[data-testid="stChatInput"] > div {
                border-radius: 20px !important;
                padding: 3px 6px !important;
            }

            textarea[data-testid="stChatInputTextArea"] {
                padding: 10px 12px !important;
                font-size: 14px !important;
                min-height: 40px !important;
            }

            div[data-testid="stChatInput"] button {
                min-width: 36px !important;
                min-height: 36px !important;
                width: 36px !important;
                height: 36px !important;
            }

            div[data-testid="stChatInput"] button svg {
                width: 16px !important;
                height: 16px !important;
            }
        }

        /* ============================================
           EXTRA SMALL DEVICES (max-width: 400px)
        ============================================ */
        @media screen and (max-width: 400px) {
            div[data-testid="stBottomBlockContainer"] {
                padding-left: 4px !important;
                padding-right: 4px !important;
            }

            div[data-testid="stChatInput"] {
                width: 98% !important;
            }

            div[data-testid="stChatInput"] > div {
                border-radius: 18px !important;
                padding: 2px 4px !important;
            }

            textarea[data-testid="stChatInputTextArea"] {
                padding: 8px 10px !important;
                font-size: 13px !important;
                min-height: 36px !important;
            }

            div[data-testid="stChatInput"] button {
                min-width: 34px !important;
                min-height: 34px !important;
                width: 34px !important;
                height: 34px !important;
            }

            div[data-testid="stChatInput"] button svg {
                width: 14px !important;
                height: 14px !important;
            }
        }

        /* ============================================
           BOTTOM SPACING ADJUSTMENT
        ============================================ */
        div[data-testid="stBottomBlockContainer"] > div {
            padding-bottom: 1rem !important;
        }

        @media screen and (max-width: 600px) {
            div[data-testid="stBottomBlockContainer"] > div {
                padding-bottom: 0.5rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_message(role: str, content: str):
    """
    Render a chat message bubble with responsive styling.

    Args:
        role: Either "user" or "assistant"
        content: The message content to display
    """
    role_class = "user" if role == "user" else "assistant"

    st.markdown(
        f"""
        <div class="chat-container {role_class}">
            <div class="chat-bubble {role_class}">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_mode_banner():
    """
    Render the current mode banner (Agent, RAG, or Chat mode).
    Uses responsive CSS classes.
    """
    if st.session_state.get("agent_enabled", False):
        icon = "🤖"
        mode_text = "Agent Mode"
        description = "Advanced research capabilities enabled"
    elif st.session_state.get("document_id"):
        icon = "📄"
        mode_text = "RAG Mode"
        description = "Querying uploaded document"
    else:
        icon = "💬"
        mode_text = "Chat Mode"
        description = "General conversation"

    st.markdown(
        f"""
        <div class="banner">
            {icon} <b>{mode_text}</b> — {description}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_processing_banner(placeholder, text: str):
    """
    Render a processing status banner.

    Args:
        placeholder: Streamlit placeholder object
        text: Status message to display
    """
    placeholder.markdown(
        f"""
        <div class="banner status">
            ⏳ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_success_banner(placeholder, text: str):
    """
    Render a success status banner.

    Args:
        placeholder: Streamlit placeholder object
        text: Success message to display
    """
    placeholder.markdown(
        f"""
        <div class="banner status">
            ✅ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_error_banner(placeholder, text: str):
    """
    Render an error status banner.

    Args:
        placeholder: Streamlit placeholder object
        text: Error message to display
    """
    placeholder.markdown(
        f"""
        <div class="banner status">
            ❌ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_generating_banner(placeholder, text: str):
    """
    Render a generating/loading status banner.

    Args:
        placeholder: Streamlit placeholder object
        text: Loading message to display
    """
    placeholder.markdown(
        f"""
        <div class="banner status">
            ✨ <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
