"""
app.py
AI-Powered Study Assistant with Authentication, Multi-Language Support, Voice Q&A,
and Hybrid ChatGPT / General Knowledge Mode + Chat Clearing

Features:
- Secure Local Authentication (Login / Signup with bcrypt hashing in users.json)
- Per-user data isolation (User A's notes and chat history never leak to User B)
- Multi-Language UI and Answer Generation (English, Hindi, Telugu, Tamil)
- Hybrid Answering Mode: Prioritizes course notes when relevant, answers general/random questions like ChatGPT!
- Clear Options: Dedicated "💬 Clear Chat History" button and "🗑️ Clear Knowledge Base" with confirmation dialogs
- Browser-native Web Speech API Voice Input (Speech-to-Text) & Voice Output (Text-to-Speech)
- Ultra High-Visibility Buttons with prominent vibrant gradients and solid borders
- High-contrast, WCAG-compliant frosted-glass theme with 135deg gradient background
- Strictly grounded RAG pipeline with source traceability & Extractive Fallback Mode
"""

import os
import shutil
import tempfile
import html
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from rag_engine import RagEngine, Embedder
import auth
from ui_strings import UI_STRINGS, SUPPORTED_LANGUAGES, t


# ---------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ==========================================================================
   1. CLEAN MODERN NIGHT BACKGROUND (Deep Slate & Ambient Dark Glow)
   ========================================================================== */
[data-testid="stAppViewContainer"], .stApp {
    background-color: #07090e !important;
    background-image: 
        radial-gradient(at 100% 0%, rgba(79, 70, 229, 0.16) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 50%, #0d111a 0px, #07090e 100%) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
    color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ==========================================================================
   2. MAIN CONTAINER & SIDEBAR (SOLID HIGH-CONTRAST SURFACES)
   ========================================================================== */
.main .block-container {
    background: #111827 !important;
    border: 1.5px solid #1f293d !important;
    border-radius: 18px !important;
    padding: 2.2rem 2.8rem !important;
    margin-top: 1rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6) !important;
}

/* Universal text visibility inside main container */
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4,
.main .block-container h5,
.main .block-container h6 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

.main .block-container p,
.main .block-container span,
.main .block-container li,
.main .block-container div,
.main .block-container strong,
.main .block-container em {
    color: #F8FAFC !important;
}

.main .block-container [data-testid="stCaptionContainer"] p,
.stCaption {
    color: #CBD5E1 !important;
    font-size: 0.88rem !important;
}

/* Sidebar surface */
[data-testid="stSidebar"] {
    background: #0d111a !important;
    border-right: 1.5px solid #1f293d !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
}

/* Universal text visibility inside sidebar */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* User profile header badge */
.user-profile-badge {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1e1b4b !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 14px;
}
.user-profile-badge .user-name {
    font-size: 0.94rem;
    font-weight: 700;
    color: #FFFFFF !important;
}

/* ==========================================================================
   3. SELECTBOX, RADIO BUTTONS & EXPANDERS
   ========================================================================== */
/* Selectbox container & trigger */
[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

[data-baseweb="select"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-baseweb="popover"], [data-baseweb="menu"] {
    background-color: #1e293b !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 10px !important;
}

li[role="option"] {
    background-color: #1e293b !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
}

li[role="option"]:hover, li[aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Radio button options text */
div[role="radiogroup"] label,
div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #161f30 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] summary svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
    color: #F8FAFC !important;
}

/* ==========================================================================
   4. BUTTONS: ULTRA HIGH-CONTRAST & EXPLICIT VISIBILITY ACROSS ALL TYPES
   ========================================================================== */
/* Universal Base Button Style */
div.stButton > button, 
button[kind="primary"], 
button[kind="secondary"],
.stButton button,
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-header"],
button[data-testid="baseButton-download"],
div[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.3rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
    cursor: pointer !important;
    text-align: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    letter-spacing: 0.02em !important;
    opacity: 1.0 !important;
}

/* Force pure white text inside all buttons */
div.stButton > button *,
button[kind="primary"] *,
button[kind="secondary"] *,
.stButton button *,
div[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* PRIMARY BUTTONS: Solid Indigo Accent (Log In, Create Account, Build KB, Confirmation Yes) */
div.stButton > button[kind="primary"], 
div.stButton > button[data-testid="baseButton-primary"],
button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 2px solid #A78BFA !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5) !important;
}

div.stButton > button[kind="primary"]:hover, 
div.stButton > button[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: #4F46E5 !important;
    border-color: #C4B5FD !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.75) !important;
}

/* SECONDARY BUTTONS: Solid Slate/Navy Card (Sample Notes, Clear KB Trigger, Clear Chat Trigger, Cancel, Starter Prompts) */
div.stButton > button[kind="secondary"], 
div.stButton > button[data-testid="baseButton-secondary"],
div.stButton > button:not([kind="primary"]),
button[kind="secondary"] {
    background: #1E293B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1.5px solid #6366F1 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}

div.stButton > button[kind="secondary"]:hover, 
div.stButton > button[data-testid="baseButton-secondary"]:hover,
div.stButton > button:not([kind="primary"]):hover,
button[kind="secondary"]:hover {
    background: #312E81 !important;
    border-color: #A78BFA !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

/* DISABLED BUTTON STATE: Visible solid outline & readable grey text */
button:disabled, 
button[disabled], 
div.stButton > button:disabled,
button[data-testid="baseButton-primary"]:disabled,
button[data-testid="baseButton-secondary"]:disabled {
    background: #1E293B !important;
    border: 1.5px solid #475569 !important;
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
    opacity: 0.7 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

button:disabled *, 
button[disabled] * {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
}

/* Tabs styling */
[data-baseweb="tab-list"] {
    background: #0d111a !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1.5px solid #334155 !important;
    gap: 6px !important;
}

[data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 18px !important;
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
    font-weight: 700 !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.12) !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.5) !important;
}

/* ==========================================================================
   5. INPUTS & ICON-ONLY BUTTONS (Password Eye, File Uploader)
   ========================================================================== */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border-radius: 10px !important;
    border: 2px solid #CBD5E1 !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    padding: 9px 12px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    opacity: 1 !important;
}

/* PASSWORD EYE ICON BUTTON: Clear Dark Button on White Input */
[data-testid="stTextInput"] button,
[data-testid="stTextInput"] button[aria-label*="password" i],
button[aria-label="Show password text"],
button[aria-label="Hide password text"] {
    background-color: #F1F5F9 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
    margin: 2px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

[data-testid="stTextInput"] button svg,
button[aria-label="Show password text"] svg,
button[aria-label="Hide password text"] svg {
    fill: #0F172A !important;
    color: #0F172A !important;
    stroke: #0F172A !important;
    width: 18px !important;
    height: 18px !important;
}

[data-testid="stTextInput"] button:hover {
    background-color: #E2E8F0 !important;
    border-color: #94A3B8 !important;
}

/* Slider styling */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stSlider div[data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBar"] div,
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] .stSlider [role="slider"] {
    background-color: #6366F1 !important;
    border: 2px solid #FFFFFF !important;
}

/* FILE UPLOADER & "BROWSE FILES" BUTTON */
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #161f30 !important;
    border: 2px dashed #6366F1 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section p {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section svg {
    fill: #818CF8 !important;
    color: #818CF8 !important;
}

/* File Uploader "Browse files" button */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
[data-testid="stFileUploaderBrowseFilesButton"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: 1.5px solid #A78BFA !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.45) !important;
    padding: 7px 16px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover {
    background: #4F46E5 !important;
    border-color: #C4B5FD !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.65) !important;
}

/* File item remove / delete button */
[data-testid="stFileUploader"] button[aria-label*="Delete" i],
[data-testid="stFileUploader"] button[aria-label*="Remove" i] {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] button[aria-label*="Delete" i] svg,
[data-testid="stFileUploader"] button[aria-label*="Remove" i] svg {
    fill: #F87171 !important;
    color: #F87171 !important;
}

/* ==========================================================================
   6. CHAT MESSAGES (High-Contrast Solid Cards & Crisp White Text)
   ========================================================================== */
div[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
}

/* Assistant message card */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
    background: #1e293b !important;
    border: 1.5px solid #334155 !important;
    border-radius: 18px 18px 18px 4px !important;
    margin-right: auto !important;
    max-width: 92% !important;
}

/* User message card */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: #312e81 !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 18px 18px 4px 18px !important;
    margin-left: auto !important;
    max-width: 85% !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em,
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 0.98rem !important;
    line-height: 1.6 !important;
}

div[data-testid="stChatMessage"] code {
    background: #0f172a !important;
    color: #38BDF8 !important;
    -webkit-text-fill-color: #38BDF8 !important;
    border-radius: 6px !important;
    padding: 3px 7px !important;
    font-size: 0.9em !important;
    border: 1px solid #334155 !important;
}

div[data-testid="stChatMessage"] hr {
    border: none !important;
    border-top: 1px solid #334155 !important;
    margin: 1.2rem 0 !important;
}

.assistant-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #E0E7FF !important;
    -webkit-text-fill-color: #E0E7FF !important;
    background: #4338ca !important;
    padding: 4px 11px;
    border-radius: 6px;
    margin-bottom: 12px;
    border: 1px solid #6366f1 !important;
}

/* ==========================================================================
   7. CHAT INPUT BAR & SEND ARROW BUTTON
   ========================================================================== */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 2.5px solid #6366f1 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45) !important;
    padding: 4px !important;
}

[data-testid="stChatInput"] textarea {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    background-color: transparent !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    opacity: 1 !important;
}

/* Chat Send Button: Solid Accent Color & Pure White Arrow Icon */
[data-testid="stChatInputSubmitButton"],
button[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button {
    background: #6366F1 !important;
    border: 1.5px solid #A78BFA !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    opacity: 1.0 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInputSubmitButton"]:hover,
button[data-testid="stChatInputSubmitButton"]:hover {
    background: #4F46E5 !important;
    border-color: #C4B5FD !important;
    transform: scale(1.05) !important;
}

[data-testid="stChatInputSubmitButton"] svg,
button[data-testid="stChatInputSubmitButton"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

/* ==========================================================================
   8. RESTYLED ALERTS & METRICS
   ========================================================================== */
[data-testid="stAlert"], .stAlert {
    border-radius: 14px !important;
    padding: 1rem 1.3rem !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 1rem !important;
}

[data-testid="stAlert"] *, .stAlert * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]),
div[data-testid="stAlert"][data-test-color="info"],
.stAlert:has(svg[data-testid="stAlertIcon-info"]) {
    background-color: #0c4a6e !important;
    border: 1.5px solid #0284c7 !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
div[data-testid="stAlert"][data-test-color="success"],
.stAlert:has(svg[data-testid="stAlertIcon-success"]) {
    background-color: #064e3b !important;
    border: 1.5px solid #059669 !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
div[data-testid="stAlert"][data-test-color="warning"],
.stAlert:has(svg[data-testid="stAlertIcon-warning"]) {
    background-color: #78350f !important;
    border: 1.5px solid #d97706 !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
div[data-testid="stAlert"][data-test-color="error"],
.stAlert:has(svg[data-testid="stAlertIcon-error"]) {
    background-color: #7f1d1d !important;
    border: 1.5px solid #dc2626 !important;
}

[data-testid="stMetric"] {
    background-color: #161f30 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}

[data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] *, [data-testid="stMetricValue"] div {
    color: #38BDF8 !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
}

/* ==========================================================================
   9. SOURCE PILLS & METRIC BADGES
   ========================================================================== */
.sources-pill-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #334155;
}

.sources-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #CBD5E1 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #FFFFFF !important;
    background: #3730A3 !important;
    padding: 4px 12px;
    border-radius: 9999px;
    border: 1.5px solid #6366f1 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

.metric-badge-container {
    display: flex;
    gap: 8px;
    margin: 12px 0 16px 0;
}

.metric-badge-card {
    flex: 1;
    background: #161f30 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
}

.metric-badge-card .metric-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #38BDF8 !important;
    line-height: 1.2;
}

.metric-badge-card .metric-lbl {
    font-size: 0.68rem;
    font-weight: 700;
    color: #CBD5E1 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 3px;
}

/* ==========================================================================
   10. SCROLLBAR & FOOTER
   ========================================================================== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0d111a;
}
::-webkit-scrollbar-thumb {
    background: #6366f1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #818cf8;
}

.app-footer {
    margin-top: 2.8rem;
    padding-top: 1.4rem;
    border-top: 1px solid #1f293d;
    text-align: center;
}

.pipeline-flow {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.65rem;
}

.pipeline-flow .step-badge {
    background: #161f30;
    border: 1.5px solid #334155;
    color: #F8FAFC !important;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 8px;
}

.pipeline-flow .arrow {
    color: #818cf8 !important;
    font-size: 0.8rem;
    font-weight: bold;
}

.credit-line {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
    margin-top: 0.35rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Cached Embedding Model Resource
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def get_cached_embedder() -> Embedder:
    """Cache sentence-transformer embedding model across user sessions."""
    return Embedder("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# 3. Session State Initialization
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "current_lang" not in st.session_state:
    st.session_state.current_lang = "en"

if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = "hybrid"

if "engine" not in st.session_state:
    embedder = get_cached_embedder()
    st.session_state.engine = RagEngine(embedder=embedder)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_info" not in st.session_state:
    st.session_state.indexed_info = {
        "status": "Not Built",
        "chunks_count": 0,
        "sources": [],
    }

if "confirm_clear_kb" not in st.session_state:
    st.session_state.confirm_clear_kb = False

if "confirm_clear_chat" not in st.session_state:
    st.session_state.confirm_clear_chat = False

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"


# ---------------------------------------------------------------------------
# 4. Helper Functions: Speech, Audio & Pills
# ---------------------------------------------------------------------------
def render_source_pills(sources: list, lang: str = "en") -> str:
    """Renders source filenames as an HTML pill list with high contrast."""
    if not sources:
        return ""
    pills_html = "".join(f'<span class="source-pill">📄 {s}</span>' for s in sources)
    label = t("sources_label", lang)
    return f"""
    <div class="sources-pill-row">
        <span class="sources-label">{label}</span>
        {pills_html}
    </div>
    """


def render_voice_input_widget(lang_code: str = "en"):
    """
    Renders an HTML5 / Web Speech API interactive voice recorder component.
    Speech recognition runs directly in the client browser with native language support.
    """
    speech_lang = SUPPORTED_LANGUAGES.get(lang_code, {}).get("speech_code", "en-US")
    native_name = SUPPORTED_LANGUAGES.get(lang_code, {}).get("native", "English")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 4px;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .voice-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                background: #111827;
                border: 1.5px solid #334155;
                border-radius: 12px;
                padding: 8px 14px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.4);
            }}
            .mic-btn {{
                background: #6366F1;
                border: 2px solid #A78BFA;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                color: #FFFFFF;
                font-size: 1.15rem;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
                transition: all 0.2s ease;
            }}
            .mic-btn:hover {{
                background: #4F46E5;
                border-color: #C4B5FD;
                transform: scale(1.08);
                box-shadow: 0 6px 18px rgba(99, 102, 241, 0.7);
            }}
            .mic-btn.recording {{
                background: #DC2626;
                border-color: #F87171;
                animation: pulse 1.5s infinite;
                box-shadow: 0 0 16px rgba(220, 38, 38, 0.8);
            }}
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.1); }}
                100% {{ transform: scale(1); }}
            }}
            .transcript-box {{
                flex: 1;
                font-size: 0.92rem;
                font-weight: 600;
                color: #F8FAFC;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .copy-btn {{
                background: #6366F1;
                border: 1.5px solid #A78BFA;
                color: #FFFFFF;
                padding: 6px 14px;
                border-radius: 8px;
                font-size: 0.82rem;
                cursor: pointer;
                font-weight: 700;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                transition: all 0.2s ease;
            }}
            .copy-btn:hover {{
                background: #4F46E5;
                border-color: #C4B5FD;
                transform: translateY(-1px);
            }}
        </style>
    </head>
    <body>
        <div class="voice-container">
            <button id="micBtn" class="mic-btn" title="Click to speak question">🎙️</button>
            <div id="transcript" class="transcript-box">Click mic to speak in {native_name} ({speech_lang})</div>
            <button id="copyBtn" class="copy-btn" style="display:none;">📋 Copy</button>
        </div>

        <script>
            const micBtn = document.getElementById('micBtn');
            const transcript = document.getElementById('transcript');
            const copyBtn = document.getElementById('copyBtn');
            let recognition = null;
            let isRecording = false;

            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

            if (SpeechRec) {{
                recognition = new SpeechRec();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = '{speech_lang}';

                recognition.onstart = () => {{
                    isRecording = true;
                    micBtn.classList.add('recording');
                    transcript.innerText = 'Listening... Speak now';
                }};

                recognition.onresult = (event) => {{
                    let text = '';
                    for (let i = 0; i < event.results.length; i++) {{
                        text += event.results[i][0].transcript;
                    }}
                    transcript.innerText = text;
                    copyBtn.style.display = 'inline-block';
                }};

                recognition.onerror = (event) => {{
                    transcript.innerText = 'Error: ' + event.error;
                    isRecording = false;
                    micBtn.classList.remove('recording');
                }};

                recognition.onend = () => {{
                    isRecording = false;
                    micBtn.classList.remove('recording');
                }};

                micBtn.onclick = () => {{
                    if (isRecording) {{
                        recognition.stop();
                    }} else {{
                        recognition.start();
                    }}
                }};

                copyBtn.onclick = () => {{
                    navigator.clipboard.writeText(transcript.innerText);
                    copyBtn.innerText = '✅ Copied!';
                    setTimeout(() => {{ copyBtn.innerText = '📋 Copy'; }}, 2000);
                }};
            }} else {{
                transcript.innerText = 'Web Speech API is not supported in this browser.';
                micBtn.disabled = true;
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=64)


def render_speech_synthesis_button(text: str, lang_code: str = "en", button_id: str = "speak_btn"):
    """Renders a browser Web Speech API SpeechSynthesis audio player button."""
    speech_lang = SUPPORTED_LANGUAGES.get(lang_code, {}).get("speech_code", "en-US")
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}
            .tts-btn {{
                background: #1E293B;
                border: 1.5px solid #6366F1;
                color: #FFFFFF;
                padding: 6px 14px;
                border-radius: 8px;
                font-size: 0.8rem;
                font-weight: 700;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.3);
                transition: all 0.2s ease;
            }}
            .tts-btn:hover {{
                background: #312E81;
                border-color: #A78BFA;
                transform: translateY(-1px);
                box-shadow: 0 5px 14px rgba(99, 102, 241, 0.45);
            }}
        </style>
    </head>
    <body>
        <button id="{button_id}" class="tts-btn">🔊 {t('read_aloud', lang_code)}</button>
        <script>
            document.getElementById('{button_id}').onclick = function() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance("{clean_text}");
                    utterance.lang = '{speech_lang}';
                    utterance.rate = 1.0;
                    window.speechSynthesis.speak(utterance);
                }} else {{
                    alert('Speech synthesis not supported in this browser.');
                }}
            }};
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=38)


# ---------------------------------------------------------------------------
# 5. Authentication Screen (Login / Sign-Up)
# ---------------------------------------------------------------------------
if not st.session_state.authenticated:
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #FFFFFF; font-size: 2.2rem; margin-bottom: 6px;">🎓 AI Study Assistant</h1>
                <p style="color: #CBD5E1; font-size: 0.95rem;">Private Course Knowledge Base & Grounded RAG Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Create Account"])

        with tab_login:
            st.subheader("Student Login")
            login_user = st.text_input("Username", key="auth_login_user", placeholder="e.g. ramya")
            login_pwd = st.text_input("Password", type="password", key="auth_login_pwd", placeholder="••••••••")
            
            if st.button("🚀 Log In", use_container_width=True, type="primary"):
                success, msg, user_info = auth.login_user(login_user, login_pwd)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user_info
                    
                    # Load user's private chat history and indexed files
                    saved_messages, saved_indexed_info = auth.load_user_state(user_info["username"])
                    st.session_state.messages = saved_messages
                    st.session_state.indexed_info = saved_indexed_info
                    
                    # If user previously indexed sample notes, rebuild index in memory
                    if saved_indexed_info.get("status") == "Ready" and "sample_notes.txt" in saved_indexed_info.get("sources", []):
                        if SAMPLE_DOCS_DIR.exists():
                            st.session_state.engine.ingest_folder(str(SAMPLE_DOCS_DIR))
                    
                    st.success(f"Welcome back, {user_info['full_name']}!")
                    st.rerun()
                else:
                    st.error(msg)

        with tab_signup:
            st.subheader("New Student Sign Up")
            signup_name = st.text_input("Full Name", key="auth_signup_name", placeholder="e.g. Ramya Krishnan")
            signup_user = st.text_input("Username", key="auth_signup_user", placeholder="e.g. ramya")
            signup_pwd = st.text_input("Password", type="password", key="auth_signup_pwd", placeholder="Minimum 4 characters")
            
            if st.button("✨ Create Account", use_container_width=True, type="primary"):
                success, msg = auth.signup_user(signup_user, signup_pwd, signup_name)
                if success:
                    st.success(msg)
                    st.info("You can now switch to the 'Log In' tab and enter your credentials.")
                else:
                    st.error(msg)

    st.stop()


# ---------------------------------------------------------------------------
# 6. Authenticated Application Header & Sidebar UI
# ---------------------------------------------------------------------------
current_user = st.session_state.user or {"username": "student", "full_name": "Student"}
username = current_user["username"]
full_name = current_user["full_name"]
current_lang = st.session_state.current_lang

with st.sidebar:
    # User Profile & Logout Header
    st.markdown(
        f"""
        <div class="user-profile-badge">
            <div>
                <span style="font-size:0.75rem; color:#CBD5E1; text-transform:uppercase; font-weight:700;">{t('logged_in_as', current_lang)}</span>
                <div class="user-name">👋 {full_name} (@{username})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if st.button(t("logout_button", current_lang), use_container_width=True):
        # Save user session before logout
        auth.save_user_state(username, st.session_state.messages, st.session_state.indexed_info)
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.indexed_info = {"status": "Not Built", "chunks_count": 0, "sources": []}
        st.rerun()

    st.divider()

    # 1. Language Selector
    st.subheader(t("language_label", current_lang))
    lang_name_map = {
        "English": "en",
        "हिन्दी (Hindi)": "hi",
        "తెలుగు (Telugu)": "te",
        "தமிழ் (Tamil)": "ta",
    }
    lang_keys = list(lang_name_map.keys())
    current_idx = list(lang_name_map.values()).index(current_lang) if current_lang in lang_name_map.values() else 0
    
    chosen_lang_name = st.selectbox(
        "Select Language",
        options=lang_keys,
        index=current_idx,
        label_visibility="collapsed",
    )
    new_lang_code = lang_name_map[chosen_lang_name]
    if new_lang_code != st.session_state.current_lang:
        st.session_state.current_lang = new_lang_code
        st.rerun()

    st.divider()

    # 2. Assistant Answering Mode (Hybrid ChatGPT vs Strict Notes)
    st.subheader(t("mode_label", current_lang))
    mode_map = {
        t("mode_hybrid", current_lang): "hybrid",
        t("mode_strict", current_lang): "strict",
    }
    mode_keys = list(mode_map.keys())
    cur_mode_idx = 0 if st.session_state.assistant_mode == "hybrid" else 1
    chosen_mode_label = st.radio(
        "Assistant Mode",
        options=mode_keys,
        index=cur_mode_idx,
        label_visibility="collapsed",
    )
    st.session_state.assistant_mode = mode_map[chosen_mode_label]

    st.divider()

    # 3. LLM API Key Input
    st.subheader(t("llm_header", current_lang))
    api_key_input = st.text_input(
        t("api_key_label", current_lang),
        type="password",
        placeholder="sk-...",
        help=t("api_key_help", current_lang),
    )
    st.session_state.engine.set_api_key(api_key_input)

    if api_key_input.strip():
        st.success(t("api_key_active", current_lang), icon="✅")
    else:
        st.info(t("fallback_mode_active", current_lang), icon="ℹ️")

    st.divider()

    # 4. Retrieval Parameters
    st.subheader(t("retrieval_header", current_lang))
    top_k = st.slider(
        t("top_k_label", current_lang),
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help=t("top_k_help", current_lang),
    )

    st.divider()

    # 5. Document Ingestion & Sample Notes Quick-Start
    st.subheader(t("materials_header", current_lang))

    uploaded_files = st.file_uploader(
        t("upload_label", current_lang),
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help=t("upload_help", current_lang),
    )

    col_build, col_sample = st.columns(2)

    with col_build:
        build_btn = st.button(
            t("build_button", current_lang),
            use_container_width=True,
            type="primary",
        )

    with col_sample:
        sample_btn = st.button(
            t("sample_button", current_lang),
            use_container_width=True,
        )

    # Handle indexing uploaded files
    if build_btn:
        if not uploaded_files:
            st.warning(t("upload_warning", current_lang))
        else:
            with st.spinner(t("indexing_spinner", current_lang)):
                with tempfile.TemporaryDirectory() as temp_dir:
                    saved_names = []
                    for uploaded_file in uploaded_files:
                        save_path = Path(temp_dir) / uploaded_file.name
                        save_path.write_bytes(uploaded_file.getvalue())
                        saved_names.append(uploaded_file.name)

                    num_chunks = st.session_state.engine.ingest_folder(temp_dir)
                    st.session_state.indexed_info = {
                        "status": "Ready",
                        "chunks_count": num_chunks,
                        "sources": saved_names,
                    }
                    auth.save_user_state(username, st.session_state.messages, st.session_state.indexed_info)
                    st.success(f"Indexed {num_chunks} chunks!")
                    st.rerun()

    # Handle indexing built-in sample notes
    if sample_btn:
        if not SAMPLE_DOCS_DIR.exists():
            st.error("Sample docs directory not found.")
        else:
            with st.spinner(t("sample_spinner", current_lang)):
                num_chunks = st.session_state.engine.ingest_folder(str(SAMPLE_DOCS_DIR))
                sample_files = [f.name for f in SAMPLE_DOCS_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".pdf"]]
                st.session_state.indexed_info = {
                    "status": "Ready",
                    "chunks_count": num_chunks,
                    "sources": sample_files,
                }
                auth.save_user_state(username, st.session_state.messages, st.session_state.indexed_info)
                st.success(f"Indexed {num_chunks} chunks from sample notes!")
                st.rerun()

    st.divider()

    # 6. Knowledge Base Status & Metric Badges
    st.subheader(t("kb_status_header", current_lang))
    status = st.session_state.indexed_info["status"]
    count = st.session_state.indexed_info["chunks_count"]
    sources = st.session_state.indexed_info["sources"]

    if status == "Ready":
        st.markdown(f"**Status:** 🟢 **{t('status_ready', current_lang)}**")
        
        metric_badges_html = f"""
        <div class="metric-badge-container">
            <div class="metric-badge-card">
                <div class="metric-val">{len(sources)}</div>
                <div class="metric-lbl">{t('files_badge', current_lang)}</div>
            </div>
            <div class="metric-badge-card">
                <div class="metric-val">{count}</div>
                <div class="metric-lbl">{t('chunks_badge', current_lang)}</div>
            </div>
            <div class="metric-badge-card">
                <div class="metric-val">MiniLM</div>
                <div class="metric-lbl">{t('model_badge', current_lang)}</div>
            </div>
        </div>
        """
        st.markdown(metric_badges_html, unsafe_allow_html=True)

        with st.expander(t("indexed_files", current_lang), expanded=False):
            for s in sources:
                st.write(f"- `{s}`")
    else:
        st.markdown(f"**Status:** ⚪ *{t('status_not_built', current_lang)}*")

    st.divider()

    # 7. Clear Options: Clear Chat History & Clear Knowledge Base
    # A. Clear Chat History
    if not st.session_state.confirm_clear_chat:
        if st.button(t("clear_chat_button", current_lang), use_container_width=True, help="Clear active conversation messages while keeping your indexed notes intact."):
            st.session_state.confirm_clear_chat = True
            st.rerun()
    else:
        st.warning(t("clear_chat_warning", current_lang))
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button(t("clear_chat_confirm", current_lang), type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.confirm_clear_chat = False
                auth.save_user_state(username, [], st.session_state.indexed_info)
                st.rerun()
        with col_c2:
            if st.button(t("clear_chat_cancel", current_lang), use_container_width=True):
                st.session_state.confirm_clear_chat = False
                st.rerun()

    # B. Clear Knowledge Base
    if not st.session_state.confirm_clear_kb:
        if st.button(t("clear_kb_button", current_lang), use_container_width=True, help="Reset the vector store and erase all indexed documents."):
            st.session_state.confirm_clear_kb = True
            st.rerun()
    else:
        st.warning(t("clear_warning", current_lang))
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button(t("clear_confirm", current_lang), type="primary", use_container_width=True):
                st.session_state.engine = RagEngine(embedder=get_cached_embedder())
                st.session_state.messages = []
                st.session_state.indexed_info = {
                    "status": "Not Built",
                    "chunks_count": 0,
                    "sources": [],
                }
                st.session_state.confirm_clear_kb = False
                auth.save_user_state(username, [], st.session_state.indexed_info)
                st.rerun()
        with col_no:
            if st.button(t("clear_cancel", current_lang), use_container_width=True):
                st.session_state.confirm_clear_kb = False
                st.rerun()


# ---------------------------------------------------------------------------
# 7. Main Chat Area UI
# ---------------------------------------------------------------------------
st.title(t("app_title", current_lang))
st.caption(t("pipeline_caption", current_lang))

# Top Hero Metric Badges
if st.session_state.indexed_info["status"] == "Ready":
    top_metrics_html = f"""
    <div class="metric-badge-container" style="max-width: 500px; margin-bottom: 20px;">
        <div class="metric-badge-card">
            <div class="metric-val">{len(st.session_state.indexed_info['sources'])}</div>
            <div class="metric-lbl">📁 {t('files_badge', current_lang)}</div>
        </div>
        <div class="metric-badge-card">
            <div class="metric-val">{st.session_state.indexed_info['chunks_count']}</div>
            <div class="metric-lbl">🧩 {t('chunks_badge', current_lang)}</div>
        </div>
        <div class="metric-badge-card">
            <div class="metric-val">all-MiniLM-L6-v2</div>
            <div class="metric-lbl">🧠 {t('model_badge', current_lang)}</div>
        </div>
    </div>
    """
    st.markdown(top_metrics_html, unsafe_allow_html=True)

# Welcoming guide if no messages
if not st.session_state.messages:
    welcome_text = f"""
    {t('welcome_title', current_lang)}

    {t('welcome_step1', current_lang)}
    
    {t('welcome_step2', current_lang)}
    
    {t('welcome_step3', current_lang)}
    """
    st.info(welcome_text)

    # Quick starter query buttons when sample notes are ready
    if st.session_state.indexed_info["status"] == "Ready" and "sample_notes.txt" in st.session_state.indexed_info["sources"]:
        st.markdown(f"**{t('try_asking', current_lang)}**")
        sample_questions = [
            t("sq_1", current_lang),
            t("sq_2", current_lang),
            t("sq_3", current_lang),
            t("sq_4", current_lang),
        ]
        cols = st.columns(2)
        for i, sq in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"👉 {sq}", key=f"sq_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": sq})
                    with st.spinner(t("searching_spinner", current_lang)):
                        answer, sources = st.session_state.engine.answer(
                            sq,
                            top_k=top_k,
                            language=chosen_lang_name,
                            mode=st.session_state.assistant_mode,
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                    auth.save_user_state(username, st.session_state.messages, st.session_state.indexed_info)
                    st.rerun()

# Render existing chat conversation
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="assistant-badge">{t("assistant_badge", current_lang)}</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(render_source_pills(msg["sources"], current_lang), unsafe_allow_html=True)
            # Speech synthesis button for assistant response
            render_speech_synthesis_button(msg["content"], current_lang, button_id=f"tts_{idx}")
        else:
            st.markdown(msg["content"])

# Voice Input Widget (Web Speech API)
render_voice_input_widget(current_lang)

# Handle student question input
user_question = st.chat_input(t("chat_placeholder", current_lang))

if user_question:
    # 1. Display and persist user query
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Generate grounded answer with RAG Engine (in Hybrid or Strict mode)
    with st.chat_message("assistant"):
        st.markdown(f'<div class="assistant-badge">{t("assistant_badge", current_lang)}</div>', unsafe_allow_html=True)
        with st.spinner(t("searching_spinner", current_lang)):
            answer_text, source_files = st.session_state.engine.answer(
                user_question,
                top_k=top_k,
                language=chosen_lang_name,
                mode=st.session_state.assistant_mode,
            )
            st.markdown(answer_text)
            if source_files:
                st.markdown(render_source_pills(source_files, current_lang), unsafe_allow_html=True)
            render_speech_synthesis_button(answer_text, current_lang, button_id=f"tts_live")

    # 3. Persist assistant response to session state & disk
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": source_files,
    })
    auth.save_user_state(username, st.session_state.messages, st.session_state.indexed_info)


# ---------------------------------------------------------------------------
# 8. Footer Caption & Pipeline Flow
# ---------------------------------------------------------------------------
footer_html = f"""
<div class="app-footer">
    <div class="pipeline-flow">
        <span class="step-badge">📄 Documents</span>
        <span class="arrow">➔</span>
        <span class="step-badge">✂️ Chunking (~800 chars)</span>
        <span class="arrow">➔</span>
        <span class="step-badge">🧠 MiniLM-L6-v2</span>
        <span class="arrow">➔</span>
        <span class="step-badge">⚡ FAISS IndexFlatIP</span>
        <span class="arrow">➔</span>
        <span class="step-badge">🔍 Top-K Retriever</span>
        <span class="arrow">➔</span>
        <span class="step-badge">🤖 Hybrid ChatGPT Mode</span>
    </div>
    <p class="credit-line">{t('footer_credit', current_lang)}</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
