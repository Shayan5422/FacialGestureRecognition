import streamlit as st
from utils import (
    render_intro_section,
    render_objective_section,
    render_architecture_section,
    render_process_section,
    render_technology_section,
    render_applications_section,
    render_live_demo_section
)

# Page configuration
st.set_page_config(
    page_title="Facial Gesture Recognition AI",
    page_icon="🧠",
    layout="wide"
)

# Custom styles for the presentation
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 24px;
        color: #1E88E5;
        margin-top: 0;
    }
    .section-header {
        font-size: 32px;
        font-weight: bold;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<p class="main-header">Facial Gesture Recognition Using AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time video processing with CNN-LSTM architecture</p>', unsafe_allow_html=True)

# Navigation sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Introduction", "Project Objective", "Architecture & Methodology", 
     "Process Flow", "Technologies", "Applications", "Live Demo"]
)

# Render the selected section
if section == "Introduction":
    render_intro_section()
elif section == "Project Objective":
    render_objective_section()
elif section == "Architecture & Methodology":
    render_architecture_section()
elif section == "Process Flow":
    render_process_section()
elif section == "Technologies":
    render_technology_section()
elif section == "Applications":
    render_applications_section()
elif section == "Live Demo":
    render_live_demo_section()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This interactive presentation showcases a facial gesture recognition system "
    "using CNN-LSTM architecture for real-time video processing."
)
