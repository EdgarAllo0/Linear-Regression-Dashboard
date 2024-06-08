# Libraries
import streamlit as st
from streamlit_option_menu import option_menu

# Modules
from src.linear_regression_dashboard import OLSExampleLayout

# Set the website layouts
st.set_page_config(
    page_title="OLS Analysis",
    layout="wide",
)

# Set the sidebar content
with st.sidebar:
    st.header('Linear Regression Analysis')

    selected = option_menu(
        None,
        ["Explained example", "Try your own example"],
        icons=['house', "cash"],
        menu_icon="cast",
        default_index=0
    )

    st.subheader('A Work by Not a Recommendation')

    st.text('Author: Edgar Alcántara')

    st.link_button(
        "Check out my LinkedIn profile",
        "https://www.linkedin.com/in/edgar-mauricio-alc%C3%A1ntara-l%C3%B3pez-33505b237/"
    )

    st.link_button(
        "Go check my Portfolio on GitHub",
        "https://github.com/EdgarAllo0"
    )

if selected == 'Explained example':

    st.title("OLS Explained Example")

    OLSExampleLayout()

elif selected == 'Try your own example':

    st.title('Try Your Own Example')

    st.warning('We are working on it...', icon="⚠️")
