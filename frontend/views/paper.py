import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer

st.title("📄 Thesis Report")
st.write("This document is written in Spanish.")


paper_path = os.path.join("data", "paper.pdf")

pdf_viewer(input=paper_path, height=700)