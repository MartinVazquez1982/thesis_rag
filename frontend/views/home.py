import streamlit as st
import os

st.title("RAG-Powered Thesis Explorer")

st.subheader("Basic segmentation model of dedvices for the treatment of cerebral aneuryms from 3D medical images")

st.text("During the Supervised Professional Practice (SPP) period, a computer vision project in medicine focusing on brain aneurysm segmentation from 3D medical images was undertaken. This experience for Systems Engineers explored the application of artificial intelligence in medicine, enhancing both diagnosis and treatment. The project followed a combined approach, leveraging both in-person and remote work, with fixed schedules and seamless communication with the Yatiris team. A cyclical methodology of training and evaluation was utilized to optimize the segmentation model's performance, employing tools such as PyTorch, SLURM, and Slicer 3D.")

st.info(
    """
*This application has two sections:*

1. Paper in Spanish (PDF): an integrated view to read the full work.

2. RAG Assistant (Questions): a module to ask questions about the paper and the topic; it answers using Retrieval-Augmented Generation (RAG).

> You can switch between sections from the left sidebar menu.
"""
)