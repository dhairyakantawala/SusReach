import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from functions import process_pdf_get_ans


st.set_page_config(page_title="SusReach BRSR Assistant", page_icon="📄", layout="wide")


def save_uploaded_pdf(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def make_download(df):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer


st.title("SusReach BRSR Assistant")
st.write("Upload a BRSR PDF and receive a curated CSV of answers")

with st.form("upload_form"):
    uploaded_pdf = st.file_uploader("Choose a BRSR PDF", type=["pdf"])
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("Processing may take several minutes while embeddings and answers are generated.")
    with col2:
        submit = st.form_submit_button("Run Analysis", type="primary")

if submit and uploaded_pdf:
    status = st.status("Preparing to process the report", expanded=True)
    status.write("Saving uploaded PDF")
    pdf_path = save_uploaded_pdf(uploaded_pdf)
    try:
        status.write("Generating embeddings and aligning questions")
        with st.spinner("Analyzing PDF..."):
            result_df = process_pdf_get_ans(pdf_path)
        status.write("Formatting results")
        download_buffer = make_download(result_df)
        status.update(label="Processing complete", state="complete")
        st.success("The report has been processed.")
        st.metric("Questions analyzed", len(result_df))
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=download_buffer,
            file_name="susreach_brsr_output.csv",
            mime="text/csv",
        )
    finally:
        Path(pdf_path).unlink(missing_ok=True)
elif submit and not uploaded_pdf:
    st.error("Please upload a PDF to begin.")

