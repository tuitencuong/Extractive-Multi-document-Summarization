"""Streamlit Cloud entry point retaining the existing visualizer UI."""
import os

# Limit parallel numerical workers to suit shared Cloud resources.
for option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(option, "2")

from pathlib import Path
import streamlit as st
from web_visualizer.streamlit_bridge import render_document, dispatch_request


st.set_page_config(page_title="Tóm tắt đa văn bản", page_icon="😵‍💫", layout="wide")
st.html("""<style>
  .stMainBlockContainer { max-width: none; padding: 3.5rem 1rem .5rem; }
</style>""")


@st.cache_resource
def visualizer_component():
    javascript = (Path(__file__).parent / "web_visualizer/static/streamlit_component.js").read_text(encoding="utf-8")
    return st.components.v2.component(
        "multi_document_visualizer",
        html='<iframe class="visualizer" title="Trực quan tóm tắt đa văn bản" allow="clipboard-write" sandbox="allow-scripts allow-same-origin allow-downloads"></iframe>',
        css=".visualizer { width:100%; height:calc(100vh - 80px); min-height:780px; border:0; border-radius:12px; }",
        js=javascript,
    )


component = visualizer_component()
result = component(
    key="visualizer",
    data={"document": render_document(), "response": st.session_state.get("visualizer_response")},
    on_request_change=lambda: None,
)

request = result.request
if request and request.get("id") != st.session_state.get("visualizer_last_request"):
    st.session_state.visualizer_last_request = request.get("id")
    # Responses belong to this Streamlit session, never to a shared data cache.
    st.session_state.visualizer_response = dispatch_request(request)
    st.rerun()
