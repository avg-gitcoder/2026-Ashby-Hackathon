"""Streamlit front end for Run_Pathology_Report.ipynb's ReportSession."""
import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from PIL import Image

from notebook_runner import Settings
from staged_runner import ReportSession
from ui_support import notebook_defaults, prediction_rows, run_stages, save_upload

CONNECT = Path(__file__).resolve().parent
st.set_page_config(page_title="Ashby · Pathology workspace", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container{max-width:1220px;padding-top:2rem}
h1{letter-spacing:.04em} [data-testid="stMetric"]{border-radius:8px}
[data-testid="stSidebar"]{min-width:290px}
</style>""", unsafe_allow_html=True)
defaults = notebook_defaults(CONNECT)
base = Settings(CONNECT)
st.title("ASHBY")
st.caption("Pathology workspace · PathGen / MedGemma / MUSK")

with st.sidebar:
    st.header("Run settings")
    st.caption("Uses the same runner and saved results as the latest notebook.")
    image_root = st.text_input("Delta image folder", str(base.image_root))
    with st.expander("Model and Python paths"):
        checkpoint = st.text_input("MUSK checkpoint", str(base.checkpoint))
        pathgen_model = st.text_input("PathGen model", str(base.pathgen_model))
        pathgen_python = st.text_input("PathGen Python", str(base.pathgen_python))
        medgemma_python = st.text_input("MUSK / MedGemma Python", str(base.medgemma_python))
        medgemma_base = st.text_input("MedGemma base", str(base.medgemma_base))
    st.caption("Inference runs through the existing Slurm GPU jobs. Run one analysis at a time.")

left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Input image")
    source = st.radio("Image source", ["Delta file", "Upload image"], horizontal=True)
    filename = st.text_input("Image filename", str(defaults["IMAGE_FILENAME"]), disabled=source != "Delta file")
    upload = st.file_uploader("TIFF · PNG · JPEG", type=["tif", "tiff", "png", "jpg", "jpeg"], disabled=source != "Upload image")
    image = None
    try:
        if source == "Upload image" and upload:
            image = save_upload(CONNECT, upload.name, upload.getvalue())
        elif source == "Delta file":
            if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("Enter a filename without a folder path.")
            image = (Path(image_root).expanduser() / filename).resolve(strict=True)
        if image:
            with Image.open(image) as opened:
                preview = opened.convert("RGB")
                preview.thumbnail((800, 800))
            st.image(preview, caption=image.name, use_container_width=True)
    except (OSError, ValueError) as error:
        st.error(str(error))
        image = None
    use_caption = st.checkbox("Use existing PathGen caption", value=bool(defaults["EXISTING_CAPTION"]))
    caption_text = st.text_input("Caption JSON path", defaults["EXISTING_CAPTION"] or "", disabled=not use_caption)
    if use_caption:
        st.caption("Uncheck this for a different image. Captions that do not match the image are rejected.")
    submitted = st.button("Run / resume analysis", type="primary", use_container_width=True, disabled=image is None)
    with st.expander("Run or retry individual stages"):
        stage_choice = st.selectbox("Run through stage", ["pathgen", "medgemma", "musk"])
        partial = st.button("Run through selected stage", disabled=image is None)
        st.caption("Earlier stages reuse valid saved results. Completed stages are not recomputed.")

settings = Settings(CONNECT, image_root=image.parent if image else Path(image_root),
    checkpoint=Path(checkpoint).expanduser(), pathgen_model=Path(pathgen_model).expanduser(),
    pathgen_python=Path(pathgen_python).expanduser(), medgemma_python=Path(medgemma_python).expanduser(),
    medgemma_base=medgemma_base)
# Result visibility follows all inputs; never show a previous image's result as current.
identity = json.dumps({"image": str(image), "stamp": (image.stat().st_size, image.stat().st_mtime_ns) if image else None,
    "settings": asdict(settings), "caption": caption_text if use_caption else None}, default=str, sort_keys=True)
if st.session_state.get("identity") != identity:
    st.session_state.update(identity=identity, result=None, caption=None, explanation=None, prediction=None, error=None)

with right:
    st.subheader("Analysis results")
    if submitted or partial:
        session = None
        st.session_state.result = None
        try:
            session = ReportSession(image.name, settings)
            imported = None
            if use_caption:
                if not caption_text.strip():
                    raise ValueError("Enter the caption JSON path.")
                imported = Path(caption_text).expanduser()
                if not imported.is_absolute():
                    imported = CONNECT / imported
            st.session_state.error = None
            with st.status("Checking saved results / waiting for GPU jobs", expanded=True) as status:
                def announce(stage):
                    st.write({"pathgen": "1. PathGen · Image caption", "medgemma": "2. MedGemma · Simplified explanation", "musk": "3. MUSK · Morphology classification and report"}[stage])
                result = run_stages(session, stage_choice if partial else "musk", imported, announce)
                if not partial or stage_choice == "musk":
                    st.session_state.result = result
                status.update(label="Complete", state="complete", expanded=False)
        except Exception as error:
            st.session_state.error = str(error)
        finally:
            if session:
                st.session_state.caption = session.caption
                st.session_state.explanation = session.explanation
                st.session_state.prediction = session.prediction
    if st.session_state.error:
        st.error("Analysis stopped. Completed stages are saved. Resolve the error, then resume.")
        with st.expander("Error and job logs", expanded=True):
            st.code(st.session_state.error, language=None)
    explanation = st.session_state.explanation
    if explanation:
        st.markdown("#### Image explanation · MedGemma")
        st.write(explanation["simplified_caption"])
        st.caption("Explanation text is shown as generated by the model.")
    if st.session_state.prediction:
        st.markdown("#### Morphological features · MUSK")
        st.dataframe(prediction_rows(st.session_state.prediction), hide_index=True, use_container_width=True,
            column_config={"Selected-class probability": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Probability is the model output for the selected class.")
    if st.session_state.caption:
        with st.expander("Original PathGen caption"):
            st.write(st.session_state.caption["caption"])
    result = st.session_state.result
    if result:
        a, b = st.columns(2)
        a.download_button("Download report TXT", result["report"], file_name="ashby-report.txt", mime="text/plain")
        b.download_button("Download full results JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name="ashby-report.json", mime="application/json")
        with st.expander("Run details"):
            st.write({"Image": str(image), "Report": result.get("report_path"), "MUSK checkpoint": checkpoint})
    if not st.session_state.caption and not st.session_state.error:
        st.info("Select an image and run the analysis to see its explanation and five classifications here.")
