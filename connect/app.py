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
    st.header("실행 설정")
    st.caption("마지막 노트북과 동일한 실행 코드·저장 결과를 사용합니다.")
    image_root = st.text_input("Delta 이미지 폴더", str(base.image_root))
    with st.expander("모델 및 Python 경로"):
        checkpoint = st.text_input("MUSK checkpoint", str(base.checkpoint))
        pathgen_model = st.text_input("PathGen model", str(base.pathgen_model))
        pathgen_python = st.text_input("PathGen Python", str(base.pathgen_python))
        medgemma_python = st.text_input("MUSK / MedGemma Python", str(base.medgemma_python))
        medgemma_base = st.text_input("MedGemma base", str(base.medgemma_base))
    st.caption("모델 추론은 기존 Slurm GPU 작업으로 실행됩니다. 한 번에 한 분석을 실행하세요.")

left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("분석할 이미지")
    source = st.radio("입력 방식", ["Delta 파일", "이미지 업로드"], horizontal=True)
    filename = st.text_input("이미지 파일명", str(defaults["IMAGE_FILENAME"]), disabled=source != "Delta 파일")
    upload = st.file_uploader("TIFF · PNG · JPEG", type=["tif", "tiff", "png", "jpg", "jpeg"], disabled=source != "이미지 업로드")
    image = None
    try:
        if source == "이미지 업로드" and upload:
            image = save_upload(CONNECT, upload.name, upload.getvalue())
        elif source == "Delta 파일":
            if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("폴더 경로 없이 파일명만 입력하세요.")
            image = (Path(image_root).expanduser() / filename).resolve(strict=True)
        if image:
            with Image.open(image) as opened:
                preview = opened.convert("RGB")
                preview.thumbnail((800, 800))
            st.image(preview, caption=image.name, use_container_width=True)
    except (OSError, ValueError) as error:
        st.error(str(error))
        image = None
    use_caption = st.checkbox("기존 PathGen 캡션 사용", value=bool(defaults["EXISTING_CAPTION"]))
    caption_text = st.text_input("캡션 JSON 경로", defaults["EXISTING_CAPTION"] or "", disabled=not use_caption)
    if use_caption:
        st.caption("다른 이미지를 선택했다면 체크를 해제하세요. 이미지가 맞지 않는 캡션은 실행 코드가 거부합니다.")
    submitted = st.button("분석 실행 / 이어서 실행", type="primary", use_container_width=True, disabled=image is None)
    with st.expander("단계별 실행 / 재시도"):
        stage_choice = st.selectbox("여기까지 실행", ["pathgen", "medgemma", "musk"])
        partial = st.button("선택 단계까지 실행", disabled=image is None)
        st.caption("앞 단계는 저장 결과가 있으면 재사용합니다. 성공한 단계를 처음부터 다시 계산하지 않습니다.")

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
    st.subheader("분석 결과")
    if submitted or partial:
        session = None
        st.session_state.result = None
        try:
            session = ReportSession(image.name, settings)
            imported = None
            if use_caption:
                if not caption_text.strip():
                    raise ValueError("캡션 JSON 경로를 입력하세요.")
                imported = Path(caption_text).expanduser()
                if not imported.is_absolute():
                    imported = CONNECT / imported
            st.session_state.error = None
            with st.status("저장 결과 확인 / GPU 작업 대기", expanded=True) as status:
                def announce(stage):
                    st.write({"pathgen": "① PathGen · 이미지 설명", "medgemma": "② MedGemma · 쉬운 설명", "musk": "③ MUSK · 형태 분류 및 보고서"}[stage])
                result = run_stages(session, stage_choice if partial else "musk", imported, announce)
                if not partial or stage_choice == "musk":
                    st.session_state.result = result
                status.update(label="완료", state="complete", expanded=False)
        except Exception as error:
            st.session_state.error = str(error)
        finally:
            if session:
                st.session_state.caption = session.caption
                st.session_state.explanation = session.explanation
                st.session_state.prediction = session.prediction
    if st.session_state.error:
        st.error("분석이 중단되었습니다. 성공한 단계는 저장되어 있으며 수정 후 이어서 실행할 수 있습니다.")
        with st.expander("오류 및 작업 로그", expanded=True):
            st.code(st.session_state.error, language=None)
    explanation = st.session_state.explanation
    if explanation:
        st.markdown("#### 이미지 설명 · MedGemma")
        st.write(explanation["simplified_caption"])
        st.caption("설명은 모델 출력 원문입니다.")
    if st.session_state.prediction:
        st.markdown("#### 형태학적 특징 · MUSK")
        st.dataframe(prediction_rows(st.session_state.prediction), hide_index=True, use_container_width=True,
            column_config={"선택 클래스 확률": st.column_config.NumberColumn(format="%.3f")})
        st.caption("확률은 모델이 선택한 클래스의 출력값입니다.")
    if st.session_state.caption:
        with st.expander("PathGen 원문"):
            st.write(st.session_state.caption["caption"])
    result = st.session_state.result
    if result:
        a, b = st.columns(2)
        a.download_button("보고서 TXT 다운로드", result["report"], file_name="ashby-report.txt", mime="text/plain")
        b.download_button("전체 결과 JSON 다운로드", json.dumps(result, ensure_ascii=False, indent=2), file_name="ashby-report.json", mime="application/json")
        with st.expander("실행 정보"):
            st.write({"이미지": str(image), "보고서": result.get("report_path"), "MUSK checkpoint": checkpoint})
    if not st.session_state.caption and not st.session_state.error:
        st.info("이미지를 선택하고 분석을 실행하면 설명과 5개 분류 결과가 여기에 표시됩니다.")
