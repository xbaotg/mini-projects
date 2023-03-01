import streamlit as st
import model

from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Digit Recognizer", layout="wide", initial_sidebar_state="collapsed")
cols = st.columns([1, 2])


with st.container():
    with cols[0]:
        canvas_result = st_canvas(
            stroke_width=37,
            stroke_color="#fff",
            background_color="#000",
            update_streamlit=True,
            height=320,
            width=320,
            drawing_mode="freedraw",
            key="canvas",
        )

        st.write("#### Kết quả tốt nhất, khi vẽ tại trung tâm ảnh")

    with cols[1]:
        if 'init' not in st.session_state:
            model.build()
            progress = {
                x: st.progress(0) for x in range(10)
            }
            st.session_state.init = {"progress": progress}

        progress = st.session_state.init["progress"]


# process callback
if canvas_result.image_data is not None:
    output = model.predict(canvas_result.image_data)

    for k, v in output.items():
        progress[k].progress(v, text=f"{k} - {v*100:.4f}%")
