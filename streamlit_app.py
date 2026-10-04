import os
import tempfile

import streamlit as st

import detector

st.set_page_config(page_title="Fake Voice Detector")
st.title("Fake Voice Detector")
st.write("Upload or record a few seconds of speech. The model estimates whether the voice "
         "is a real human or AI-generated.")
st.caption("A student project. It can be wrong, especially on voice generators it has not "
           "seen (about 11-13% error in testing). Clips are processed and then deleted.")

uploaded = st.file_uploader("Upload a voice clip",
                            type=["wav", "mp3", "ogg", "opus", "m4a", "flac", "aac", "webm"])
recorded = st.audio_input("Or record one")
clip = uploaded or recorded

if clip is not None:
    st.audio(clip)
    suffix = os.path.splitext(clip.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(clip.getvalue())
    try:
        p = detector.predict(tmp.name)
    except Exception:
        st.error("Could not read this file. Try a different audio format.")
    else:
        fake = float(p.mean())
        if fake > 0.5:
            st.error(f"Verdict: AI-generated ({fake:.0%} probability fake)")
        else:
            st.success(f"Verdict: real human ({fake:.0%} probability fake)")
        st.progress(fake, text="Probability the voice is AI-generated")
        st.caption(f"Scored {len(p)} four-second window(s): " + ", ".join(f"{x:.0%}" for x in p))
    finally:
        os.unlink(tmp.name)