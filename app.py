import gradio as gr

import detector


def check(path):
    if path is None:
        return None, "Upload or record a clip first."
    try:
        p = detector.predict(path)
    except Exception as e:
        return None, f"Could not read this file: {e}"
    fake = float(p.mean())
    detail = (f"Scored {len(p)} four-second window(s). Probability fake per window: "
              + ", ".join(f"{x:.0%}" for x in p))
    return {"AI-generated": fake, "Real human": 1 - fake}, detail


demo = gr.Interface(
    fn=check,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Voice clip"),
    outputs=[gr.Label(label="Verdict"), gr.Textbox(label="Details")],
    title="Fake Voice Detector",
    description=("Upload or record a few seconds of speech. The model estimates whether the voice "
                 "is a real human or AI-generated. This is a student project and can be wrong, "
                 "especially on voice generators it has not seen."),
)

if __name__ == "__main__":
    demo.launch()