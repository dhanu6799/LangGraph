# product_intel/main.py
# ---------------------
# Entry point — just launches the Gradio UI.

from ui import build_ui

if __name__ == "__main__":
    ui = build_ui()
    # inbrowser=True is handy in dev; omit on servers
    ui.launch(inbrowser=True)
