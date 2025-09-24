# product_intel/ui.py
# -------------------
# Gradio UI that mirrors your Streamlit tabs as three "modes".
# Buttons pre-fill the success criteria so the worker selects the correct prompt.
#
# - "Competitor Analysis" → competitor mode
# - "Market Sentiment"    → sentiment mode
# - "Launch Metrics"      → metrics mode
#
# The chat lets you paste a company or product and follow up naturally.

import gradio as gr
from typing import List, Dict, Tuple, Any
from graph import Sidekick


# --- Helpers wired to Gradio events ---

async def setup() -> Sidekick:
    sk = Sidekick()
    await sk.setup()
    return sk


async def process_message(sidekick: Sidekick, message: str, success_criteria: str, history: List[Dict[str, str]]):
    if not message and not success_criteria:
        return history, sidekick

    new_history = await sidekick.run_step(message or "", success_criteria or "", history or [])
    return new_history, sidekick


async def set_mode_and_go(sidekick: Sidekick, mode_label: str, prompt: str, history: List[Dict[str, str]]):
    """
    Set a structured success_criteria string based on chosen mode, then run.
    Users can still edit the textbox to refine goals later.
    """
    mode_map = {
        "competitor": "competitor analysis: produce positioning, strengths, weaknesses, takeaways with sources.",
        "sentiment": "sentiment analysis: produce positive vs negative themes, short summary with sources.",
        "metrics": "metrics analysis: produce KPI table, qualitative signals, implications with sources.",
    }
    sc = mode_map.get(mode_label, "competitor analysis")
    new_history = await sidekick.run_step(prompt or "", sc, history or [])
    return new_history, sc, sidekick


async def reset():
    sk = Sidekick()
    await sk.setup()
    return [], "", "", sk


def free_resources(sidekick: Sidekick):
    # Called by Gradio on delete; ensures Playwright is closed.
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Cleanup error: {e}")


# --- Build the UI ---

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Product Launch Intelligence (LangGraph)", theme=gr.themes.Default(primary_hue="emerald")) as ui:
        gr.Markdown("## 🚀 Product Launch Intelligence — LangGraph Sidekick")

        sidekick = gr.State(value=None, delete_callback=free_resources)

        with gr.Row():
            chatbot = gr.Chatbot(label="Sidekick", height=380, type="messages")

        with gr.Group():
            with gr.Row():
                message = gr.Textbox(show_label=False, placeholder="Describe the company/product and what you want.", lines=2)
            with gr.Row():
                success_criteria = gr.Textbox(show_label=False, placeholder="(Optional) e.g., 'competitor analysis' or 'metrics analysis' ...", lines=2)

        with gr.Row():
            competitor_btn = gr.Button("🔍 Competitor Analysis")
            sentiment_btn  = gr.Button("💬 Market Sentiment")
            metrics_btn    = gr.Button("📈 Launch Metrics")
            go_button      = gr.Button("Go!", variant="primary")
            reset_button   = gr.Button("Reset", variant="stop")

        # Wire events
        ui.load(setup, [], [sidekick])

        # Freeform run
        message.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
        success_criteria.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
        go_button.click(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])

        # Mode buttons (prefill success_criteria + run once)
        competitor_btn.click(
            set_mode_and_go,
            [sidekick, gr.State("competitor"), message, chatbot],
            [chatbot, success_criteria, sidekick],
        )
        sentiment_btn.click(
            set_mode_and_go,
            [sidekick, gr.State("sentiment"), message, chatbot],
            [chatbot, success_criteria, sidekick],
        )
        metrics_btn.click(
            set_mode_and_go,
            [sidekick, gr.State("metrics"), message, chatbot],
            [chatbot, success_criteria, sidekick],
        )

        # Reset the entire session
        reset_button.click(reset, [], [chatbot, message, success_criteria, sidekick])

        gr.Markdown(
            "Tip: Paste a company name like **'OpenAI'** and click one of the three analysis buttons. "
            "Follow up in chat to refine scope, e.g., 'focus on Q3 launches' or 'compare to Anthropic'."
        )

    return ui
