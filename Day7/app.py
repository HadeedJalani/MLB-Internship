import gradio as gr
from utils import (
    load_dataset,
    clean_dataset,
    generate_dashboard,
    generate_charts
)

# -----------------------------
# Functions
# -----------------------------

def preview_dataset():
    return load_dataset()

def clean_data():
    df = clean_dataset()
    return (
        "Dataset cleaned successfully!\n"
        "• Removed duplicate records\n"
        "• Filled missing values\n"
        "• Added Average_Score\n"
        "• Added Performance category",
        df
    )

def charts():

    return generate_charts()    

def dashboard():
    return generate_dashboard()

# -----------------------------
# UI
# -----------------------------

with gr.Blocks(
    title="Student Performance Analytics Dashboard",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
# 🎓 Student Performance Analytics Dashboard

### Analyze, clean, and visualize student performance data.
"""
    )

    with gr.Tab("📂 Dataset"):

        preview_btn = gr.Button("Load Dataset")

        dataset = gr.Dataframe()

        preview_btn.click(
            preview_dataset,
            outputs=dataset
        )

    with gr.Tab("🧹 Data Cleaning"):

        clean_btn = gr.Button("Clean Dataset")

        status = gr.Textbox(label="Status")

        cleaned = gr.Dataframe()

        clean_btn.click(
            clean_data,
            outputs=[status, cleaned]
        )

    with gr.Tab("📊 Dashboard"):

        dashboard_btn = gr.Button("Generate Dashboard")

        report = gr.Textbox(
            lines=30,
            label="Analytics Report"
        )

        dashboard_btn.click(
            dashboard,
            outputs=report
        )

    with gr.Tab("📈 Visualizations"):

        chart_btn = gr.Button("Generate Charts")

        gallery = gr.Gallery(
            label="Charts",
            columns=2,
            height=600
    )

        chart_btn.click(
            charts,
            outputs=gallery
    )

gr.Markdown(
"""
---
### 👨‍💻 Developed by Hadeed Jalani
MLBench Summer Internship – Day 7

Python • Pandas • Matplotlib • Gradio
"""
)

app.launch()