"""Cassava disease demo GUI.

Run with:
    .\venv\Scripts\python.exe -m streamlit run app.py
"""

from pathlib import Path
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
import torch
import timm
from PIL import Image

from dataset import CLASS_NAMES, eval_transform
from model_reports import MODELS, REPORT_DIR, comparison_table


ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "outputs" / "ablation" / "ablation_summary.csv"

THAI_NAMES = {
    "cbb": "ใบไหม้ (Cassava Bacterial Blight)",
    "cbs": "ใบจุดสีน้ำตาล (Brown Leaf Spot)",
    "cmd": "ใบด่าง (Cassava Mosaic Disease)",
    "healthy": "ใบปกติ (Healthy)",
}

st.set_page_config(page_title="Cassava Leaf AI", page_icon="🌿", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.prediction-card {border:1px solid #334155;border-radius:18px;padding:20px;
 background:linear-gradient(135deg,rgba(34,197,94,.10),rgba(59,130,246,.06));min-height:240px;}
.model-name {font-size:17px;font-weight:700;margin-bottom:16px;}
.prediction-label {font-size:24px;font-weight:700;color:#4ade80;}
.confidence-value {font-size:32px;font-weight:700;margin-top:12px;}
.card-note {font-size:12px;color:#94a3b8;}
.score-row {display:flex;justify-content:space-between;gap:12px;border-top:1px solid #334155;
 margin-top:16px;padding-top:12px;font-size:14px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    CHECKPOINT = MODELS[model_name]["checkpoint"]
    inference_checkpoint = CHECKPOINT.with_name(CHECKPOINT.stem.replace("_best", "_inference") + ".pt")
    if inference_checkpoint.exists():
        CHECKPOINT = inference_checkpoint
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"ไม่พบ checkpoint: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = timm.create_model(
        checkpoint.get("model_name", "efficientnet_b0"),
        pretrained=False,
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, device, checkpoint


def predict(image: Image.Image, model_name):
    model, class_names, device, checkpoint = load_model(model_name)
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu()
    pred_idx = int(probabilities.argmax())
    return class_names[pred_idx], probabilities, class_names, checkpoint, device


def get_test_accuracy(model_name):
    """Return accuracy reconstructed from the saved per-class test metrics."""
    METRICS = MODELS[model_name]["evaluation"] / "test_metrics.csv"
    if not METRICS.exists():
        return None
    metrics = pd.read_csv(METRICS)
    if not {"recall", "support"}.issubset(metrics.columns):
        return None
    total = metrics["support"].sum()
    return float((metrics["recall"] * metrics["support"]).sum() / total) if total else None


def show_metrics(model_name):
    METRICS = MODELS[model_name]["evaluation"] / "test_metrics.csv"
    st.subheader("ผลประเมินโมเดล")
    if METRICS.exists():
        metrics = pd.read_csv(METRICS)
        macro_f1 = metrics["f1"].mean()
        accuracy = get_test_accuracy(model_name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Macro-F1", f"{macro_f1:.4f}")
        c2.metric("Accuracy โดยประมาณ", f"{accuracy * 100:.2f}%" if accuracy is not None else "-")
        c3.metric("จำนวนคลาส", str(len(metrics)))
        display = metrics.copy()
        display["class"] = display["class"].map(lambda x: f"{x} - {THAI_NAMES.get(x, x)}")
        for col in ["precision", "recall", "f1"]:
            display[col] = display[col].map(lambda x: f"{x:.4f}")
        st.dataframe(display, width="stretch", hide_index=True)
    else:
        st.warning("ไม่พบไฟล์ test_metrics.csv")


def show_charts(model_name):
    HISTORY = MODELS[model_name]["history"]
    st.subheader("กราฟการฝึกโมเดล")
    if HISTORY.exists():
        history = pd.read_csv(HISTORY)
        left, right = st.columns(2)
        with left:
            st.caption("Loss")
            st.line_chart(history.set_index("epoch")[['train_loss', 'val_loss']], width="stretch")
        with right:
            st.caption("Accuracy และ Macro-F1")
            st.line_chart(history.set_index("epoch")[['train_accuracy', 'val_accuracy', 'val_macro_f1']], width="stretch")

        f1_cols = [c for c in history.columns if c.endswith("_f1")]
        if f1_cols:
            st.caption("F1 รายคลาสต่อ epoch")
            st.line_chart(history.set_index("epoch")[f1_cols], width="stretch")
    else:
        st.warning("ไม่พบไฟล์ history.csv")

    if model_name == "EfficientNet-B0" and ABLATION.exists():
        st.subheader("การทดลอง Ablation ของ EfficientNet-B0")
        ablation = pd.read_csv(ABLATION)
        view = ablation[["experiment", "best_val_macro_f1", "best_val_accuracy", "best_epoch"]].copy()
        view["best_val_accuracy_pct"] = view["best_val_accuracy"] * 100
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Best validation Macro-F1")
            st.bar_chart(view.set_index("experiment")[["best_val_macro_f1"]], width="stretch")
        with c2:
            st.caption("Best validation Accuracy (%)")
            st.bar_chart(view.set_index("experiment")[["best_val_accuracy_pct"]], width="stretch")
        st.dataframe(view[["experiment", "best_val_macro_f1", "best_val_accuracy_pct", "best_epoch"]], width="stretch", hide_index=True)


def show_matrices(model_name):
    CM = MODELS[model_name]["evaluation"] / "confusion_matrix.png"
    CM_NORMALIZED = MODELS[model_name]["evaluation"] / "confusion_matrix_normalized.png"
    st.subheader("Confusion Matrix")
    c1, c2 = st.columns(2)
    with c1:
        if CM.exists():
            st.image(str(CM), caption="Confusion Matrix - Test Set", width="stretch")
    with c2:
        if CM_NORMALIZED.exists():
            st.image(str(CM_NORMALIZED), caption="Normalized Confusion Matrix", width="stretch")


def show_prediction(name, result):
    label, probabilities, class_names, checkpoint, device = result
    confidence = float(probabilities.max())
    accuracy = get_test_accuracy(name)
    metrics = pd.read_csv(MODELS[name]["evaluation"] / "test_metrics.csv")
    accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "-"
    short_label = THAI_NAMES.get(label, label).split(" (")[0]
    st.markdown(f"""<div class="prediction-card">
      <div class="model-name">{escape(name)}</div>
      <div class="card-note">ผลทำนาย</div>
      <div class="prediction-label">{escape(short_label)} · {escape(label.upper())}</div>
      <div class="confidence-value">{confidence * 100:.2f}%</div>
      <div class="card-note">ความมั่นใจของภาพนี้</div>
      <div class="score-row"><span>Test Accuracy<br><b>{accuracy_text}</b></span>
      <span>Test Macro-F1<br><b>{metrics.f1.mean():.4f}</b></span></div>
    </div>""", unsafe_allow_html=True)


def show_probability_comparison(results):
    rows = []
    for name, result in results.items():
        for label, probability in zip(result[2], result[1]):
            rows.append({"คลาส": THAI_NAMES[label].split(" (")[0], "โมเดล": name,
                         "ความมั่นใจ": float(probability)})
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("คลาส:N", sort=[THAI_NAMES[x].split(" (")[0] for x in CLASS_NAMES], title=None),
        x=alt.X("ความมั่นใจ:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
        yOffset="โมเดล:N",
        color=alt.Color("โมเดล:N", scale=alt.Scale(domain=list(MODELS), range=["#60a5fa", "#34d399"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["โมเดล:N", "คลาส:N", alt.Tooltip("ความมั่นใจ:Q", format=".2%")],
    ).properties(height=220)
    st.altair_chart(chart, width="stretch")


st.title("🌿 Cassava Leaf Disease Classification")
st.caption("EfficientNet-B0 / ResNet18 | 4 classes | Macro-F1 เป็น metric หลัก")

tab_predict, tab_report, tab_compare = st.tabs(["🖼️ ทดสอบรูป", "📊 กราฟและผลประเมิน", "⚖️ เปรียบเทียบโมเดล"])

with tab_predict:
    st.info("อัปโหลดภาพใบมันสำปะหลัง 1 ภาพ เพื่อให้โมเดลทำนาย")
    mode = st.radio("โหมดทดสอบ", ["เลือกโมเดลเดียว", "เปรียบเทียบทั้งสองโมเดล"], horizontal=True)
    selected_model = st.selectbox("โมเดลสำหรับทำนาย", list(MODELS), index=1, disabled=mode == "เปรียบเทียบทั้งสองโมเดล")
    uploaded = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([0.85, 1.35], gap="large")
        with left:
            st.image(image, caption=f"ภาพต้นฉบับ ({image.width} × {image.height})", width="stretch")
        with right:
            st.subheader("ผลการวิเคราะห์")
            names = list(MODELS) if mode == "เปรียบเทียบทั้งสองโมเดล" else [selected_model]
            results = {}
            for name in names:
                with st.spinner(f"กำลังทำนายด้วย {name}..."):
                    results[name] = predict(image, name)
            cards = st.columns(len(names), gap="small")
            for card, name in zip(cards, names):
                with card:
                    show_prediction(name, results[name])
            if len(results) == 2:
                labels = [result[0] for result in results.values()]
                if labels[0] == labels[1]:
                    st.caption("✓ ทั้งสองโมเดลทำนายคลาสเดียวกัน")
                else:
                    st.warning("สองโมเดลให้ผลต่างกัน — ควรตรวจสอบอาการในภาพเพิ่มเติม")
            st.caption("Confidence = ความมั่นใจของภาพนี้ · Accuracy / Macro-F1 = คะแนนรวมจากชุด Test")
            show_probability_comparison(results)

with tab_report:
    report_model = st.selectbox("เลือกโมเดลดูผลประเมิน", list(MODELS), index=1)
    show_metrics(report_model)
    show_charts(report_model)
    show_matrices(report_model)
    curve = REPORT_DIR / f"learning_curve_{MODELS[report_model]['architecture']}.png"
    if curve.exists():
        st.download_button("ดาวน์โหลด Learning Curve (PNG)", curve.read_bytes(), file_name=curve.name, mime="image/png")

with tab_compare:
    st.subheader("ประสิทธิภาพบนชุด Test เดียวกัน")
    try:
        table = comparison_table()
        st.dataframe(table, width="stretch", hide_index=True)
        st.bar_chart(table.set_index("Model")[["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1"]])
        st.caption("คะแนนอยู่ในช่วง 0–1; Macro ให้แต่ละคลาสมีน้ำหนักเท่ากัน")
        st.info("เลือก checkpoint จาก Validation Macro-F1; คะแนน Test ใช้รายงานผล ไม่ใช้เลือก epoch")
        st.download_button("ดาวน์โหลดตารางเปรียบเทียบ (CSV)", table.to_csv(index=False).encode("utf-8-sig"), "model_comparison.csv", "text/csv")
        chart = REPORT_DIR / "model_comparison.png"
        if chart.exists():
            st.download_button("ดาวน์โหลดกราฟเปรียบเทียบ (PNG)", chart.read_bytes(), chart.name, "image/png")
        st.subheader("F1 รายคลาส")
        per_class = pd.DataFrame({name: pd.read_csv(config["evaluation"] / "test_metrics.csv").set_index("class")["f1"] for name, config in MODELS.items()})
        st.dataframe(per_class, width="stretch")
        st.bar_chart(per_class)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
