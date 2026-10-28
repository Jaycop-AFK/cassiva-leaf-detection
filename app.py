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
from sklearn.metrics import confusion_matrix

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
        chart_data = history.set_index("epoch")
        loss_col, accuracy_col, f1_col = st.columns(3)
        with loss_col:
            st.caption("Loss")
            loss_view = chart_data[["train_loss", "val_loss"]].rename(columns={"train_loss": "Train", "val_loss": "Validation"})
            st.line_chart(loss_view, width="stretch")
            st.dataframe(loss_view.reset_index().round(4), width="stretch", hide_index=True)
        with accuracy_col:
            st.caption("Accuracy")
            accuracy_view = chart_data[["train_accuracy", "val_accuracy"]].rename(columns={"train_accuracy": "Train", "val_accuracy": "Validation"})
            st.line_chart(accuracy_view, width="stretch")
            st.dataframe(accuracy_view.reset_index().round(4), width="stretch", hide_index=True)
        with f1_col:
            st.caption("Validation Macro-F1")
            f1_view = chart_data[["val_macro_f1"]].rename(columns={"val_macro_f1": "Macro-F1"})
            st.line_chart(f1_view, width="stretch")
            st.dataframe(f1_view.reset_index().round(4), width="stretch", hide_index=True)

        f1_cols = [c for c in history.columns if c.endswith("_f1")]
        if f1_cols:
            st.caption("F1 รายคลาสต่อ epoch")
            st.line_chart(chart_data[f1_cols], width="stretch")
            st.dataframe(chart_data[f1_cols].reset_index().round(4), width="stretch", hide_index=True)
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
            ablation_f1 = view.melt(id_vars="experiment", value_vars=["best_val_macro_f1"], var_name="Metric", value_name="Value")
            chart = alt.Chart(ablation_f1).mark_bar().encode(
                x=alt.X("experiment:N", title="Experiment", axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Metric:N", legend=None),
                tooltip=["experiment:N", alt.Tooltip("Value:Q", format=".4f")],
            )
            labels = alt.Chart(ablation_f1).mark_text(dy=-8).encode(x="experiment:N", y="Value:Q", text=alt.Text("Value:Q", format=".4f"))
            st.altair_chart(chart + labels, width="stretch")
        with c2:
            st.caption("Best validation Accuracy (%)")
            ablation_acc = view.melt(id_vars="experiment", value_vars=["best_val_accuracy_pct"], var_name="Metric", value_name="Value")
            chart = alt.Chart(ablation_acc).mark_bar().encode(
                x=alt.X("experiment:N", title="Experiment", axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Value:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Metric:N", legend=None),
                tooltip=["experiment:N", alt.Tooltip("Value:Q", format=".2f")],
            )
            labels = alt.Chart(ablation_acc).mark_text(dy=-8).encode(x="experiment:N", y="Value:Q", text=alt.Text("Value:Q", format=".2f"))
            st.altair_chart(chart + labels, width="stretch")
        st.dataframe(view[["experiment", "best_val_macro_f1", "best_val_accuracy_pct", "best_epoch"]], width="stretch", hide_index=True)


def show_matrices(model_name):
    predictions_path = MODELS[model_name]["evaluation"] / "test_predictions.csv"
    st.subheader("Normalized Confusion Matrix")
    st.caption("อ่านเป็นเปอร์เซ็นต์ตามคลาสจริงในแต่ละแถว · ตัวเลขในวงเล็บคือจำนวนภาพ")
    if not predictions_path.exists():
        st.warning("ไม่พบไฟล์ test_predictions.csv")
        return
    predictions = pd.read_csv(predictions_path)
    labels = list(CLASS_NAMES)
    counts = confusion_matrix(predictions.true_idx, predictions.pred_idx, labels=range(len(labels)))
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / row_totals.clip(min=1)
    matrix = pd.DataFrame(normalized, index=labels, columns=labels).reset_index(names="จริง")
    long = matrix.melt(id_vars="จริง", var_name="ทำนาย", value_name="สัดส่วน")
    count_long = pd.DataFrame(counts, index=labels, columns=labels).reset_index(names="จริง").melt(id_vars="จริง", var_name="ทำนาย", value_name="จำนวน")
    long["จำนวน"] = count_long["จำนวน"]
    long["ข้อความ"] = long.apply(lambda r: f"{r['สัดส่วน']:.1%}\n(n={int(r['จำนวน'])})", axis=1)
    heatmap = alt.Chart(long).mark_rect().encode(
        x=alt.X("ทำนาย:N", title="Predicted class", sort=labels),
        y=alt.Y("จริง:N", title="True class", sort=labels),
        color=alt.Color("สัดส่วน:Q", title="สัดส่วน", scale=alt.Scale(domain=[0, 1], scheme="viridis")),
        tooltip=["จริง:N", "ทำนาย:N", alt.Tooltip("สัดส่วน:Q", format=".2%"), "จำนวน:Q"],
    ).properties(height=420)
    text = alt.Chart(long).mark_text(fontSize=13, color="white", lineBreak="\n").encode(
        x=alt.X("ทำนาย:N", sort=labels), y=alt.Y("จริง:N", sort=labels), text="ข้อความ:N"
    )
    st.altair_chart(heatmap + text, width="stretch")


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
        color=alt.Color("โมเดล:N", scale=alt.Scale(domain=list(MODELS), range=["#60a5fa", "#34d399", "#f59e0b"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["โมเดล:N", "คลาส:N", alt.Tooltip("ความมั่นใจ:Q", format=".2%")],
    ).properties(height=220)
    labels = alt.Chart(pd.DataFrame(rows)).mark_text(dx=5, align="left", color="white").encode(
        y=alt.Y("คลาส:N", sort=[THAI_NAMES[x].split(" (")[0] for x in CLASS_NAMES]),
        x="ความมั่นใจ:Q", yOffset="โมเดล:N", text=alt.Text("ความมั่นใจ:Q", format=".1%"),
    )
    st.altair_chart(chart + labels, width="stretch")


st.title("🌿 Cassava Leaf Disease Classification")
st.caption("EfficientNet-B0 / ResNet18 / MobileNetV3-Large | 4 classes | Macro-F1 เป็น metric หลัก")

tab_predict, tab_report, tab_compare = st.tabs(["🖼️ ทดสอบรูป", "📊 กราฟและผลประเมิน", "⚖️ เปรียบเทียบโมเดล"])

with tab_predict:
    st.info("อัปโหลดภาพใบมันสำปะหลัง 1 ภาพ เพื่อให้โมเดลทำนาย")
    mode = st.radio("โหมดทดสอบ", ["เลือกโมเดลเดียว", "เปรียบเทียบทุกโมเดล"], horizontal=True)
    selected_model = st.selectbox("โมเดลสำหรับทำนาย", list(MODELS), index=1, disabled=mode == "เปรียบเทียบทุกโมเดล")
    uploaded = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([0.85, 1.35], gap="large")
        with left:
            st.image(image, caption=f"ภาพต้นฉบับ ({image.width} × {image.height})", width="stretch")
        with right:
            st.subheader("ผลการวิเคราะห์")
            names = list(MODELS) if mode == "เปรียบเทียบทุกโมเดล" else [selected_model]
            results = {}
            for name in names:
                with st.spinner(f"กำลังทำนายด้วย {name}..."):
                    results[name] = predict(image, name)
            cards = st.columns(len(names), gap="small")
            for card, name in zip(cards, names):
                with card:
                    show_prediction(name, results[name])
            if len(results) >= 2:
                labels = [result[0] for result in results.values()]
                if len(set(labels)) == 1:
                    st.caption("✓ ทุกโมเดลทำนายคลาสเดียวกัน")
                else:
                    st.warning("โมเดลให้ผลต่างกัน — ควรตรวจสอบอาการในภาพเพิ่มเติม")
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
        comparison = table.melt(id_vars="Model", value_vars=["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1"], var_name="Metric", value_name="Score")
        bars = alt.Chart(comparison).mark_bar().encode(
            x=alt.X("Model:N", title="Model"),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")],
        )
        labels = alt.Chart(comparison).mark_text(dy=-8, fontSize=11).encode(
            x="Model:N", xOffset=alt.XOffset("Metric:N"), y="Score:Q", text=alt.Text("Score:Q", format=".4f"), color=alt.value("white")
        )
        st.altair_chart((bars + labels).properties(height=420), width="stretch")
        st.caption("คะแนนอยู่ในช่วง 0–1; Macro ให้แต่ละคลาสมีน้ำหนักเท่ากัน")
        st.info("เลือก checkpoint จาก Validation Macro-F1; คะแนน Test ใช้รายงานผล ไม่ใช้เลือก epoch")
        st.download_button("ดาวน์โหลดตารางเปรียบเทียบ (CSV)", table.to_csv(index=False).encode("utf-8-sig"), "model_comparison.csv", "text/csv")
        chart = REPORT_DIR / "model_comparison.png"
        if chart.exists():
            st.download_button("ดาวน์โหลดกราฟเปรียบเทียบ (PNG)", chart.read_bytes(), chart.name, "image/png")
        st.subheader("F1 รายคลาส")
        metric_rows = []
        for name, config in MODELS.items():
            metrics = pd.read_csv(config["evaluation"] / "test_metrics.csv")
            metrics["Model"] = name
            metric_rows.append(metrics)
        class_metrics = pd.concat(metric_rows, ignore_index=True)
        class_metrics["Class"] = class_metrics["class"].map(lambda x: THAI_NAMES.get(x, x).split(" (")[0])
        class_metrics = class_metrics.rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Test Support"})
        detail = class_metrics[["Model", "Class", "Precision", "Recall", "F1", "Test Support"]].copy()
        st.caption("ตารางรายละเอียดผลของทุกโมเดลและทุกคลาส")
        st.dataframe(
            detail.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
            width="stretch", hide_index=True,
        )

        summary_rows = []
        for class_name, group in detail.groupby("Class", sort=False):
            best = group.loc[group["F1"].idxmax()]
            worst = group.loc[group["F1"].idxmin()]
            summary_rows.append({
                "Class": class_name,
                "Best model": best["Model"], "Best F1": best["F1"],
                "Worst model": worst["Model"], "Worst F1": worst["F1"],
                "Test Support": int(group["Test Support"].iloc[0]),
            })
        summary = pd.DataFrame(summary_rows)
        st.caption("สรุปคลาสที่ทำได้ดีที่สุดและแย่ที่สุด")
        st.dataframe(
            summary.style.format({"Best F1": "{:.4f}", "Worst F1": "{:.4f}"}),
            width="stretch", hide_index=True,
        )

        per_class = detail.pivot(index="Class", columns="Model", values="F1")
        per_class_long = detail[["Class", "Model", "F1"]].copy()
        bars = alt.Chart(per_class_long).mark_bar().encode(
            x=alt.X("Class:N", title="Class"), xOffset=alt.XOffset("Model:N"),
            y=alt.Y("F1:Q", title="F1 score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["class:N", "Model:N", alt.Tooltip("F1:Q", format=".4f")],
        )
        labels = alt.Chart(per_class_long).mark_text(dy=-8, fontSize=10).encode(
            x="Class:N", xOffset=alt.XOffset("Model:N"), y="F1:Q", text=alt.Text("F1:Q", format=".4f"), color=alt.value("white")
        )
        st.altair_chart((bars + labels).properties(height=420), width="stretch")
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
