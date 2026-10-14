"""
Machine Learning Web Application
- Part 1: Medical Insurance Cost Prediction (Regression Model)
- Part 2: Thai Character Classifier (Image Classification Model)

Authors:
1. 6630300866 นาย สกลภัทร ภิญโญ
2. 6730300281 นาย นันทพัฒน์ วรรณลักษณ์
3. 6730300591 นาย ศุภวิชญ์ ไชยานุศักดิ์
4. 6730300779 นาย ธรรศธรณ์ เจียมตน
5. 6730300850 นาย พัฒนเดช มะหิเมือง
"""

from pathlib import Path
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import gradio as gr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cnn_model import ThaiCNNPredictor
from preprocessing import _to_pil_image

# ============================================================
# 1. Load Regression Model (Medical Insurance Cost)
# ============================================================
INSURANCE_MODEL_PATH = BASE_DIR / "models" / "insurance_model.joblib"
if not INSURANCE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Insurance model not found at: {INSURANCE_MODEL_PATH}")

print(f"Loading Regression model from: {INSURANCE_MODEL_PATH}")
insurance_pipeline = joblib.load(INSURANCE_MODEL_PATH)


def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return "น้ำหนักต่ำกว่าเกณฑ์ (Underweight)"
    elif bmi < 25.0:
        return "น้ำหนักปกติ / สุขภาพดี (Normal Weight)"
    elif bmi < 30.0:
        return "น้ำหนักเกินเกณฑ์ (Overweight)"
    else:
        return "ภาวะโรคอ้วน (Obese, เสี่ยงต่อโรคเรื้อรัง)"


def predict_insurance(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": int(age),
        "sex": str(sex),
        "bmi": float(bmi),
        "children": int(children),
        "smoker": str(smoker),
        "region": str(region)
    }])

    pred_thb = insurance_pipeline.predict(input_df)[0]
    pred_thb = max(0.0, float(pred_thb))

    bmi_status = calculate_bmi_category(bmi)
    risk_factors = []

    if smoker == "สูบบุหรี่":
        risk_factors.append("- **มีประวัติสูบบุหรี่:** เป็นปัจจัยหลักที่ทำให้เบี้ยประกันสูงขึ้น 3-4 เท่า")
    else:
        risk_factors.append("- **ไม่สูบบุหรี่:** อัตราเบี้ยประกันอยู่ในเกณฑ์ความเสี่ยงต่ำ")

    if bmi >= 30.0:
        risk_factors.append(f"- **BMI {bmi:.1f} ({bmi_status}):** เพิ่มความเสี่ยงโรคหัวใจและหลอดเลือด")
    else:
        risk_factors.append(f"- **BMI {bmi:.1f} ({bmi_status}):** อยู่ในเกณฑ์ความเสี่ยงปกติ")

    if age >= 50:
        risk_factors.append(f"- **อายุ {age} ปี:** เบี้ยประกันปรับเพิ่มตามความเสี่ยงด้านอายุ")

    risk_summary = "\n\n".join(risk_factors)
    result_header = f"### ค่าเบี้ยประกันสุขภาพรายปีที่คาดการณ์:\n## **{pred_thb:,.2f} บาท**"

    return result_header, bmi_status, risk_summary


# ============================================================
# 2. Load Image Classification Model (Thai Characters)
# ============================================================
IMAGE_MODEL_PATH = BASE_DIR / "models" / "thai_cnn_classifier.pt"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

if not IMAGE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Thai CNN model not found: {IMAGE_MODEL_PATH}")

print(f"Loading Image Classification model from: {IMAGE_MODEL_PATH}")
thai_predictor = ThaiCNNPredictor(model_path=IMAGE_MODEL_PATH)

examples = []
if TEST_IMAGES_DIR.exists():
    examples = sorted(
        [[str(p)] for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


def predict_sketch_or_image(image):
    if image is None:
        raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
    if isinstance(image, dict):
        comp = image.get("composite", image.get("background", None))
        if comp is None:
            raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
        image = comp
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            if np.max(image[:, :, 3]) == 0:
                raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
        elif np.std(image) < 0.5:
            raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
    return thai_predictor.predict(image)


# ============================================================
# 3. Gradio Web Application
# ============================================================
with gr.Blocks(title="Machine Learning Web Application", theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <style>
    .sketch-container {
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        padding: 4px !important;
    }
    .sketch-container canvas {
        cursor: crosshair !important;
    }
    </style>
    """)

    with gr.Tabs() as main_tabs:
        # ==========================================
        # TAB 1: REGRESSION MODEL
        # ==========================================
        with gr.TabItem("1. Regression Model (ทำนายค่าเบี้ยประกันสุขภาพ)"):
            gr.Markdown("### ระบบทำนายค่าใช้จ่ายเบี้ยประกันสุขภาพรายปี (Medical Insurance Cost Prediction)")
            gr.Markdown("ทำนายค่าเบี้ยประกันสุขภาพรายบุคคลจากปัจจัยชีววิทยาและพฤติกรรม ด้วยแบบจำลอง **Gradient Boosting Regressor ($R^2 = 87.92%$)**")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### กรอกข้อมูลส่วนบุคคลของผู้เอาประกัน")
                    reg_age = gr.Slider(minimum=18, maximum=80, value=30, step=1, label="อายุ (ปี)")
                    reg_sex = gr.Radio(choices=["ชาย", "หญิง"], value="ชาย", label="เพศสภาพ")
                    reg_bmi = gr.Slider(minimum=15.0, maximum=55.0, value=25.0, step=0.1, label="ดัชนีมวลกาย (BMI: kg/m²)")
                    reg_children = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="จำนวนบุตร / ผู้อยู่ในความอุปการะ (คน)")
                    reg_smoker = gr.Radio(choices=["ไม่สูบบุหรี่", "สูบบุหรี่"], value="ไม่สูบบุหรี่", label="ประวัติการสูบบุหรี่")
                    reg_region = gr.Dropdown(
                        choices=["ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคใต้"],
                        value="ภาคกลาง",
                        label="ภูมิภาคที่อยู่อาศัยในประเทศไทย"
                    )
                    reg_btn = gr.Button("คำนวณเบี้ยประกันสุขภาพ (Predict)", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("#### ผลการทำนายและวิเคราะห์ความเสี่ยง")
                    reg_out_charges = gr.Markdown(value="*กรุณากดปุ่มเพื่อดูผลการทำนาย*")
                    reg_out_bmi_cat = gr.Textbox(label="การประเมินดัชนีมวลกาย (BMI Category)", interactive=False)
                    reg_out_risk = gr.Markdown(label="ปัจจัยที่มีผลต่ออัตราเบี้ยประกัน")

            reg_btn.click(
                fn=predict_insurance,
                inputs=[reg_age, reg_sex, reg_bmi, reg_children, reg_smoker, reg_region],
                outputs=[reg_out_charges, reg_out_bmi_cat, reg_out_risk]
            )

            gr.Markdown("---")
            gr.Markdown("#### ตัวอย่างข้อมูลนำเข้าสำหรับทดสอบ (Examples Presets)")
            gr.Examples(
                examples=[
                    [19, "หญิง", 22.5, 0, "ไม่สูบบุหรี่", "ภาคกลาง"],
                    [38, "ชาย", 28.0, 2, "ไม่สูบบุหรี่", "ภาคเหนือ"],
                    [45, "ชาย", 35.5, 1, "สูบบุหรี่", "ภาคตะวันออกเฉียงเหนือ"],
                    [60, "หญิง", 26.0, 0, "ไม่สูบบุหรี่", "ภาคใต้"]
                ],
                inputs=[reg_age, reg_sex, reg_bmi, reg_children, reg_smoker, reg_region],
                outputs=[reg_out_charges, reg_out_bmi_cat, reg_out_risk],
                fn=predict_insurance,
                cache_examples=False
            )

        # ==========================================
        # TAB 2: IMAGE CLASSIFICATION MODEL
        # ==========================================
        with gr.TabItem("2. Image Classification Model (จำแนกพยัญชนะไทย ก - ฮ)"):
            gr.Markdown("### ระบบจำแนกพยัญชนะภาษาไทย 44 รูป (Thai Character Classifier)")
            gr.Markdown("จำแนกตัวอักษรไทยทั้งจากลายมือเขียนและภาพถ่าย ด้วยแบบจำลอง **Deep CNN (Test Accuracy 99.14%, Macro F1 0.9912)**")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("วาดเขียนด้วยลายมือ (Drawing Canvas)"):
                            gr.Markdown("**วาดพยัญชนะไทย 1 ตัวอักษรลงในกรอบด้านล่าง:**")
                            img_sketch = gr.Sketchpad(
                                label="กระดานวาดเขียน (Drawing Pad)",
                                type="numpy",
                                canvas_size=(360, 360),
                                height=380,
                                fixed_canvas=True,
                                layers=False,
                                brush=gr.Brush(default_size=18, colors=["#000000"], default_color="#000000", color_mode="fixed"),
                                eraser=gr.Eraser(default_size=28),
                                elem_classes=["sketch-container"]
                            )
                            with gr.Row():
                                img_sketch_btn = gr.Button("วิเคราะห์ภาพที่วาด (Predict Drawing)", variant="primary", scale=2, size="lg")
                                img_sketch_clear_btn = gr.Button("ล้างกระดาน (Clear)", variant="secondary", scale=1, size="lg")
                            gr.Markdown("*คำแนะนำ: วาดพยัญชนะขนาดใหญ่พอดีกรอบ ลากเส้นให้ชัดเจน*")

                        with gr.TabItem("อัปโหลดรูปภาพ (Upload Image)"):
                            img_upload = gr.Image(type="numpy", label="อัปโหลดรูปภาพพยัญชนะไทย")
                            img_upload_btn = gr.Button("วิเคราะห์ภาพที่อัปโหลด (Predict)", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output_label = gr.Label(num_top_classes=5, label="ผลการทำนาย 5 อันดับแรก (Top-5 Predictions)")
                    img_output_preview = gr.Image(type="numpy", label="ภาพหลังทำ Preprocessing (28 × 28 พิกเซล)", height=280)

            img_sketch_btn.click(
                fn=predict_sketch_or_image,
                inputs=img_sketch,
                outputs=[img_output_label, img_output_preview]
            )

            img_sketch_clear_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=img_sketch
            )

            img_upload_btn.click(
                fn=thai_predictor.predict,
                inputs=img_upload,
                outputs=[img_output_label, img_output_preview]
            )

            if examples:
                gr.Markdown("---")
                gr.Examples(
                    examples=examples,
                    inputs=img_upload,
                    label="ชุดภาพตัวอย่างสำหรับทดสอบ (Sample Test Images)"
                )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Gradio web service on 0.0.0.0:{port} ...")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
