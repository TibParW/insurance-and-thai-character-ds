# Data Science Final Project
### โครงงานพัฒนาแบบจำลองการถดถอย (Regression) และจำแนกประเภทรูปภาพ (Image Classification) พร้อมเว็บแอปพลิเคชัน

---

## ข้อมูลผู้จัดทำ (Authors Information)

> **หมายเหตุ:** เรียงลำดับตามรหัสนิสิตจากน้อยไปมากตามเกณฑ์ข้อกำหนดของรายวิชา

| ลำดับ | รหัสนิสิต | ชื่อ - นามสกุล |
| :---: | :---: | :--- |
| 1 | `6630300866` | นาย สกลภัทร ภิญโญ |
| 2 | `6730300281` | นาย นันทพัฒน์ วรรณลักษณ์ |
| 3 | `6730300591` | นาย ศุภวิชญ์ ไชยานุศักดิ์ |
| 4 | `6730300779` | นาย ธรรศธรณ์ เจียมตน |
| 5 | `6730300850` | นาย พัฒนเดช มะหิเมือง |

---

## ลิงก์ระบบและเว็บแอปพลิเคชัน (Live Application & Repository)

* **Live Web Application (Render):** [https://insurance-and-thai-character-ds.onrender.com](https://insurance-and-thai-character-ds.onrender.com)
* **GitHub Repository:** [https://github.com/TibParW/insurance-and-thai-character-ds](https://github.com/TibParW/insurance-and-thai-character-ds)
* **Google Colab Notebooks (คลิกเปิดรันบนคลาวด์ได้ทันที):**
  * **Regression Model Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TibParW/insurance-and-thai-character-ds/blob/main/notebooks/insurance_cost_regression.ipynb)
  * **Image Classification Model Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TibParW/insurance-and-thai-character-ds/blob/main/notebooks/thai_character_classification.ipynb)

---

## 1. แบบจำลองการถดถอย (Regression Model)

### 1.1 แนวคิด ปัญหา และประโยชน์ในการนำไปใช้ (Concept & Problem Statement)
* **ปัญหาที่ต้องการแก้:** ค่าใช้จ่ายในการรักษาพยาบาลและค่าเบี้ยประกันสุขภาพรายปี (Health Insurance Charges) มีความผันผวนสูงตามปัจจัยสุขภาพและพฤติกรรมบุคคล เช่น การสูบบุหรี่และภาวะน้ำหนักเกิน หากประเมินคลาดเคลื่อนอาจทำให้บริษัทประกันขาดทุนหรือตั้งราคาไม่เป็นธรรมต่อผู้เอาประกัน
* **สิ่งที่ต้องการทำนายและหน่วย:** ค่าใช้จ่ายเบี้ยประกันสุขภาพรายปีรายบุคคล มีหน่วยเป็น **บาท (THB)**
* **ตัวแปรนำเข้า (Input Features: 6 ตัวแปร):**
  1. `age`: อายุของผู้เอาประกัน (18 - 80 ปี)
  2. `sex`: เพศสภาพ (`ชาย`, `หญิง`)
  3. `bmi`: ดัชนีมวลกาย ($kg/m^2$)
  4. `children`: จำนวนบุตรหรือผู้อยู่ในความอุปการะ (0 - 5 คน)
  5. `smoker`: ประวัติการสูบบุหรี่ (`ไม่สูบบุหรี่`, `สูบบุหรี่`)
  6. `region`: ภูมิภาคที่อยู่อาศัยในประเทศไทย (`ภาคกลาง`, `ภาคเหนือ`, `ภาคตะวันออกเฉียงเหนือ`, `ภาคใต้`)
* **ประโยชน์ในการนำไปใช้:**
  * **บริษัทประกันภัย (Underwriting & Risk Pricing):** ใช้ประเมินความเสี่ยงและตั้งราคาเบี้ยประกันสุขภาพที่สะท้อนต้นทุนจริง (Risk-based Pricing)
  * **ประชาชนทั่วไป:** วางแผนงบประมาณสุขภาพในหน่วยเงินบาท และตระหนักถึงผลกระทบทางการเงินของการสูบบุหรี่และภาวะอ้วน

### 1.2 แหล่งที่มาและการแบ่งชุดข้อมูล (Dataset & Splitting)
* **แหล่งที่มา:** ข้อมูลสถิติค่าเบี้ยประกันสุขภาพรายบุคคล อ้างอิงตามเกณฑ์อัตราเบี้ยประกันภัยมาตรฐาน คปภ. (สำนักงานคณะกรรมการกำกับและส่งเสริมการประกอบธุรกิจประกันภัย - OIC New Health Standard) (1,500 แถว, 7 คอลัมน์ ไม่มี Missing Values) บันทึกค่าเงินเป็นบาทและจำแนกตามภูมิภาคของประเทศไทย เก็บไว้ในโฟลเดอร์ [`data/insurance.csv`](data/insurance.csv)
* **การแบ่งชุดข้อมูล (80:20):**
  * Training Set (80%): 1,200 ตัวอย่าง
  * Test Set (20%): 300 ตัวอย่าง (ซ่อนไว้สำหรับวัดผลการทำนายจริง โดยใช้ `random_state=42`)

### 1.3 แบบจำลองที่ใช้และการประเมินผล (Model & Evaluation)
* **การเตรียมข้อมูล (Preprocessing Pipeline):** ตัวเลขผ่าน `StandardScaler`, ตัวแปรกลุ่มผ่าน `OneHotEncoder(drop='first')`
* **ผลการประเมินบนชุดทดสอบ (Test Set: 300 รายการ):**

| แบบจำลอง (Regression Model) | R² Score | RMSE (บาท) | MAE (บาท) | สรุปผล |
| :--- | :---: | :---: | :---: | :--- |
| **Gradient Boosting (โมเดลหลัก)** | **0.9646 (96.5%)** | **2,538.42 บาท** | **2,026.70 บาท** | **แม่นยำสูงสุด ดักจับ Non-linear Interaction ได้ดีเยี่ยม** |
| **Random Forest Regressor** | 0.9457 | 3,146.30 บาท | 2,490.04 บาท | เสถียร ทนทานต่อ Outliers |
| **Linear Regression (Baseline)** | 0.8995 | 4,278.63 บาท | 3,191.64 บาท | ประสิทธิภาพดีสำหรับความสัมพันธ์เชิงเส้น |

* **การแปลความหมายเมตริกและวิเคราะห์ข้อผิดพลาด:**
  * โมเดล Gradient Boosting สามารถอธิบายความแปรปรวนของค่าเบี้ยประกันสุขภาพได้ถึง 96.46%
  * ความคลาดเคลื่อนสัมบูรณ์เฉลี่ย (MAE) อยู่ที่เพียง 2,026.70 บาท จากค่าเบี้ยประกันเฉลี่ย 30,871 บาท
  * ปัจจัยสำคัญที่สุดคือ `age` (ตามเส้นโค้งความเสี่ยงโรค), `smoker` (ความเสี่ยงปอดและมะเร็ง) และ `bmi` (ภาวะอ้วนตามเกณฑ์คนเอเชีย)
* **ข้อจำกัดของแบบจำลอง (Model Limitations):**
  * อิงตามปัจจัยพื้นฐาน 6 ด้านตามเกณฑ์มาตรฐาน แต่ในการพิจารณาประกันจริงอาจมีปัจจัยเชิงลึก เช่น โรคประจำตัวเรื้อรัง (NCDs) หรือประวัติพันธุกรรม
  * เหมาะสำหรับผู้เอาประกันภัยช่วงอายุ 18 - 80 ปี โดยไม่ครอบคลุมกลุ่มประกันสุขภาพเด็กเล็กแรกเกิด
* **สมุดโค้ดสำหรับการทดลอง (Jupyter Notebook):** [`notebooks/insurance_cost_regression.ipynb`](notebooks/insurance_cost_regression.ipynb) หรือเปิดรันบน [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TibParW/insurance-and-thai-character-ds/blob/main/notebooks/insurance_cost_regression.ipynb)

---

## 2. แบบจำลองการจำแนกประเภทรูปภาพ (Image Classification Model)

### 2.1 แนวคิด ปัญหา และประโยชน์ในการนำไปใช้ (Concept & Problem Statement)
* **ปัญหาที่ต้องการแก้:** พยัญชนะไทย 44 รูป (ก ถึง ฮ) มีสัณฐานวิทยาที่ซับซ้อน ลายเส้นมีความคล้ายคลึงกันสูงในหลายคู่ เช่น การมีหยัก (ข กับ ช), ความยาวหาง (บ กับ ป, ผ กับ ฝ), และทิศทางการม้วนหัว (ด กับ ต)
* **สิ่งที่ต้องการจำแนก:** จำแนกรูปภาพพยัญชนะไทยออกเป็น 44 ประเภท (ก - ฮ)
* **ประโยชน์ในการนำไปใช้:**
  * **ระบบแปลงเอกสารสแกนเป็นข้อความดิจิทัล (OCR):** แปลงเอกสารราชการและเวชระเบียน
  * **ระบบอ่านป้ายทะเบียนรถยนต์อัตโนมัติ (ALPR):** ตรวจจับหมวดอักษรบนป้ายทะเบียน
  * **สื่อการเรียนการสอนภาษาไทย:** ให้ผู้เรียนฝึกเขียนพยัญชนะบนหน้าจอ และระบบตรวจทานให้ผลย้อนกลับทันที

### 2.2 แหล่งที่มา จำนวนข้อมูล และการแบ่งชุดข้อมูล
* **แหล่งที่มา:** การผสานรวม (Data Fusion) ระหว่างภาพลายมือคนเขียนจริง + ตัวพิมพ์คอมพิวเตอร์ 14 ฟอนต์มาตรฐาน พร้อม Data Augmentation
* **จำนวนข้อมูล:** 22,000 ภาพ (44 คลาส คลาสละ 500 ภาพ) เก็บไว้ใน [`data/thai_characters/`](data/thai_characters/)
* **การแบ่งชุดข้อมูล (80:20 Stratified Split):**
  * Training Set (80%): 17,600 ภาพ (400 ภาพต่อคลาส)
  * Test Set (20%): 4,400 ภาพ (100 ภาพต่อคลาส)

### 2.3 ขั้นตอนการเตรียมภาพ (Preprocessing Pipeline)
1. **Grayscale Conversion:** แปลงภาพเป็น 1 ช่องสัญญาณ
2. **Foreground/Background Standardization:** ปรับให้เส้นหมึกเป็นสีขาว (>0) และพื้นหลังเป็นสีดำ (=0)
3. **Auto-crop Bounding Box:** ตรวจหากรอบล้อมรอบตัวอักษรจริงเพื่อตัดขอบว่างภายนอกออก
4. **Centering & Resizing:** ปรับขนาดตัวอักษรให้อยู่ในกรอบ 20x20 พิกเซล แล้ววางกึ่งกลางภาพขนาด 28x28 พิกเซล
5. **Pixel Normalization:** ปรับสเกลค่าพิกเซลให้อยู่ในช่วง [0, 1]

### 2.4 แบบจำลองที่ใช้และการประเมินผล (Model & Evaluation)

| แบบจำลอง (Classification Model) | Architecture | Test Accuracy (4,400 ภาพ) | Macro F1-Score | จุดเด่น |
| :--- | :--- | :---: | :---: | :--- |
| **Deep CNN (โมเดลหลัก)** | 3 Conv Blocks + BatchNorm + Dropout | **99.14%** | **0.9912** | **แยกได้ชัดเจน ทั้งตัวพิมพ์และลายมือ** |
| **Random Forest** | `n_estimators=200`, `n_jobs=-1` | 92.64% | 0.9258 | ใช้เป็น Baseline |
| **Support Vector Machine (SVM)** | RBF Kernel, `C=10`, `StandardScaler` | 88.18% | 0.8827 | โมเดลคลาสสิก |

* **การแปลความหมายเมตริก:**
  * **Test Accuracy (99.14%):** ทำนายถูกต้องมากกว่า 4,360 ภาพ จาก 4,400 ภาพทดสอบ
  * **Macro F1-Score (0.9912):** ให้น้ำหนักทุกคลาสเท่าเทียมกัน ยืนยันว่าโมเดลทำนายได้ดีสม่ำเสมอทุกพยัญชนะ
  * **การแยกแยะคู่คล้าย:** `ข` ถูกต้อง 99.84%, `ก` ถูกต้อง 99.99%, `บ` ถูกต้อง 99.99%, `ป` ถูกต้อง 100.00%, `ฮ` ถูกต้อง 100.00%
* **ข้อจำกัดของแบบจำลอง (Model Limitations):**
  * ออกแบบมาสำหรับจำแนกพยัญชนะเดี่ยว 1 ตัวอักษร ยังไม่รองรับสระ วรรณยุกต์ หรือข้อความต่อเนื่องเป็นประโยค
  * การวาดลายมือบนกระดานควรวาดให้อยู่กึ่งกลางกรอบและมีขนาดพอดี หากวาดมีขนาดเล็กเกินไปอาจส่งผลต่อการตรวจจับขอบ Bounding Box
* **สมุดโค้ดสำหรับการทดลอง (Jupyter Notebook):** [`notebooks/thai_character_classification.ipynb`](notebooks/thai_character_classification.ipynb) หรือเปิดรันบน [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TibParW/insurance-and-thai-character-ds/blob/main/notebooks/thai_character_classification.ipynb)

---

## 3. เว็บแอปพลิเคชัน (Web Application)

ระบบถูกออกแบบให้รองรับทั้งสองแบบจำลอง โดยมีแท็บให้เลือกทดสอบใช้งานได้อย่างสะดวก:

### 3.1 แท็บที่ 1: Regression App
* **การใช้งาน:** เลื่อนแถบระบุ อายุ, เพศสภาพ, BMI, จำนวนบุตร, ประวัติการสูบบุหรี่, และภูมิภาคในประเทศไทย แล้วกดปุ่มทำนาย
* **ผลลัพธ์:** แสดงค่าเบี้ยประกันที่คาดการณ์เป็น **บาท (THB)** พร้อมประเมินเกณฑ์ BMI และแจกแจงปัจจัยเสี่ยง
* **ตัวอย่างข้อมูลนำเข้า (Preset Examples):**
  1. *วัยเริ่มทำงาน (อายุ 22, หญิง, BMI 21.5, บุตร 0, ไม่สูบบุหรี่, ภาคกลาง):* ~18,120 บาท
  2. *วัยทำงาน มีครอบครัว (อายุ 38, ชาย, BMI 25.5, บุตร 2, ไม่สูบบุหรี่, ภาคเหนือ):* ~22,720 บาท
  3. *ผู้สูบบุหรี่ + ภาวะอ้วน (อายุ 45, ชาย, BMI 32.0, บุตร 1, สูบบุหรี่, ภาคตะวันออกเฉียงเหนือ):* ~61,840 บาท
  4. *ผู้สูงอายุ สุขภาพปกติ (อายุ 65, หญิง, BMI 23.5, บุตร 0, ไม่สูบบุหรี่, ภาคใต้):* ~32,680 บาท

### 3.2 แท็บที่ 2: Image Classification App
* **การใช้งาน:** รองรับทั้งการอัปโหลดภาพถ่าย และการวาดเขียนด้วยลายมือบนหน้าจอ (Drawing Canvas)
* **ผลลัพธ์:** แสดงผลการทำนายพยัญชนะไทย 5 อันดับแรก (Top-5 Probabilities) พร้อมภาพขนาด 28x28 ที่ผ่านกระบวนการ Preprocessing
* **ตัวอย่างภาพ:** มีปุ่มภาพตัวอย่างด้านล่างให้คลิกทดสอบได้ทันที

---

## 4. ขั้นตอนการติดตั้งและรันระบบบนเครื่อง (Setup & Installation)

```bash
# 1. Clone repository
git clone https://github.com/TibParW/insurance-and-thai-character-ds.git
cd insurance-and-thai-character-ds

# 2. ติดตั้งแพ็กเกจที่จำเป็น
pip install -r requirements.txt

# 3. รันเว็บแอปพลิเคชัน
python app.py
```
เปิดเบราว์เซอร์ไปที่: `http://localhost:7860`

---

## 5. โครงสร้างไฟล์และโฟลเดอร์สำคัญในโครงการ (Repository Structure)

```text
insurance-and-thai-character-ds/
├── data/
│   ├── insurance.csv                       : ข้อมูลเบี้ยประกันสุขภาพ คปภ. (1,500 รายการ)
│   └── thai_characters/                    : ข้อมูลภาพพยัญชนะไทย 44 คลาส (22,000 ภาพ)
├── models/
│   ├── insurance_model.joblib              : โมเดล Gradient Boosting Pipeline (R² = 96.5%)
│   ├── thai_cnn_classifier.pt              : โมเดล Deep CNN PyTorch (Accuracy = 99.1%)
│   └── thai_character_classifier.joblib    : โมเดล Random Forest Baseline
├── notebooks/
│   ├── insurance_cost_regression.ipynb     : สมุดโค้ดวิเคราะห์และสร้างโมเดล Regression
│   └── thai_character_classification.ipynb : สมุดโค้ดวิเคราะห์และสร้างโมเดล Classification
├── src/                                    : ซอร์สโค้ดและโมดูลประมวลผลระบบ
├── fonts/                                  : ฟอนต์ตัวอักษรไทยสำหรับ Data Augmentation
├── test_images/                            : ภาพตัวอย่างสำหรับทดสอบระบบ
├── app.py                                  : ซอร์สโค้ดหลักของ Web Application (Gradio)
├── render.yaml                             : ไฟล์ตั้งค่า Deployment บน Render Cloud
├── requirements.txt                        : รายการไลบรารีที่จำเป็นทั้งหมด
└── README.md                               : เอกสารสรุปโครงการฉบับสมบูรณ์
```
