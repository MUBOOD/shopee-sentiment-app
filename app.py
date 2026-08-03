import streamlit as st
import pandas as pd

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shopee Sentiment Analyzer AI", layout="wide")

st.title("🛒 Shopee Product Review Sentiment Analyzer")
st.write("ระบบวิเคราะห์ความคิดเห็นรีวิวสินค้า Shopee (รองรับทั้งแบบมีข้อความและให้ดาวอย่างเดียว)")

# --- 1. ฟังก์ชันจัดการข้อมูล ( Data Normalization ) ---
def process_shopee_dataset(df):
    clean_df = pd.DataFrame()

    # ดึงคอลัมน์พื้นฐาน
    clean_df['review_date'] = df['review_date'] if 'review_date' in df.columns else "N/A"
    clean_df['product_name'] = df['product_name'] if 'product_name' in df.columns else "N/A"
    clean_df['variation'] = df['variation'] if 'variation' in df.columns else "N/A"
    
    # แปลง Rating เป็นตัวเลข
    if 'rating' in df.columns:
        clean_df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        clean_df['rating'] = None

    # จัดการข้อความรีวิว: ถ้าเป็น NaN, ว่างเปล่า หรือมีแค่ช่องว่าง ให้ระบุชัดเจน
    if 'review_text' in df.columns:
        clean_df['review_text'] = df['review_text'].apply(
            lambda x: "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)" if pd.isna(x) or str(x).strip() == "" else str(x).strip()
        )
    else:
        clean_df['review_text'] = "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)"

    return clean_df

# --- 2. ฟังก์ชันประเมิน Sentiment (อิง Rating เป็นหลัก) ---
def evaluate_sentiment(row):
    rating = row['rating']
    text = row['review_text']

    # 1. เงื่อนไขหลัก: ถ้ามี Rating ให้ใช้คะแนนดาวตัดสินเป็นหลักทันที (1-2 แย่ / 3 กลาง / 4-5 ดี)
    if pd.notna(rating):
        r = int(rating)
        if r in [1, 2]:
            return 'Negative (แย่ / ด้านลบ)'
        elif r == 3:
            return 'Neutral (ปานกลาง / เป็นกลาง)'
        elif r in [4, 5]:
            return 'Positive (ดี / ด้านบวก)'

    # 2. เงื่อนไขสำรอง: กรณีไม่มี Rating จริงๆ (เป็น NaN) ถึงค่อยใช้อ่านข้อความ
    if text != "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)":
        neg_words = ['ชำรุด', 'พัง', 'ช้า', 'แย่', 'ห่วย', 'เสีย', 'ปลอม', 'ไม่ตรง', 'ผิด', 'ปัญหา']
        pos_words = ['ดี', 'ชอบ', 'ไว', 'เร็ว', 'คุ้ม', 'ลื่น', 'ตรงปก', 'ประทับใจ', 'แน่นหนา', 'แท้']

        pos_score = sum(1 for w in pos_words if w in text)
        neg_score = sum(1 for w in neg_words if w in text)

        if neg_score > pos_score:
            return 'Negative (แย่ / ด้านลบ)'
        elif pos_score > neg_score:
            return 'Positive (ดี / ด้านบวก)'

    return 'Neutral (ปานกลาง / เป็นกลาง)'

# --- 3. ส่วนแสดงผล UI Streamlit ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV รีวิวสินค้า Shopee", type=['csv'])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.success("อัปโหลดไฟล์สำเร็จแล้ว!")

    with st.spinner('กำลังประมวลผลข้อมูล...'):
        std_df = process_shopee_dataset(raw_df)
        std_df['sentiment_result'] = std_df.apply(evaluate_sentiment, axis=1)

    # 📊 Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    pos_count = (std_df['sentiment_result'] == 'Positive (ดี / ด้านบวก)').sum()
    neg_count = (std_df['sentiment_result'] == 'Negative (แย่ / ด้านลบ)').sum()
    neu_count = (std_df['sentiment_result'] == 'Neutral (ปานกลาง / เป็นกลาง)').sum()

    col1.metric("รีวิวดี (4-5 ดาว) 😀", f"{pos_count} รายการ")
    col2.metric("รีวิวแย่ (1-2 ดาว) 😡", f"{neg_count} รายการ")
    col3.metric("รีวิวกลางๆ (3 ดาว) 😐", f"{neu_count} รายการ")

    # 📈 กราฟ
    st.subheader("📊 กราฟสรุปสัดส่วน Sentiment")
    st.bar_chart(std_df['sentiment_result'].value_counts())

    # 📋 ตาราง
    st.subheader("📋 ตารางแสดงผลลัพธ์การวิเคราะห์")
    st.dataframe(std_df[['review_date', 'product_name', 'variation', 'rating', 'review_text', 'sentiment_result']], use_container_width=True)

    # 📥 ดาวน์โหลด CSV
    csv = std_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="📥 ดาวน์โหลดตารางสรุปผลลัพธ์ (CSV)",
        data=csv,
        file_name='sentiment_results.csv',
        mime='text/csv',
    )