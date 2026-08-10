import json
import pandas as pd
import streamlit as st
import altair as alt
from google import genai

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shopee Sentiment Analyzer AI", layout="wide")

st.title("🛒 Shopee Product Review Sentiment Analyzer")
st.write("ระบบวิเคราะห์ความคิดเห็นรีวิวสินค้า Shopee (รองรับทั้งแบบมีข้อความและให้ดาวอย่างเดียว)")

# --- Sidebar สำหรับตั้งค่า AI ---
st.sidebar.header("⚙️ การตั้งค่า AI")

# 1. ประกาศสร้างตัวแปรสำหรับรับค่าจากช่องกรอก
api_key_input = st.sidebar.text_input(
    "ใส่ Gemini API Key (ถ้ามี)",
    type="password",
    help="หากเจ้าของระบบตั้งค่า Key ไว้แล้ว ไม่จำเป็นต้องกรอกช่องนี้"
)

# 2. เช็กว่าใน Secrets ของ Streamlit มี Key อยู่ไหม
has_secret_key = "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"] != ""

# 3. กำหนดค่า api_key ที่จะนำไปใช้งาน
if api_key_input:
    api_key = api_key_input
elif has_secret_key:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.caption("🟢 เปิดใช้งานระบบ AI อัตโนมัติเรียบร้อย")
else:
    api_key = None

# --- 0. แม็บชื่อคอลัมน์ รองรับทั้งภาษาไทย/อังกฤษ ---
COLUMN_MAP = {
    'rating': ['rating', 'จำนวนดาว'],
    'review_text': ['review_text', 'ข้อความรีวิว'],
    'review_date': ['review_date', 'วันที่รีวิว'],
    'product_name': ['product_name', 'ชื่อสินค้า'],
    'variation': ['variation', 'ตัวเลือกสินค้า'],
}

def find_column(df, candidates):
    """หาว่าคอลัมน์ไหนใน df ตรงกับชื่อที่เป็นไปได้ (ไทย/อังกฤษ) คืนชื่อคอลัมน์แรกที่เจอ หรือ None"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- 1. ฟังก์ชันจัดการข้อมูล (Data Normalization) ---
@st.cache_data
def process_shopee_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = pd.DataFrame()

    for target_col, candidates in COLUMN_MAP.items():
        src_col = find_column(df, candidates)
        clean_df[target_col] = df[src_col] if src_col else "N/A"

    # แปลง Rating เป็นตัวเลข (ค่าที่แปลงไม่ได้จะกลายเป็น NaN)
    clean_df['rating'] = pd.to_numeric(clean_df['rating'], errors='coerce')

    # จัดการข้อความรีวิว
    clean_df['review_text'] = clean_df['review_text'].apply(
        lambda x: "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)"
        if pd.isna(x) or str(x).strip() == "" or str(x).strip() == "N/A"
        else str(x).strip()
    )

    return clean_df

# --- 2. ฟังก์ชันประเมิน Sentiment (อิง Rating เป็นหลัก) ---
NEG_WORDS = [
    'ชำรุด', 'พัง', 'ช้า', 'แย่', 'ห่วย', 'เสีย', 'ปลอม', 'ไม่ตรง', 'ผิด', 'ปัญหา',
    'ผิดหวัง', 'ไม่แนะนำ', 'คุณภาพแย่', 'บาง', 'ขาด', 'รั่ว', 'เหม็น', 'สกปรก',
    'ไม่คุ้ม', 'หลอกลวง', 'ของปลอม', 'ไม่พอใจ', 'แกะไม่ได้', 'ส่งช้า', 'บริการแย่',
]
POS_WORDS = [
    'ดี', 'ชอบ', 'ไว', 'เร็ว', 'คุ้ม', 'ลื่น', 'ตรงปก', 'ประทับใจ', 'แน่นหนา', 'แท้',
    'รวดเร็ว', 'บริการดี', 'คุณภาพดี', 'สวย', 'พอใจ', 'แนะนำ', 'ครบ', 'ดีมาก',
    'ประทับใจมาก', 'ส่งไว', 'แพ็คดี', 'คุ้มค่า', 'น่ารัก',
]

def evaluate_sentiment(row) -> str:
    rating = row['rating']
    text = row['review_text']

    if pd.notna(rating):
        r = round(rating)
        if r <= 2:
            return 'Negative (แย่ / ด้านลบ)'
        elif r == 3:
            return 'Neutral (ปานกลาง / เป็นกลาง)'
        else:
            return 'Positive (ดี / ด้านบวก)'

    if text != "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)":
        pos_score = sum(1 for w in POS_WORDS if w in text)
        neg_score = sum(1 for w in NEG_WORDS if w in text)

        if neg_score > pos_score:
            return 'Negative (แย่ / ด้านลบ)'
        elif pos_score > neg_score:
            return 'Positive (ดี / ด้านบวก)'

    return 'Neutral (ปานกลาง / เป็นกลาง)'

# --- 3. ฟังก์ชันเรียกใช้ Gemini API สรุป Pros & Cons ---
def analyze_pros_cons_with_ai(df: pd.DataFrame, api_key: str):
    if not api_key:
        st.warning("⚠️ กรุณาใส่ Gemini API Key ในเมนูด้านซ้ายเพื่อเปิดใช้งานระบบสรุปจุดดี-จุดด้อยด้วย AI")
        return None

    # คัดเลือกเฉพาะรีวิวที่มีข้อความจริง
    text_reviews = df[df['review_text'] != "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)"]['review_text'].tolist()

    if not text_reviews:
        st.info("ไม่พบข้อความรีวิวเพียงพอสำหรับวิเคราะห์ Pros & Cons")
        return None

    # สุ่มดึงสูงสุด 80 รีวิวเพื่อประมวลผลได้รวดเร็ว
    sample_text = "\n".join([f"- {txt}" for txt in text_reviews[:80]])

    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์รีวิวสินค้า 
    จากข้อความรีวิวต่อไปนี้ จงสรุปจุดเด่น (Pros) 3 ข้อ และจุดด้อย (Cons) 3 ข้อ ที่ผู้ซื้อพูดถึงมากที่สุด 
    พร้อมประเมินสัดส่วนเป็นเปอร์เซ็นต์ (%) ของคนที่พูดถึงเรื่องนั้นๆ 

    รายการรีวิวสินค้า:
    {sample_text}

    **ข้อกำหนดในการตอบ:** ให้ตอบกลับมาในรูปแบบ JSON ตามโครงสร้างนี้เท่านั้น ห้ามมีข้อความอื่นนอกเหนือจาก JSON:
    {{
      "pros": [
        {{"topic": "ข้อดีเรื่องที่ 1", "pct": 85}},
        {{"topic": "ข้อดีเรื่องที่ 2", "pct": 70}},
        {{"topic": "ข้อดีเรื่องที่ 3", "pct": 50}}
      ],
      "cons": [
        {{"topic": "ข้อเสียเรื่องที่ 1", "pct": 15}},
        {{"topic": "ข้อเสียเรื่องที่ 2", "pct": 10}},
        {{"topic": "ข้อเสียเรื่องที่ 3", "pct": 5}}
      ]
    }}
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # คลีนผลลัพธ์ข้อความที่ได้ เพื่อให้อยู่ในรูป JSON ที่ถูกต้อง
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini AI: {e}")
        return None

# --- 4. ส่วนแสดงผล UI Streamlit ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV รีวิวสินค้า Shopee", type=['csv'])

if uploaded_file is not None:
    raw_df = None
    for enc in ['utf-8-sig', 'utf-8', 'cp874', 'tis-620']:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if raw_df is None:
        st.error("ไม่สามารถอ่านไฟล์ได้ กรุณาตรวจสอบว่าเป็นไฟล์ CSV ที่ถูกต้อง")
        st.stop()

    if raw_df.empty:
        st.warning("ไฟล์ที่อัปโหลดไม่มีข้อมูล")
        st.stop()

    st.success(f"อัปโหลดไฟล์สำเร็จแล้ว! พบข้อมูล {len(raw_df)} แถว")

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

    # 💡 ส่วนแสดงผล AI Pros & Cons Summary
    st.markdown("---")
    st.subheader("💡 สรุปจุดดี-จุดด้อย (วิเคราะห์ด้วย AI)")

    if api_key:
        with st.spinner("🤖 AI กำลังประมวลผล..."):
            ai_data = analyze_pros_cons_with_ai(std_df, api_key)

        if ai_data:
            col_pros, col_cons = st.columns(2)
            
            with col_pros:
                st.markdown("#### 🟢 ข้อดีที่คนพูดถึงเยอะที่สุด")
                for item in ai_data.get("pros", []):
                    pct = min(max(int(item.get('pct', 0)), 0), 100)
                    st.container(border=True).markdown(f"**{item.get('topic')}**\n*ผู้พูดถึง: {pct}%*")
                    st.progress(pct / 100.0)

            with col_cons:
                st.markdown("#### 🔴 ข้อเสียที่ต้องระวัง / จุดควรปรับปรุง")
                for item in ai_data.get("cons", []):
                    pct = min(max(int(item.get('pct', 0)), 0), 100)
                    st.container(border=True).markdown(f"**{item.get('topic')}**\n*ผู้พูดถึง: {pct}%*")
                    st.progress(pct / 100.0)
    else:
        st.info("👉 กรุณาใส่ Gemini API Key ที่ Sidebar ด้านซ้าย เพื่อเปิดการใช้งานส่วนสรุป AI")

    st.markdown("---")

    # 📈 กราฟ สรุปสัดส่วน Sentiment
    st.subheader("📊 กราฟสรุปสัดส่วน Sentiment")
    st.caption("💡 คลิกที่แท่งกราฟเพื่อกรองตารางด้านล่างให้เหลือเฉพาะกลุ่มนั้น")

    sentiment_counts = std_df['sentiment_result'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_result', 'count']

    sentiment_order = ['Negative (แย่ / ด้านลบ)', 'Neutral (ปานกลาง / เป็นกลาง)', 'Positive (ดี / ด้านบวก)']
    color_map = {
        'Negative (แย่ / ด้านลบ)': '#e74c3c',
        'Neutral (ปานกลาง / เป็นกลาง)': '#f1c40f',
        'Positive (ดี / ด้านบวก)': '#2ecc71',
    }

    click_selection = alt.selection_point(name='select', fields=['sentiment_result'])

    chart = (
        alt.Chart(sentiment_counts)
        .mark_bar()
        .encode(
            x=alt.X('sentiment_result:N', title='Sentiment', sort=sentiment_order),
            y=alt.Y('count:Q', title='จำนวนรีวิว'),
            color=alt.Color(
                'sentiment_result:N',
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=None,
            ),
            opacity=alt.condition(click_selection, alt.value(1.0), alt.value(0.4)),
            tooltip=[alt.Tooltip('sentiment_result:N', title='Sentiment'), alt.Tooltip('count:Q', title='จำนวน')],
        )
        .add_params(click_selection)
        .properties(height=350)
    )

    event = st.altair_chart(chart, use_container_width=True, on_select="rerun", key="sentiment_chart")

    # ดึงว่าผู้ใช้คลิกแท่งไหนไว้บ้าง
    selected_sentiments = []
    if event and event.get("selection") and event["selection"].get("select"):
        selected_sentiments = [p["sentiment_result"] for p in event["selection"]["select"]]

    if selected_sentiments:
        filtered_df = std_df[std_df['sentiment_result'].isin(selected_sentiments)]
        st.info(f"🔍 กำลังกรองเฉพาะ: **{', '.join(selected_sentiments)}** ({len(filtered_df)} รายการ) — คลิกแท่งเดิมอีกครั้งเพื่อยกเลิกกรอง")
    else:
        filtered_df = std_df

    # 📋 ตารางแสดงผล
    st.subheader("📋 ตารางแสดงผลลัพธ์การวิเคราะห์")
    st.dataframe(
        filtered_df[['review_date', 'product_name', 'variation', 'rating', 'review_text', 'sentiment_result']],
        use_container_width=True,
    )

    # 📥 ดาวน์โหลด CSV
    csv = filtered_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    download_label = (
        f"📥 ดาวน์โหลดผลลัพธ์ที่กรองแล้ว ({', '.join(selected_sentiments)}) (CSV)"
        if selected_sentiments
        else "📥 ดาวน์โหลดตารางสรุปผลลัพธ์ทั้งหมด (CSV)"
    )
    st.download_button(
        label=download_label,
        data=csv,
        file_name='sentiment_results.csv',
        mime='text/csv',
    )
else:
    st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์")