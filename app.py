import json
import altair as alt
from google import genai
from google.genai import types
import pandas as pd
import streamlit as st
from weasyprint import HTML

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shopee Sentiment Analyzer AI", layout="wide")

st.title("🛒 Shopee Product Review Sentiment Analyzer")
st.write(
    "ระบบวิเคราะห์ความคิดเห็นรีวิวสินค้า Shopee"
    " (รองรับทั้งแบบมีข้อความและให้ดาวอย่างเดียว)"
)

# --- Sidebar สำหรับตั้งค่า AI ---
st.sidebar.header("⚙️ การตั้งค่า AI")

api_key_input = st.sidebar.text_input(
    "ใส่ Gemini API Key (ถ้ามี)",
    type="password",
    help="หากเจ้าของระบบตั้งค่า Key ไว้แล้ว ไม่จำเป็นต้องกรอกช่องนี้",
)

has_secret_key = (
    "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"] != ""
)

if api_key_input:
  api_key = api_key_input
elif has_secret_key:
  api_key = st.secrets["GEMINI_API_KEY"]
  st.sidebar.caption("🟢 เปิดใช้งานระบบ AI อัตโนมัติเรียบร้อย")
else:
  api_key = None

# --- 0. แม็บชื่อคอลัมน์ รองรับทั้งภาษาไทย/อังกฤษ ---
COLUMN_MAP = {
    "rating": ["rating", "จำนวนดาว", "rating_star"],
    "review_text": ["review_text", "ข้อความรีวิว", "comment"],
    "review_date": ["review_date", "วันที่รีวิว", "create_time"],
    "product_name": ["product_name", "ชื่อสินค้า"],
    "variation": ["variation", "ตัวเลือกสินค้า"],
}


def find_column(df, candidates):
  """หาว่าคอลัมน์ไหนใน df ตรงกับชื่อที่เป็นไปได้ คืนชื่อคอลัมน์แรกที่เจอ"""
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

  # แปลง Rating เป็นตัวเลข
  clean_df["rating"] = pd.to_numeric(clean_df["rating"], errors="coerce")

  # จัดการข้อความรีวิว
  clean_df["review_text"] = clean_df["review_text"].apply(
      lambda x: (
          "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)"
          if pd.isna(x) or str(x).strip() in ["", "N/A"]
          else str(x).strip()
      )
  )

  return clean_df


# --- 2. ฟังก์ชันประเมิน Sentiment ---
NEG_WORDS = [
    "ชำรุด",
    "พัง",
    "ช้า",
    "แย่",
    "ห่วย",
    "เสีย",
    "ปลอม",
    "ไม่ตรง",
    "ผิด",
    "ปัญหา",
    "ผิดหวัง",
    "ไม่แนะนำ",
    "คุณภาพแย่",
    "บาง",
    "ขาด",
    "รั่ว",
    "เหม็น",
    "สกปรก",
    "ไม่คุ้ม",
    "หลอกลวง",
    "ของปลอม",
    "ไม่พอใจ",
    "แกะไม่ได้",
    "ส่งช้า",
    "บริการแย่",
]
POS_WORDS = [
    "ดี",
    "ชอบ",
    "ไว",
    "เร็ว",
    "คุ้ม",
    "ลื่น",
    "ตรงปก",
    "ประทับใจ",
    "แน่นหนา",
    "แท้",
    "รวดเร็ว",
    "บริการดี",
    "คุณภาพดี",
    "สวย",
    "พอใจ",
    "แนะนำ",
    "ครบ",
    "ดีมาก",
    "ประทับใจมาก",
    "ส่งไว",
    "แพ็คดี",
    "คุ้มค่า",
    "น่ารัก",
]


def evaluate_sentiment(row) -> str:
  rating = row["rating"]
  text = row["review_text"]

  if pd.notna(rating):
    r = round(rating)
    if r <= 2:
      return "Negative (แย่ / ด้านลบ)"
    elif r == 3:
      return "Neutral (ปานกลาง / เป็นกลาง)"
    else:
      return "Positive (ดี / ด้านบวก)"

  if text != "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)":
    pos_score = sum(1 for w in POS_WORDS if w in text)
    neg_score = sum(1 for w in NEG_WORDS if w in text)

    if neg_score > pos_score:
      return "Negative (แย่ / ด้านลบ)"
    elif pos_score > neg_score:
      return "Positive (ดี / ด้านบวก)"

  return "Neutral (ปานกลาง / เป็นกลาง)"


# --- 3. ฟังก์ชันเรียกใช้ Gemini API สรุป Pros & Cons (ใส่ Cache ป้องกันยิงซ้ำ) ---
GEMINI_MODEL = "gemini-flash-latest"  # อัปเดตจาก gemini-2.0-flash-lite ที่ถูกปิดใช้งานแล้ว


@st.cache_data(show_spinner=False)
def analyze_pros_cons_with_ai(review_texts: list, api_key: str):
  if not review_texts:
    return None

  sample_text = "\n".join([f"- {txt}" for txt in review_texts[:80]])

  prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์รีวิวสินค้า 
    จากข้อความรีวิวต่อไปนี้ จงสรุปจุดเด่น (Pros) 3 ข้อ และจุดด้อย (Cons) 3 ข้อ ที่ผู้ซื้อพูดถึงมากที่สุด 
    พร้อมประเมินสัดส่วนเป็นเปอร์เซ็นต์ (%) ของคนที่พูดถึงเรื่องนั้นๆ 

    รายการรีวิวสินค้า:
    {sample_text}

    โครงสร้าง JSON ที่ต้องการ:
    {{
      "pros": [
        {{"topic": "ชื่อจุดดี 1", "pct": 85}},
        {{"topic": "ชื่อจุดดี 2", "pct": 70}},
        {{"topic": "ชื่อจุดดี 3", "pct": 50}}
      ],
      "cons": [
        {{"topic": "ชื่อจุดเสีย 1", "pct": 15}},
        {{"topic": "ชื่อจุดเสีย 2", "pct": 10}},
        {{"topic": "ชื่อจุดเสีย 3", "pct": 5}}
      ]
    }}
    """

  try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)
  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini AI: {e}")
    return None


# --- 3.5 ฟังก์ชันสร้าง PDF Report ---
def generate_pdf_report(
    total_reviews: int,
    pos_pct: float,
    neu_pct: float,
    neg_pct: float,
    ai_data: dict = None,
) -> bytes:
  ai_available = bool(ai_data and (ai_data.get("pros") or ai_data.get("cons")))

  if ai_available:
    pros_html = ""
    for item in ai_data.get("pros", []):
      pct = min(max(int(item.get("pct", 0)), 0), 100)
      pros_html += f"""
          <div style="margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; font-size: 9.5pt; font-weight: 600; color: #334155; margin-bottom: 3px;">
                  <span>{item.get('topic')}</span>
                  <span style="color: #16a34a;">{pct}%</span>
              </div>
              <div style="background-color: #f1f5f9; height: 8px; border-radius: 4px; overflow: hidden;">
                  <div style="background-color: #22c55e; height: 100%; width: {pct}%;"></div>
              </div>
          </div>
          """

    cons_html = ""
    for item in ai_data.get("cons", []):
      pct = min(max(int(item.get("pct", 0)), 0), 100)
      cons_html += f"""
          <div style="margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; font-size: 9.5pt; font-weight: 600; color: #334155; margin-bottom: 3px;">
                  <span>{item.get('topic')}</span>
                  <span style="color: #dc2626;">{pct}%</span>
              </div>
              <div style="background-color: #f1f5f9; height: 8px; border-radius: 4px; overflow: hidden;">
                  <div style="background-color: #ef4444; height: 100%; width: {pct}%;"></div>
              </div>
          </div>
          """

    ai_section_html = f"""
        <div class="section-title">💡 ผลการวิเคราะห์ Pros & Cons (วิเคราะห์ด้วย AI)</div>
        <table class="content-table">
            <tr>
                <td class="column"><div class="col-title-pros">🟢 จุดเด่นที่ลูกค้าประทับใจ (Pros)</div>{pros_html}</td>
                <td class="column"><div class="col-title-cons">🔴 ข้อเสีย / จุดที่ควรปรับปรุง (Cons)</div>{cons_html}</td>
            </tr>
        </table>
        """
  else:
    ai_section_html = """
        <div class="section-title">💡 ผลการวิเคราะห์ Pros & Cons (วิเคราะห์ด้วย AI)</div>
        <div style="background-color: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px; text-align: center; color: #9a3412; font-size: 10pt;">
            ⚠️ ไม่สามารถวิเคราะห์จุดเด่น-จุดด้อยด้วย AI ได้ในขณะนี้ (เช่น เชื่อมต่อ Gemini AI ไม่สำเร็จ หรือยังไม่ได้ใส่ API Key)<br>
            รายงานนี้แสดงเฉพาะภาพรวมสัดส่วน Sentiment เท่านั้น
        </div>
        """

  html_template = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 15mm; background-color: #f8fafc; }}
            body {{ font-family: 'Sarabun', 'Loma', 'Garuda', 'Noto Sans Thai', 'Helvetica Neue', Arial, sans-serif; color: #1e293b; margin: 0; padding: 0; font-size: 10pt; line-height: 1.5; }}
            .header {{ background-color: #0f172a; color: #ffffff; margin: -15mm -15mm 20px -15mm; padding: 25px 20px; text-align: center; }}
            .header h1 {{ margin: 0 0 6px 0; font-size: 18pt; font-weight: 700; }}
            .header p {{ margin: 0; font-size: 10pt; color: #94a3b8; }}
            .meta-grid {{ width: 100%; margin-bottom: 20px; border-collapse: collapse; }}
            .meta-card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; text-align: center; }}
            .meta-title {{ font-size: 8.5pt; color: #64748b; font-weight: 600; margin-bottom: 4px; }}
            .meta-value {{ font-size: 13pt; font-weight: 700; color: #0f172a; }}
            .section-title {{ font-size: 12pt; font-weight: 700; color: #0f172a; border-left: 4px solid #3b82f6; padding-left: 8px; margin: 20px 0 12px 0; }}
            .content-table {{ width: 100%; border-collapse: separate; border-spacing: 10px 0; }}
            .column {{ width: 50%; vertical-align: top; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
            .col-title-pros {{ color: #166534; font-size: 11pt; font-weight: 700; border-bottom: 2px solid #bbf7d0; padding-bottom: 6px; margin-bottom: 12px; }}
            .col-title-cons {{ color: #991b1b; font-size: 11pt; font-weight: 700; border-bottom: 2px solid #fecaca; padding-bottom: 6px; margin-bottom: 12px; }}
            .footer {{ margin-top: 25px; text-align: center; font-size: 8pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>รายงานสรุปผลการวิเคราะห์รีวิวสินค้า (Shopee AI)</h1>
            <p>รายงานภาพรวมความรู้สึกและจุดเด่น-จุดด้อยสินค้า</p>
        </div>
        <table class="meta-grid">
            <tr>
                <td style="width: 25%; padding: 0 4px;"><div class="meta-card"><div class="meta-title">จำนวนรีวิวทั้งหมด</div><div class="meta-value">{total_reviews:,}</div></div></td>
                <td style="width: 25%; padding: 0 4px;"><div class="meta-card"><div class="meta-title">รีวิวดี (4-5 ดาว)</div><div class="meta-value" style="color: #16a34a;">{pos_pct:.1f}%</div></div></td>
                <td style="width: 25%; padding: 0 4px;"><div class="meta-card"><div class="meta-title">รีวิวกลางๆ (3 ดาว)</div><div class="meta-value" style="color: #d97706;">{neu_pct:.1f}%</div></div></td>
                <td style="width: 25%; padding: 0 4px;"><div class="meta-card"><div class="meta-title">รีวิวแย่ (1-2 ดาว)</div><div class="meta-value" style="color: #dc2626;">{neg_pct:.1f}%</div></div></td>
            </tr>
        </table>
        {ai_section_html}
        <div class="footer">สร้างรายงานอัตโนมัติด้วย Shopee Sentiment Analyzer AI</div>
    </body>
    </html>
    """
  return HTML(string=html_template).write_pdf()


# --- 4. ส่วนแสดงผล UI Streamlit ---
uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์ CSV รีวิวสินค้า Shopee", type=["csv"]
)

if uploaded_file is not None:
  raw_df = None
  for enc in ["utf-8-sig", "utf-8", "cp874", "tis-620"]:
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

  with st.spinner("กำลังประมวลผลข้อมูล..."):
    std_df = process_shopee_dataset(raw_df)
    std_df["sentiment_result"] = std_df.apply(evaluate_sentiment, axis=1)

  # 📊 Dashboard Metrics
  col1, col2, col3 = st.columns(3)
  pos_count = (std_df["sentiment_result"] == "Positive (ดี / ด้านบวก)").sum()
  neg_count = (std_df["sentiment_result"] == "Negative (แย่ / ด้านลบ)").sum()
  neu_count = (
      std_df["sentiment_result"] == "Neutral (ปานกลาง / เป็นกลาง)"
  ).sum()

  col1.metric("รีวิวดี (4-5 ดาว) 😀", f"{pos_count} รายการ")
  col2.metric("รีวิวแย่ (1-2 ดาว) 😡", f"{neg_count} รายการ")
  col3.metric("รีวิวกลางๆ (3 ดาว) 😐", f"{neu_count} รายการ")

  # 💡 ส่วนแสดงผล AI Pros & Cons Summary
  st.markdown("---")
  st.subheader("💡 สรุปจุดดี-จุดด้อย (วิเคราะห์ด้วย AI)")

  if api_key:
    text_reviews = std_df[
        std_df["review_text"] != "ไม่มีข้อความรีวิว (ให้ดาวอย่างเดียว)"
    ]["review_text"].tolist()

    if text_reviews:
      with st.spinner("🤖 AI กำลังประมวลผล..."):
        ai_data = analyze_pros_cons_with_ai(text_reviews, api_key)

      if ai_data:
        col_pros, col_cons = st.columns(2)

        with col_pros:
          st.markdown("#### 🟢 ข้อดีที่คนพูดถึงเยอะที่สุด")
          for item in ai_data.get("pros", []):
            pct = min(max(int(item.get("pct", 0)), 0), 100)
            st.container(border=True).markdown(
                f"**{item.get('topic')}**\n*ผู้พูดถึง: {pct}%*"
            )
            st.progress(pct / 100.0)

        with col_cons:
          st.markdown("#### 🔴 ข้อเสียที่ต้องระวัง / จุดควรปรับปรุง")
          for item in ai_data.get("cons", []):
            pct = min(max(int(item.get("pct", 0)), 0), 100)
            st.container(border=True).markdown(
                f"**{item.get('topic')}**\n*ผู้พูดถึง: {pct}%*"
            )
            st.progress(pct / 100.0)
      else:
        st.warning(
            "⚠️ ไม่สามารถวิเคราะห์จุดเด่น-จุดด้อยด้วย AI ได้ในขณะนี้"
            " แต่ยังดาวน์โหลดรายงาน PDF (ภาพรวม Sentiment) ได้ตามปกติ"
        )

      # 📄 ปุ่มดาวน์โหลดรายงาน PDF (แสดงเสมอ ไม่ว่า AI จะสำเร็จหรือไม่)
      total_cnt = len(std_df)
      pos_p = (pos_count / total_cnt) * 100 if total_cnt > 0 else 0
      neu_p = (neu_count / total_cnt) * 100 if total_cnt > 0 else 0
      neg_p = (neg_count / total_cnt) * 100 if total_cnt > 0 else 0

      pdf_bytes = generate_pdf_report(
          total_reviews=total_cnt,
          pos_pct=pos_p,
          neu_pct=neu_p,
          neg_pct=neg_p,
          ai_data=ai_data,
      )

      st.markdown("<br>", unsafe_allow_html=True)
      st.download_button(
          label="📄 ดาวน์โหลดรายงานสรุป (PDF)",
          data=pdf_bytes,
          file_name="shopee_sentiment_summary.pdf",
          mime="application/pdf",
      )
    else:
      st.info("ไม่พบข้อความรีวิวเพียงพอสำหรับวิเคราะห์ Pros & Cons")
  else:
    st.info(
        "👉 กรุณาใส่ Gemini API Key ที่ Sidebar ด้านซ้าย"
        " เพื่อเปิดการใช้งานส่วนสรุป AI"
    )

  st.markdown("---")

  # 📈 กราฟ สรุปสัดส่วน Sentiment
  st.subheader("📊 กราฟสรุปสัดส่วน Sentiment")
  st.caption("💡 คลิกที่แท่งกราฟเพื่อกรองตารางด้านล่างให้เหลือเฉพาะกลุ่มนั้น")

  sentiment_counts = std_df["sentiment_result"].value_counts().reset_index()
  sentiment_counts.columns = ["sentiment_result", "count"]

  sentiment_order = [
      "Negative (แย่ / ด้านลบ)",
      "Neutral (ปานกลาง / เป็นกลาง)",
      "Positive (ดี / ด้านบวก)",
  ]
  color_map = {
      "Negative (แย่ / ด้านลบ)": "#e74c3c",
      "Neutral (ปานกลาง / เป็นกลาง)": "#f1c40f",
      "Positive (ดี / ด้านบวก)": "#2ecc71",
  }

  click_selection = alt.selection_point(
      name="select", fields=["sentiment_result"]
  )

  chart = (
      alt.Chart(sentiment_counts)
      .mark_bar()
      .encode(
          x=alt.X(
              "sentiment_result:N", title="Sentiment", sort=sentiment_order
          ),
          y=alt.Y("count:Q", title="จำนวนรีวิว"),
          color=alt.Color(
              "sentiment_result:N",
              scale=alt.Scale(
                  domain=list(color_map.keys()), range=list(color_map.values())
              ),
              legend=None,
          ),
          opacity=alt.condition(
              click_selection, alt.value(1.0), alt.value(0.4)
          ),
          tooltip=[
              alt.Tooltip("sentiment_result:N", title="Sentiment"),
              alt.Tooltip("count:Q", title="จำนวน"),
          ],
      )
      .add_params(click_selection)
      .properties(height=350)
  )

  event = st.altair_chart(
      chart, use_container_width=True, on_select="rerun", key="sentiment_chart"
  )

  selected_sentiments = []
  if event and event.get("selection") and event["selection"].get("select"):
    selected_sentiments = [
        p["sentiment_result"] for p in event["selection"]["select"]
    ]

  if selected_sentiments:
    filtered_df = std_df[std_df["sentiment_result"].isin(selected_sentiments)]
    st.info(
        f"🔍 กำลังกรองเฉพาะ: **{', '.join(selected_sentiments)}**"
        f" ({len(filtered_df)} รายการ) — คลิกแท่งเดิมอีกครั้งเพื่อยกเลิกกรอง"
    )
  else:
    filtered_df = std_df

  # 📋 ตารางแสดงผล
  st.subheader("📋 ตารางแสดงผลลัพธ์การวิเคราะห์")
  st.dataframe(
      filtered_df[[
          "review_date",
          "product_name",
          "variation",
          "rating",
          "review_text",
          "sentiment_result",
      ]],
      use_container_width=True,
  )

  # 📥 ดาวน์โหลด CSV
  csv = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode(
      "utf-8-sig"
  )
  download_label = (
      f"📥 ดาวน์โหลดผลลัพธ์ที่กรองแล้ว ({', '.join(selected_sentiments)}) (CSV)"
      if selected_sentiments
      else "📥 ดาวน์โหลดตารางสรุปผลลัพธ์ทั้งหมด (CSV)"
  )
  st.download_button(
      label=download_label,
      data=csv,
      file_name="sentiment_results.csv",
      mime="text/csv",
  )
else:
  st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์")