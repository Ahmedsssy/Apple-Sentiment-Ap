import streamlit as st
from transformers import pipeline

# إعداد واجهة الموقع
st.set_page_config(page_title="Apple Sentiment Analysis", page_icon="🍎")
st.title("🍎 Apple Sentiment Analysis Dashboard")
st.markdown("تحليل مشاعر التويتات باستخدام موديل RoBERTa الجاهز من Hugging Face")

# تحميل الموديل مباشرة من Hugging Face (بدون ملفات محليا)
@st.cache_resource
def get_classifier():
    # ده موديل جاهز ومتطور جدا للتحليل
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = get_classifier()

# خانة إدخال النص
user_input = st.text_area("أدخل التويتة التي تريد تحليلها هنا:", "I love the new iPhone 15 Pro Max!")

if st.button("تحليل الآن"):
    with st.spinner('جاري التحليل...'):
        result = classifier(user_input)
        
        # عرض النتيجة بشكل شيك
        label = result[0]['label']
        score = result[0]['score']
        
        if label == 'positive':
            st.success(f"النتيجة: إيجابي (Positive) ✅ - الثقة: {score:.2f}")
        elif label == 'negative':
            st.error(f"النتيجة: سلبي (Negative) ❌ - الثقة: {score:.2f}")
        else:
            st.warning(f"النتيجة: محايد (Neutral) 😐 - الثقة: {score:.2f}")

st.divider()
st.info("ملاحظة: هذا التطبيق يستخدم موديل Transformer جاهز من Hugging Face.")