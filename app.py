# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====== تحميل البيانات ======
data = pd.read_csv(r"C:\Users\كمبيو الكتريك\Documents\Downloads\archive\spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label_num'] = data.label.map({'ham':0, 'spam':1})

# ====== تقسيم البيانات ======
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# ====== تحويل النصوص إلى أرقام ======
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ====== تدريب موديلات مختلفة ======
# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)
acc_nb = accuracy_score(y_test, y_pred_nb)

# SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
acc_svm = accuracy_score(y_test, y_pred_svm)

# اختيار أفضل موديل
best_model = svm_model if acc_svm >= acc_nb else nb_model
best_pred = y_pred_svm if acc_svm >= acc_nb else y_pred_nb
best_name = "SVM ✅" if acc_svm >= acc_nb else "Naive Bayes"

# ====== قائمة كلمات السبام الموسعة ======
spam_keywords = [
    'win', 'winner', 'free', 'gift', 'money', 'offer', 'urgent', 'click',
    'selected', 'prize', 'claim', 'bonus', 'deal', 'limited', 'buy', 'buy now',
    'discount', 'voucher', 'exclusive', 'promotion', 'reward', 'cash', 'congratulations',
    'lottery', 'guarantee', 'instant', 'trial', 'subscribe', 'urgent response',
    'act now', 'account', 'verify', 'password', 'loan', 'credit', 'investment', 'cheap'
]

# ====== دالة التنبؤ ======
def predict_message(msg):
    found_keywords = [w for w in spam_keywords if w.lower() in msg.lower()]
    if found_keywords:
        return 'Spam', found_keywords
    else:
        msg_vec = vectorizer.transform([msg])
        prediction = best_model.predict(msg_vec)[0]
        return ('Spam' if prediction==1 else 'Ham', found_keywords)

# ====== واجهة Streamlit ======
st.title("📧 Spam/Ham Classifier – نسخة احترافية")
st.subheader(f"أفضل موديل: {best_name}")
st.write(f"دقة النموذج على مجموعة الاختبار: {accuracy_score(y_test, best_pred):.4f}")

# ====== Confusion Matrix ======
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(plt)

# ====== قائمة الرسائل الجاهزة للتجربة ======
test_messages = [
    "Congratulations! You have won a $1000 gift card. Click here to claim now!",
    "Limited time offer! Buy one get one free!",
    "Hey, are we still meeting for lunch today?",
    "Don't forget to bring the homework tomorrow.",
    "URGENT! Your account has been compromised. Reset your password immediately!",
    "Hi Ahmed, can you send me the report by tonight?",
    "You have been selected for a free vacation. Reply YES to claim.",
    "Security alert: Login attempt detected from a new device. Confirm it immediately."
]

selected_msg = st.selectbox("اختر رسالة لتجربتها:", test_messages)

if st.button("تصنيف الرسالة"):
    if selected_msg.strip() != "":
        result, keywords = predict_message(selected_msg)
        if result == 'Spam':
            st.error(f"🚨 الرسالة مصنفة كـ {result}")
        else:
            st.success(f"✅ الرسالة مصنفة كـ {result}")
        if keywords:
            st.warning(f"📌 الكلمات المفتاحية الموجودة: {', '.join(keywords)}")
    else:
        st.warning("الرجاء إدخال رسالة لتصنيفها!")

# ====== إدخال رسالة جديدة يدويًا ======
st.subheader("أو أدخل رسالة جديدة للتجربة:")
user_input = st.text_area("اكتب رسالتك هنا:")

if st.button("تصنيف الرسالة الجديدة"):
    if user_input.strip() != "":
        result, keywords = predict_message(user_input)
        if result == 'Spam':
            st.error(f"🚨 الرسالة مصنفة كـ {result}")
        else:
            st.success(f"✅ الرسالة مصنفة كـ {result}")
        if keywords:
            st.warning(f"📌 الكلمات المفتاحية الموجودة: {', '.join(keywords)}")
    else:
        st.warning("الرجاء إدخال رسالة لتصنيفها!")