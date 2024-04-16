import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import streamlit as st
plt.style.use('default')
st.set_page_config(
    page_title='Diabetes Screening Tool',
    page_icon='🕵️‍♀️',
    layout='wide'
)
# dashboard title
# st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 快速筛查糖尿病</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Machine Learning: Rapid Diabetes Screening</h1>", unsafe_allow_html=True)
shapdatadf =pd.read_excel(r'shapdatadf1.xlsx')

def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.selectbox('Dietary health level', [1, 2, 3, 4, 5])
    a2 = st.sidebar.selectbox("Education level", [1, 2, 3, 4, 5])
    a3 = st.sidebar.slider('Age', 16, 120, 51)
    a4 = st.sidebar.selectbox('Race', [1, 2, 3, 4, 5])
    a5 = st.sidebar.selectbox('Hypertension', [1, 2])
    a6 = st.sidebar.slider('Waist circumference', 55, 178, 99)
    a7 = st.sidebar.slider('BMI', 10,29,80)
    a8 = st.sidebar.selectbox('Nocturia', [1, 2, 3, 4, 5])
    a9 = st.sidebar.selectbox('Family medical history', [1, 2])
    output = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
    return output

outputdf = user_input_features()

st.title('SHAP Value')
Xgboost_model = XGBClassifier()
Xgboost_model.load_model(r"fraud1")
st.title('Make predictions in real time')
outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)
# st.write('User input parameters below ⬇️')
# st.write(outputdf)
p1 = Xgboost_model.predict(outputdf)[0]
p2 = Xgboost_model.predict_proba(outputdf)
placeholder6 = st.empty()
with placeholder6.container():
    f1, f2 = st.columns(2)
    with f1:
        st.write('User input parameters below ⬇️')
        st.write(outputdf)
        st.write(f'Predicted class: {p1}')
        st.write('Predicted class Probability')
        st.write('0️⃣ means its a real transaction, 1️⃣ means its a Fraud transaction')
        st.write(p2)
    with f2:
        explainer = shap.Explainer(Xgboost_model)
        shap_values = explainer(outputdf)
        # st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')