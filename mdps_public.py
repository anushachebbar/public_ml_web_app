import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# ================== Load Models ==================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open('parkinsons_scaler.sav', 'rb'))

# ================== Sidebar Navigation ==================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ================== Diabetes Prediction ==================
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')
        except:
            st.error("⚠️ Please enter valid numeric values")

# ================== Heart Disease Prediction ==================
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    age = st.text_input('Age')
    sex = st.text_input('Sex (1 = Male, 0 = Female)')
    cp = st.text_input('Chest Pain types (0–3)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    restecg = st.text_input('Resting Electrocardiographic results (0–2)')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment (0–2)')
    ca = st.text_input('Number of major vessels (0–3)')
    thal = st.text_input('Thalassemia (0–3)')

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), float(sex), float(cp), float(trestbps),
                          float(chol), float(fbs), float(restecg), float(thalach),
                          float(exang), float(oldpeak), float(slope),
                          float(ca), float(thal)]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                st.success('The person has heart disease')
            else:
                st.success('The person does not have heart disease')
        except:
            st.error("⚠️ Please enter valid numeric values")

# ================== Parkinson’s Prediction ==================
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent),
                          float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                          float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                          float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2),
                          float(D2), float(PPE)]

            user_input = np.array(user_input).reshape(1, -1)

            # Scale before prediction
            user_input_scaled = parkinsons_scaler.transform(user_input)
            parkinsons_prediction = parkinsons_model.predict(user_input_scaled)

            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")
        except:
            st.error("⚠️ Please enter valid numeric values")
