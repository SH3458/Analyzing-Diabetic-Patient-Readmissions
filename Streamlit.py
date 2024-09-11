#loading required libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score #plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
import streamlit as st
import joblib
import pickle


# #This chunk is just for testing remove once done with everything
# def main():
#     st.title("Introduction to building Streamlit WebApp")
#     st.sidebar.title("This is the sidebar")
#     st.sidebar.markdown("Let's start with binary classification!!")
# if __name__ == '__main__':
#     main()


# Obtaining the scalar used to scale the data, this will be used once all user data is input, to scale all the features based on our data.
df = pd.read_csv('cleaned_data_unscaled.csv')
scalar = StandardScaler().fit(df.iloc[:,:-1])

# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

# This is the main function in which we define our webpage, all the  features which user needs to input are entered in the main function

def main(): 
      # giving the webpage a title 
    st.title("Prediction for readmission of a patient") 
    st.header("Insert Patient Info")

    age = st.number_input('Patient Age:', min_value=5.0, max_value=95.0, value=10.0, step = 10.0)
    admission_type_id = st.selectbox('Admission Type:', ['1', '2', '3', '4', '7'])
    time_in_hospital = st.number_input('Amount of time patient was in hospital:', min_value= 1.0, max_value = 14.0, value = 1.0, step = 1.0)
    num_lab_procedures = st.number_input('Number of Lab Procedures:', min_value= 1.0, max_value = 126.0, value = 1.0, step = 1.0)
    num_procedures = st.number_input('Number of Procedures:', min_value= 0.0, max_value = 6.0, value = 1.0, step = 1.0)
    num_medications = st.number_input('Number of Medications:', min_value= 1.0, max_value = 81.0, value = 1.0, step = 1.0)
    number_outpatient = st.number_input('Number of Outpatients:', min_value= 0.0, max_value = 40.0, value = 1.0, step=1.0)
    number_emergency = st.number_input('Number of Emergencies:', min_value= 0.0, max_value = 54.0, value = 1.0, step=1.0)
    number_inpatient = st.number_input('Number of Inpatient:', min_value= 0.0, max_value = 16.0, value = 1.0, step=1.0)
    number_diagnoses = st.number_input('Number of Diagnoses:', min_value= 1.0, max_value = 16.0, value = 1.0, step=1.0)
    change  = st.selectbox('Change of Medication:', ['0', '1'])
    diabetesMed = st.selectbox('Diabetes Medication:', ['0', '1'])
    is_female = st.selectbox('If Female:', ['0', '1'])
    race__AfricanAmerican = st.selectbox('If African American:', ['0', '1'])
    race__Asian = st.selectbox('If Asian:', ['0', '1'])
    race__Caucasian = st.selectbox('If caucasian:', ['0', '1'])
    race__Hispanic = st.selectbox('If Hispanic:', ['0', '1'])
    race__Other = st.selectbox('Other Race:', ['0', '1'])

    features_list = ['metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
    
    feature_value = dict()
    for feature in features_list:
        feature_value[feature] = st.selectbox(f'{feature}:', ['0', '1', '2', '3'])

    #input the details entered by user in our Random forest model
    x_pred = np.array([age,admission_type_id,time_in_hospital,num_lab_procedures,num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,number_diagnoses] + list(feature_value.values()) + [change,diabetesMed,is_female,race__AfricanAmerican,race__Asian,race__Caucasian,race__Hispanic,race__Other])

    #scaling the features before running random forest model
    x_pred = scalar.transform(x_pred.reshape(1,-1))

    prediction = classifier.predict(x_pred.reshape(1,-1)) 

    # Mapping Prediction of readmitibility to original meaningful value
    if prediction[0] == 0:
        pred = 'None(will not be readmitted)'
    elif prediction[0] == 1:
        pred = 'after 30 days'
    else:
        pred = 'within 30 days'

    
    if st.button('Predict Readmission'):
        
        st.success(f'The chances of patient"s readmisibility is {pred}')

if __name__=='__main__': 
    main() 