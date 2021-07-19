import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/ML lab/endterm/model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('/content/drive/My Drive/ML lab/endterm/Classification Dataset33.csv')
# Extracting independent variable:
X = dataset.iloc[:, 0:8].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  output= model.predict(sc.transform([[Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("Prediction : ", output)
  if output==[1]:
    prediction=" Patient will have Desease"
  else:
    prediction=" Patient will have not Desease"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("EndTerm Model Deployment")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Glucose = st.number_input('Insert Glucose level')
    BP = st.number_input('Insert BP level')
    SkinThickness = st.number_input('Insert Skinthickness')
    Insulin = st.number_input('Insert insulin')
    BMI = st.number_input('Insert BMI')
    PedigreeFunction =  st.number_input('Insert PedigreeFunction')
    Age = st.number_input('Insert a Age',12,60)

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Pritesh Kumar")
      st.subheader("Student, Department of Computer Engineering")

if __name__=='__main__':
  main()