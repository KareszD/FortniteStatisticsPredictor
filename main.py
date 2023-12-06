import datetime
import streamlit as st
import numpy as np
import pickle
from datetime import time,date

model = pickle.load(open('new_model.pkl','rb'))

def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Debreczeni Károly Balázs Predictor</h1>", unsafe_allow_html=True)

  #'Eliminations', 'Assists', 'Placed', 'Revives', 'Hits', 'day_of_month', 'month', 'hour', 'is_high', 'is_sober', 'Accuracy_numeric_values'
  date = st.date_input("Select the date of the match",value=datetime.date(2023, 4, 14))
  time = st.time_input("Select the time of the match",value=datetime.time(12, 00))
  placement = st.slider("Select your placement", 1, 100)
  mentalState = st.selectbox("Select your mental state", options={'high','sober'})
  elims = st.number_input("Input your eliminations",value=0,min_value=0)
  assists = st.number_input("Input your assists",value=0,min_value=0)
  revives = st.number_input("Input your revives",value=0,min_value=0)
  accuracy = st.slider("Select your accuracy",value=50,min_value=0,max_value=100)
  hits = st.number_input("Input your hits",value=0,min_value=0)
  ishigh = 0
  issober = 0
  if mentalState == 'high':
    ishigh = 1
  else:
    issober = 1

  inputs = [[elims,assists,placement,revives,hits,date.day,date.month,time.hour,ishigh,issober,accuracy]] 

  if st.button('Predict'): 

    result = model.predict(inputs)
    print(result)
    updated_res = result.flatten().astype(float)

    st.success(f'The predicted amount of headshots: {np.round(updated_res[0]).astype(int)}')

if __name__ =='__main__':
  main() 
