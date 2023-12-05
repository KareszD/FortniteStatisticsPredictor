#loading our libraries
import streamlit as st
import numpy as np
import string
import pickle

#loading our model
model = pickle.load(open('best.pkl','rb'))

def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Debreczeni Károly Predictor</h1>", unsafe_allow_html=True)
  elims = st.number_input("Input your eliminations",value=0,min_value=0)
  assists = st.number_input("Input your assists",value=0,min_value=0)
  placement = st.slider("Select your placement",1,100)
  revives = st.number_input("Input your revives",value=0,min_value=0)
  hits = st.number_input("Input your hits",value=0,min_value=0)

  inputs = [[elims,assists,placement,revives,hits]] 

  if st.button('Predict'): 

    result = model.predict(inputs)
    print(result)
    updated_res = result.flatten().astype(float)

    st.success(f'The predicted amount of headshots: {np.round(updated_res[0]).astype(int)}')

if __name__ =='__main__':
  main() 
