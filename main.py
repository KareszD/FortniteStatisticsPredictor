#loading our libraries
import datetime
from keras.models import load_model
import streamlit as st
import numpy as np
import pickle

def main():
  if 'link_input' not in st.session_state:
    st.session_state['link_input'] = 'All Data'

  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Debreczeni Károly Predictor</h1>", unsafe_allow_html=True)

  modelselect = st.selectbox("Select your model", options={'Neural Network','Performante','All Data'},key='link_input',args=[st.session_state.link_input])

  if st.session_state["link_input"]== 'All Data':
    st.session_state.disabled = False
  elif st.session_state["link_input"]== 'Performante'or'Neural Network':
    st.session_state.disabled = True

  st.session_state['model'] = modelselect
  #'Eliminations', 'Assists', 'Placed', 'Revives', 'Hits', 'day_of_month', 'month', 'hour', 'is_high', 'is_sober', 'Accuracy_numeric_values'
  left, right = st.columns(2)
  with left:
    date = st.date_input("Select the date of the match", value=datetime.date(2023, 4, 14),
                         disabled=st.session_state.get("disabled", True))
  with right:
    time = st.time_input("Select the time of the match", value=datetime.time(12, 00),
                         disabled=st.session_state.get("disabled", False))
  placement = st.slider("Select your placement", 1, 100)
  mentalState = st.selectbox("Select your mental state", options={'high','sober'},disabled=st.session_state.get("disabled", False))
  left, right = st.columns(2)
  with left:
    elims = st.number_input("Input your eliminations",value=0,min_value=0)
  with right:
    assists = st.number_input("Input your assists",value=0,min_value=0)
  left, right = st.columns(2)
  with left:
    revives = st.number_input("Input your revives",value=0,min_value=0,disabled=st.session_state.get("disabled", False))
  with right:
    hits = st.number_input("Input your hits", value=0, min_value=0)

  accuracy = st.slider("Select your accuracy",value=50,min_value=0,max_value=100)

  ishigh = 0
  issober = 0
  if mentalState == 'high':
    ishigh = 1
  else:
    issober = 1

  headshots = st.number_input("Actual headshot value(only useful with NeuralNetwork)",value=0)
  if st.button('Predict'): #making and printing our prediction
    if modelselect== 'Performante':
      inputs = [[elims, assists, placement, hits, accuracy]]
      model = pickle.load(open('bestprefmodel.pkl', 'rb'))
    elif modelselect== 'All Data':
      inputs = [[elims, assists, placement, revives, hits, date.day, date.month, time.hour, ishigh, issober,accuracy]]
      model = pickle.load(open('alldatamodel.pkl', 'rb'))
    else:
      inputs = [[elims, assists, placement, hits, accuracy]]
      model = load_model('nnmodel.keras')
    result = model.predict(inputs)
    print(result)
    updated_res = result.flatten().astype(float)

    st.success(f'The predicted amount of headshots: {np.round(updated_res[0]).astype(int)}')
    if modelselect== 'Neural Network':
      model.compile(optimizer='adam', loss='mean_squared_error')
      model.fit(inputs, [headshots], 10)
      model.save('nnmodel.keras')


if __name__ =='__main__':
  main() #calling the main method
