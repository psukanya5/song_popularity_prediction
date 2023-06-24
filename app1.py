import numpy as np
import streamlit as st
import pickle
import re #regular expression
with open('model1.pkl','rb') as f:
    model = pickle.load(f)
st.title('SONG PREDICTION')
#list comprehension
collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i!=""]
#getting input
input_data = st.text_input("Enter the input data separated by commas")
num_list = []
num_list = collect_numbers(input_data)
#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(num_list)

#reshaping numpy array for predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#making a predictive system
if st.button("Classify") :
    predictions = model.predict(input_data_reshaped)
    if predictions == '1':
        st.write("The song is Popular")
    else:
        st.write("The song is not Popular")
