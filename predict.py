import streamlit as st
import pickle 
import numpy as np


def load_model():
    with open ('saved_steps.pk1','rb') as file:
        data=pickle.load(file)
    return data

data=load_model()

reg_loaded=data['reg_model']
poly_loaded=data['poly_reg']
poly_trans_loaded=data['poly_trans']
ridge_loaded=data['ridge']
lasso_loaded=data["lasso"]
elnet_loaded=data['el_net']


def show_predict_page():
    st.title("House Price Prediction for Georgia")

    st.write(""" ### We need some information to predict the house price """)
    area=st.number_input('Enter area of the house')
    room=st.slider("Number of rooms",1,10,3)
    bedroom=st.slider('Number of bedrooms',0,4,2)
    furniture = st.checkbox('Should contain furniture',value=False,)  ##furniture will contain boolean value true or false

    ok=st.button('Predict price')

    if ok:
        x_try=np.array([[area,room,bedroom,int(furniture)]])
        x_try_new=poly_trans_loaded.transform(x_try)
        
        price_pred=[reg_loaded.predict(x_try)[0],poly_loaded.predict(x_try_new)[0],ridge_loaded.predict(x_try)[0],lasso_loaded.predict(x_try)[0],elnet_loaded.predict(x_try)[0]]
        price=np.array(price_pred).mean()
        st.subheader(f"The price of the house is {price:.2f}")