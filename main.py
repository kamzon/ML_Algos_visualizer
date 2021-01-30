import streamlit as st

import numpy as np
import sklearn as skl
from sklearn import datasets 


st.title("Machine learning Algorithms visualizer")

#st.write("""ML Algorithms Visualizer""")

dataset = st.sidebar.selectbox("select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
st.write(dataset)


calssifier= st.sidebar.selectbox("select classifier", ("KNN", "SVM"))

st.write("Classifier :",calssifier)

def import_dataset(name_data):
    if(name_data=="Iris"):
        data=datasets.load_iris()

    if(name_data=="Breast Cancer"):
        data=datasets.load_breast_cancer()

    if(name_data=="Wine dataset"):
        data=datasets.load_wine()

    x = data.data
    y = data.target

    return x,y

x,y = import_dataset(dataset)


st.write("Dataset Shape : ", x.shape)
st.write("Number of classes : ", len(np.unique(y)))
st.write(x)

def add_parametrs(clsfier):
    parms = dict()
    if(clsfier == "KNN"):
        k = st.sidebar.slider("K",1, 20)
        parms['K']=k

    if(clsfier == "SVM"):
        C = st.sidebar.slider("C",0.01, 20.0)
        parms['C']=C

    return parms

pram = add_parametrs(calssifier)


        

    


