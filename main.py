import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import sklearn as skl
from sklearn import datasets 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score


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


def get_classifier(clsf_name, parms):
    
    if(clsf_name == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=parms['K'])
        

    if(clsf_name == "SVM"):
        clf = SVC(C=parms['C'])
        

    return clf

clf = get_classifier(calssifier, pram)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=1234)

clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)

st.write("Classifier : ",calssifier)
st.write("Accuracy = ",acc)




pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot(fig)



    


