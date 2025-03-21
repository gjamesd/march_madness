import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
from tqdm import tqdm
import numpy as np 
import plotly.express as px
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report 
import numpy as np
from sklearn.metrics import silhouette_score
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from plotly.graph_objects import Layout
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import pickle

st.set_page_config(
    page_title="Rugby Kicking Prediction",
    #page_icon="👋",
)

st.write("# Please follow the Drop down boxes below to create the inputs to run the algorithm and see what the probability is of making a given shot at goal.")

st.write("Also go Ireland!")


y_m = st.number_input('How many meters in from touch are you kicking?')


l_r = st.selectbox(

    'Left or Right Touch Line?',

    ('Left','Right',))
#displaying the selected option

x_m = st.number_input('How many meters in from the try line are you kicking?')


st.write('You are in from the try', x_m,' and ',y_m,' meters in from the ',l_r,' touch line.')


#Pitch diagram 
def read_in_data():
    data = pd.read_csv('../data/total_data.csv')
    return data 

data = read_in_data()

def get_scatter_data(df):
    fig = px.scatter(x=df['x_meters'], y=df['y_meters'], color = data['result'], opacity= data['opacity'], 
    
    title="Pitch Diagram With All Scrapped Data",
                    labels = {"color":'Make or Miss', 
                             "x":"Meters From Try Line",  },
                       height = 600
                    )
    
    
    fig.add_shape(type="rect",
    x0=0, y0=32.5, x1=-1, y1=37.5,
    line=dict(color="black"),
    )
    fig.add_shape(type="rect",
        x0=0, y0=32.5, x1=-10, y1=37.5,
        line=dict(color="black"),
    )
    fig.add_hline(y=70, line_color="green")
    fig.add_hline(y=0, line_color="green")
    fig.add_hline(y=65,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=65,   line_dash="dot", line_color="white")
    fig.add_hline(y=55,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=55,  line_dash="dot", line_color="white")
    fig.add_hline(y=5, line_dash="dashdot", line_color="green")
    fig.add_hline(y=5,  line_dash="dot", line_color="white")
    fig.add_hline(y=15,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=15, line_dash="dot", line_color="white")
    fig.add_vline(x=40,  line_dash="dash", line_color="green")
    fig.add_shape(type="rect",
        x0=0, y0=5, x1=-10, y1=15,
        line=dict(color="white"),
    )

    fig.add_shape(type="rect",
        x0=0, y0=55, x1=-10, y1=65,
        line=dict(color="white"),
    )


    fig.add_vline(x=0, line_color="green")
    fig.add_vline(x=-3, line_color="green")
    fig.add_vline(x=22, line_color="green")
    fig.add_vline(x=50, line_color="green")
    fig.add_vline(x=5,  line_dash="dash", line_color="green")


    fig.add_hline(y=[-3,0], line_width=3, line_color="black")
    fig.update_layout(yaxis_range=[0,70])
    fig.update_layout(xaxis_range=[-3,55])
    fig.update_layout(
        xaxis_title="Meters From Try Line",
        yaxis_title="Try Zone",
        font=dict(
            family="Times New Roman",
            size=15,
            color="#7f7f7f"
        )
    )
    fig.update_layout(
                      xaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [0,5,22,40,50], #change 2
                        ticktext = ['Try Line',5,22,40,50], #change 3
                        ),
                       font=dict(size=18, color="black"), 

    )
    fig.update_layout(
                      yaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [5,15,55,65], #change 2
                        ticktext = ['5', '15', '15', '5'], #change 3
                        ),

    )


    # Set templates
    fig.update_layout(template="plotly_white", )


    fig.show()
get_scatter_data(data)




if st.button('Calculate Probability '):
    st.write('Calculating')
    #start = time.time()
    #st.write("Started run time at: ", start)
    
    #import model from notebook 
    #print probability 
    #

    url = "http://www.goalkickers.co.za"
    response = requests.get(url)
    souph = BeautifulSoup(response.content, "html.parser")
    content =response.text
    soupl = BeautifulSoup(content, "lxml")
    