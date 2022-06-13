import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import plotly.express as px

from numpy import e

import streamlit.components.v1 as components


import setuptools

import time

import datetime
from datetime import datetime

df=pd.read_csv('https://raw.githubusercontent.com/subham27-07/Studio-LNGR/main/sensors_output2.csv')

st.title ("AMPS system Dashboard")
# st.sidebar.title("Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used to AMPS system")
# st.sidebar.markdown("This application is a Strea

st.write(df)


    
pd.options.plotting.backend = "plotly"
df.plot(x='Date', y=[ 'Temperature1','Humidity1','Temperature2','Humidity2','Soil Moisture','VOC','CO2'])
df_melt = df.melt(id_vars='Date', value_vars=['Temperature1','Humidity1','Temperature2','Humidity2','Soil Moisture','VOC','CO2'])
fig=px.line(df_melt, x='Date' , y='value' , color='variable',width=1200,height=500)
    
st.plotly_chart(fig)
    






fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Temperature1'],
                    mode='lines+markers',
                    name='Temperature1'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Humidity1'],
                    mode='lines+markers',
                    name='Humidity1'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Temperature2'],
                    mode='lines+markers',
                    name='Temperature2'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Humidity2'],
                    mode='lines+markers',
                    name='Humidity2'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Soil Moisture'],
                    mode='lines+markers',
                    name='Soil Moisture'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['VOC'],
                    mode='lines+markers',
                    name='VOC'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['CO2'],
                    mode='lines+markers',
                    name='CO2'))



st.write(fig)
