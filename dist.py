
import streamlit
from scipy.stats import poisson, geom, binom
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


distributions = streamlit.sidebar.selectbox(
    "Select the distribution",
    ("Binomial Distribution", "Poisson Distribution", "Geometric Distribution")
)






def poisson_dist(mean):
    mean = mean
    dictionary = {"k" : [] , "pmf" : [], "cdf" : [], "sf" : []}


    for i in np.arange(0, mean * 2 + 1, 1):
        dictionary["k"].append(i)
        dictionary['pmf'].append(poisson.pmf(i, mean))
        dictionary['cdf'].append(poisson.cdf(i, mean))
        dictionary['sf'].append(poisson.sf(i, mean))

    df = pd.DataFrame(dictionary)
    df = df.round(decimals=4)
    # answers = df.iloc[int(k - 1)]
    # streamlit.write(f"P(X = {answers[0]}) is {answers[1]}")
    # streamlit.write(f"P(X <= {answers[0]}) is {answers[2]}")
    # streamlit.write(f"P(X > {answers[0]}) is {answers[3]}")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Probability of X = K", "Probability of X <= K", "Probability of X > K"))


    # fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['k'], y=df["pmf"],  name="P(X = k)",  fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["cdf"], name="P(X <= k)", fill='tozeroy'),row=2, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["sf"], name="P(X > k)", fill='tozeroy'),row=3, col=1)

    fig.update_layout(hovermode="closest", height=1000)

    streamlit.plotly_chart(fig)
    

   
def geom_dist(p):
    probability = p
    size = 30

    

    dictionary = {"k" : [] , "pmf" : [], "cdf" : [], "sf" : []}


    for i in np.arange(1, size, 1):
        dictionary["k"].append(i)
        dictionary['pmf'].append(geom.pmf(i, probability))
        dictionary['cdf'].append(geom.cdf(i, probability))
        dictionary['sf'].append(geom.sf(i, probability))

    df = pd.DataFrame(dictionary)
    # df = df.round(decimals=4)

    # answers = df.iloc[int(k - 1)]
    # streamlit.write(f"P(G = {answers[0]}) is {answers[1]}")
    # streamlit.write(f"P(G <= {answers[0]}) is {answers[2]}")
    # streamlit.write(f"P(G > {answers[0]}) is {answers[3]}")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Probability of G = K", "Probability of G <= K", "Probability of G > K"))


    # fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['k'], y=df["pmf"], name="P(G = k)",  fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["cdf"], name="P(G <= k)", fill='tozeroy'),row=2, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["sf"], name="P(G > k)", fill='tozeroy'),row=3, col=1)
    

    fig.update_layout(hovermode="closest", height=1000)

    streamlit.plotly_chart(fig)


def binomial_dist(p, n):
    
    size = n

    

    dictionary = {"k" : [] , "pmf" : [], "cdf" : [], "sf" : []}


    for i in np.arange(0, size + 1, 1):
        dictionary["k"].append(i)
        dictionary['pmf'].append(binom.pmf(i, n, p))
        dictionary['cdf'].append(binom.cdf(i,n , p))
        dictionary['sf'].append(binom.sf(i,n , p))

    df = pd.DataFrame(dictionary)
    # df = df.round(decimals=4)

    # answers = df.iloc[int(k - 1)]
    # streamlit.write(f"P(G = {answers[0]}) is {answers[1]}")
    # streamlit.write(f"P(G <= {answers[0]}) is {answers[2]}")
    # streamlit.write(f"P(G > {answers[0]}) is {answers[3]}")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Probability of X = K", "Probability of X <= K", "Probability of X > K"))


    # fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['k'], y=df["pmf"], name="P(X = k)",  fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["cdf"], name="P(X <= k)", fill='tozeroy'),row=2, col=1)
    fig.add_trace(go.Scatter(x=df['k'], y=df["sf"], name="P(X > k)", fill='tozeroy'),row=3, col=1)
    

    fig.update_layout(hovermode="closest", height=1000)

    streamlit.plotly_chart(fig)

    
   
if distributions == "Poisson Distribution":
    streamlit.title("Poisson Distribution")
    mu = streamlit.sidebar.number_input("Insert the value of mu", step=1.0, min_value=1.0)
    # k = streamlit.sidebar.slider("Insert the value of K", step=1.0, min_value=0.0, max_value=2 * mu)
    if mu:
        poisson_dist(mu)

if distributions == "Geometric Distribution":
    streamlit.title("Geometric Distribution")
    probability = streamlit.sidebar.number_input("Insert the probability",format="%f" , step=0.001, min_value=0.0, max_value=1.0)
    # k = streamlit.sidebar.number_input("Insert the value of K", step=1.0, min_value=0.0)
    if probability:
        geom_dist(probability)

if distributions == "Binomial Distribution":
    streamlit.title("Binomial Distribution")
    probability = streamlit.sidebar.number_input("Insert the probability", format="%f" , step=0.001, min_value=0.0, max_value=1.0)
    trials = streamlit.sidebar.number_input("Insert the number of trials", step=1.0, min_value=0.0)
    if probability:
        binomial_dist(probability, trials)





   
    
    
    
    
    

