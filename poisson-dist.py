# import streamlit
# check if it is working
import streamlit
from scipy.stats import poisson
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import math
import pandas as pd


mu = streamlit.number_input("Insert the value of mu", step=1.0, min_value=1.0)
k = streamlit.number_input("Insert the value of K", step=1.0, min_value=0.0)






if mu and k:
    dictionary = {"k" : [] , "pmf" : [], "cdf" : [], "sf" : []}


    for i in range(1, 2 * int(mu) + 1):
        dictionary["k"].append(i)
        dictionary['pmf'].append(poisson.pmf(i, mu))
        dictionary['cdf'].append(poisson.cdf(i, mu))
        dictionary['sf'].append(poisson.sf(i, mu))

    df = pd.DataFrame(dictionary)
    def highlight_k(x):
        if x.k == k:
            return ['background-color: yellow']*4
        else:
            return ['background-color: white'] * 4

    func = streamlit.radio("Choose the function: ", ("pmf", "cdf", "sf"))
    fig = px.line(df, x="k", y=func)
    streamlit.plotly_chart(fig)     
    
    df = df.style.apply(highlight_k, axis=1)

    streamlit.write(df)
    
    
    
