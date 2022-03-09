import pandas as pd
import streamlit as st

from explore_data import show_explore_data_page
from predict_page import load_predict_page

st.set_page_config(
    page_title="Covid-19 App",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache
def load_all_necessary_data():
    # Upload of the Raw Data and then All_Data_Reformed for the SOM Analysis and also the lat_long_info of concelhos
    raw_data = pd.read_csv(r"data/raw_data_Covid19.csv", encoding="ISO-8859-1")
    raw_data_Covid19 = raw_data.copy()
    raw_data_Covid19.set_index("Concelhos", inplace=True)
    # upload of the data with all the wanted indicators
    data_all_reformed = pd.read_csv(
        r"data/all_data_reformed.csv",
        encoding="UTF-8-SIG",
        usecols=lambda c: not c.startswith("Unnamed:"),
    )
    all_data = data_all_reformed.copy()
    # just because the incidences column calculation sometimes returns numbers full of decimals
    all_data.round(4)
    all_data.set_index("Concelhos", inplace=True)
    # importing the necessary incidence_14_days data
    Incidences_14days = pd.read_csv(r"data/inc_14_fixed.csv", encoding="utf-8")
    inc_14days = Incidences_14days.copy()
    inc_14days.round(4)
    inc_14days.set_index("Concelhos", inplace=True)
    Data_incidences = inc_14days.copy()
    return raw_data_Covid19, all_data, Data_incidences


raw_data_Covid19, all_data, Data_incidences = load_all_necessary_data()


# now we just want to still make a sidebar and the other web app page, where we can explore all
# the data used for the model s training

# we can move any widge(button, selectbox, slider etc) to the sidebar by using it as a prefix
user_wanted_page = st.sidebar.selectbox(
    "Explore Covid-19 Data or Use SOM clustering as Warning System",
    ["Explore", "Warning System"],
)

if user_wanted_page == "Explore":
    show_explore_data_page(raw_data_Covid19, Data_incidences, all_data)
else:
    load_predict_page(all_data)
