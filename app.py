import os
import sys

libraries_to_install = ["gdal", "fiona", "pyproj", "rtree", "shapely"]
whl_file_names = [
    "GDAL-3.4.1-cp39-cp39-win_amd64.whl",
    "Fiona-1.8.21-cp39-cp39-win_amd64.whl",
    "pyproj-3.2.1-cp39-cp39-win_amd64.whl",
    "Rtree-0.9.7-cp39-cp39-win_amd64.whl",
    "Shapely-1.8.1.post1-cp39-cp39-win_amd64.whl",
]
path_to_libraries_to_install = [
    os.path.join(r"C:\Users\IgorD\Downloads", filename) for filename in whl_file_names
]
dic_library_filename_path = dict(
    zip(libraries_to_install, path_to_libraries_to_install)
)
# check if the library folder already exists, to avoid building everytime you load the app
for library, library_path in dic_library_filename_path.items():
    # if not os.path.isdir(f"./venv/Lib/tmp/{library}"):
    if not os.path.isdir(f"/tmp/{library}"):
        # Read and save the already downloaded whl library file into our disk
        with open(library_path, "rb") as whl_file:
            with open(f"/tmp/{library}", "wb") as file:
                # with open(f'./venv/Lib/tmp/{library}','wb') as file:
                response = whl_file.read()
                file.write(response)
        # get our current dir, to configure it back again. Just house keeping
        default_cwd = os.getcwd()
        os.chdir(r"/tmp")
        # build
        os.system("./configure --prefix=/home/appuser")
        os.system("make")
        # install
        os.system("make install")
        # install python package
        os.system(
            f'pip3 install --global-option=build_ext --global-option="-L/home/appuser/lib/" --global-option="-I/home/appuser/include/" {library}'
        )
        # back to the cwd
        os.chdir(default_cwd)
        print(os.getcwd())
        sys.stdout.flush()
# add the library to our current environment
from ctypes import *

lib0, lib1, lib2, lib3, lib4 = None, None, None, None, None
for idx, library_name in enumerate(dic_library_filename_path):
    # path = os.path.join("/home/appuser/lib", library_name, '.so.0')
    path = f"/home/appuser/lib/lib{library_name}.so.0"
    if idx == 0:
        lib0 = CDLL(path)
    elif idx == 1:
        lib1 = CDLL(path)
    elif idx == 2:
        lib2 = CDLL(path)
    elif idx == 3:
        lib3 = CDLL(path)
    else:
        lib4 = CDLL(path)
import fiona
import pandas as pd
import pyproj
import rtree
import shapely
import streamlit as st
from osgeo import gdal

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
    raw_data = pd.read_csv(r"data\raw_data_Covid19.csv", encoding="ISO-8859-1")
    raw_data_Covid19 = raw_data.copy()
    raw_data_Covid19.set_index("Concelhos", inplace=True)
    # upload of the data with all the wanted indicators
    data_all_reformed = pd.read_csv(
        r"data\all_data_reformed.csv",
        encoding="UTF-8-SIG",
        usecols=lambda c: not c.startswith("Unnamed:"),
    )
    all_data = data_all_reformed.copy()
    # just because the incidences column calculation sometimes returns numbers full of decimals
    all_data.round(4)
    all_data.set_index("Concelhos", inplace=True)
    # importing the necessary incidence_14_days data
    Incidences_14days = pd.read_csv(r"data\inc_14_fixed.csv", encoding="utf-8")
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
