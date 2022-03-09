# Import of all the necessary libraries for the work
import logging
import pickle

import geopandas as gpd

# libraries to save and load the SOM trained models
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

logging.basicConfig(level="INFO")

mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
"""
Auxiliary Function to save and load objects using the pickle module
"""


def load_obj(name):
    with open("objetos/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


"""
import all the data necessary for the plotting
"""


@st.cache
def processing_all_needed_data(all_data, altura_do_ano):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    # data_incidences a ter a info das incidencias e other_data dos
    # outras features de todos os concelhos nesta epoca de tempo
    data_incidences_df = all_data.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    the_other_data = all_data.iloc[
        :, all_data.columns.get_loc("2021-02-06") + 1 : all_data.shape[1]
    ].copy()
    all_data_needed = pd.merge(
        data_incidences_df, the_other_data, how="inner", on="Concelhos"
    )
    # data Standardization(mean of 0, standard deviation of 1) before applying SOM
    all_data_needed_st = (all_data_needed - np.mean(all_data_needed, axis=0)) / np.std(
        all_data_needed, axis=0
    )
    return all_data_needed_st


"""
import the incidence data necessary for the plotting
"""


@st.cache
def processing_incidence_needed_data(Data_incidences, altura_do_ano):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    # Data standardization and preparation with respect to the time of the year chosen
    Data_incidences_standardized = (
        (Data_incidences - np.mean(Data_incidences, axis=0))
        / np.std(Data_incidences, axis=0)
    ).copy()
    Data_incidences_needed = Data_incidences_standardized.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    return Data_incidences_needed


"""
import the geodataframes necessary for the plotting
"""


@st.cache
def importing_lat_long_concelho_data():
    # districts_format = gpd.read_file('districts_shape_file\districts_format.shp')
    concelhos_format = gpd.read_file(
        r"Geographic_Info/concelhos_shape_file/concelhos_format.shp"
    )
    concelhos_format.set_index("Concelho", inplace=True)
    # to avoid a keyerror from the mismatch between Ponte de Sor as it s in the clustering_groups and
    # Ponte de Sôr in the concelhos_format index
    concelhos_format.index = concelhos_format.index.str.replace(
        "Ponte de Sôr", "Ponte de Sor"
    )
    concelhos_lat_long_geo_data = gpd.read_file(
        r"Geographic_Info/concelhos_lat_long_shapefile/concelhos_lat_long.shp"
    )
    concelhos_lat_long_geo_data.set_index("Concelho", inplace=True)
    return concelhos_format, concelhos_lat_long_geo_data


"""
Generic Functions that allow at any stage to have the SOM_clustering_map and SOM_clustering_grid
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def SOM_clustering_grid(all_needed_data, altura_do_ano, colors_per_neuron):
    data = all_needed_data.values
    # load and test the saved  trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    # calculo de todos as posiçoes dos winning neurons para cada um dos data inputs
    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x, w_y = np.array(w_x), np.array(w_y)
    # colocar como background Matriz U de distancia, building the graphic part
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.set_title(
        f"SOM Clustering of Concelhos in {altura_do_ano}",
        fontweight="bold",
        fontsize=15,
        y=1.03,
    )
    # plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.9)
    im = plt.imshow(
        som.distance_map(),
        cmap="bone_r",
        alpha=0.9,
        origin="lower",
        aspect="equal",
        extent=[-1, 4, -1, 4],
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    label_names = all_needed_data.index
    # fazer plot da posição ocupada pelo winning neuron de cada um dos data inputs,cada um dos concelhos
    # Em cada ciclo ocorre o plot de cada concelho # é adicionado um numero random, para que não acham concelhos
    # a coincidir mesmo quando sao mapeados para o mesmo winning neuron
    for indice_concelho, concelho_input in enumerate(label_names):
        # plt.scatter(w_x[indice_concelho]+np.random.uniform(-0.65,0)-0.2,
        # w_y[indice_concelho]+np.random.uniform(-0.65,0)-0.2,
        # s=90, color=dic_colors_concelhos[concelho_input], label=concelho_input)
        plt.scatter(
            w_x[indice_concelho] + np.random.uniform(-0.65, 0) - 0.2,
            w_y[indice_concelho] + np.random.uniform(-0.65, 0) - 0.2,
            s=30,
            color=colors_per_neuron[(w_x[indice_concelho], w_y[indice_concelho])],
        )
    n_neurons, m_neurons = 5, 5
    plt.xticks(np.arange(-0.5, n_neurons - 0.5, 1), np.arange(5), fontsize=15)
    plt.yticks(np.arange(-0.5, m_neurons - 0.5, 1), np.arange(5), fontsize=15)
    plt.xlabel("N Neuron", fontsize=15, labelpad=5)
    plt.ylabel("M Neuron", fontsize=15, labelpad=5)
    # plt.legend(loc='upper right', ncol=4,bbox_to_anchor=(2.3, 1.22),fontsize=14, markerscale=2,borderpad=2)
    # plt.savefig(f'Som clustering 5x5 grid in {altura_do_ano}.png',dpi=100, bbox_inches = 'tight')
    # plt.show()
    # st.pyplot(fig)  # to be able to display this image in the streamlit web app
    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def SOM_clustering_map(all_needed_data, altura_do_ano, colors_per_neuron):
    """
    Auxiliar Funtion- allows to use the ax.fill function to color the region defined by the concelho being considered
    """

    def getPolyCoords(gp_concelhos_format, concelho, coord_type):
        """Returns the coordinates ('x|y') of edges/vertices of a Polygon or 'MultiPolygon geometry
        given by a specific concelho, allows to use the ax.fill function to color the region defined by the
        concelho being considered'"""
        # Parse the geometries and grab the coordinate
        concelho_geometry = gp_concelhos_format.loc[concelho].geometry
        if concelho_geometry.type == "Polygon":
            if coord_type == "x":
                # Get the x coordinates of the points building the exterior format of the polygon
                return list(concelho_geometry.exterior.coords.xy[0])
            elif coord_type == "y":
                # Get the y coordinates of the exterior
                return list(concelho_geometry.exterior.coords.xy[1])
        elif concelho_geometry.type == "MultiPolygon":
            all_xy = []
            for polygon in concelho_geometry.geoms:
                if coord_type == "x":
                    all_xy.append(list(polygon.exterior.coords.xy[0]))
                elif coord_type == "y":
                    all_xy.append(list(polygon.exterior.coords.xy[1]))
            return all_xy

    # importing all the necessary data standardized and processed with respect to the time of the year chosen
    data = all_needed_data.values
    # load and test the saved  trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    # calculo dos grupos concelhos que foram mapeados juntamente ao mesmo neuron
    dic_neuron_labels = som.labels_map(data, all_needed_data.index)
    # building the clustering_groups respecting the order in which the neurons appear in the colors_per_neuron
    # in order to guarantee that each neuron has really just one color in all of the stages
    clustering_groups = []
    for neuron_coordinates in list(colors_per_neuron.keys()):
        if neuron_coordinates in list(
            dic_neuron_labels.keys()
        ):  # it is possible to exist neurons without any concelhos mapped to it
            clustering_groups += [
                (list(dict(dic_neuron_labels)[neuron_coordinates].keys()))
            ]
        else:
            clustering_groups += [
                "0 concelhos mapped to this neuron"
            ]  # this way, the color of this neuron does not appear
    # build a dic_colors to atribute one distictive color to each group of "concelhos" mapped to the same neuron
    dic_colors_concelhos_per_neuron = {
        tuple(concelhos): color
        for concelhos, color in zip(clustering_groups, list(colors_per_neuron.values()))
    }
    # import the geodataframes necessary for the plotting
    concelhos_format, concelhos_lat_long_geo_data = importing_lat_long_concelho_data()
    # plotting each group of concelhos mapped to the same neuron with the same color
    fig = plt.figure(figsize=(10, 15))
    ax = fig.gca()
    ax.set_title(
        f"SOM Clustering Map of Concelhos in {altura_do_ano}",
        fontweight="bold",
        fontsize=20,
        y=0.99,
    )
    # Plotting Portugal and its district borders or concelhos borders
    # districts_format.plot(ax = ax, edgecolor='black', color='white')
    concelhos_format.plot(ax=ax, edgecolor="black", color="white")
    for concelhos in clustering_groups:
        if concelhos != "0 concelhos mapped to this neuron":
            for concelho in concelhos:
                # plots each of the concelhos according to their lat,lon and color respecting the neuron they were mapped
                # ax.scatter(concelhos_lat_long_geo_data.loc[concelho].geometry.x,concelhos_lat_long_geo_data.loc[concelho].geometry.y, color = dic_colors_concelhos_per_neuron[tuple(concelhos)], s=40)
                # ploting and filling each concelhos format into the desired color of the neuron to which the concelho was mapped
                if concelhos_format.loc[concelho].geometry.type == "Polygon":
                    ax.fill(
                        getPolyCoords(concelhos_format, concelho, "x"),
                        getPolyCoords(concelhos_format, concelho, "y"),
                        color=dic_colors_concelhos_per_neuron[tuple(concelhos)],
                    )
                # otherwise we have to fill all the polygons present in the concelhos geometry
                else:
                    for polygon_n in range(
                        len(getPolyCoords(concelhos_format, concelho, "x"))
                    ):
                        ax.fill(
                            getPolyCoords(concelhos_format, concelho, "x")[polygon_n],
                            getPolyCoords(concelhos_format, concelho, "y")[polygon_n],
                            color=dic_colors_concelhos_per_neuron[tuple(concelhos)],
                        )
    ax.set_xlabel("Longitude", fontsize=15, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=15, labelpad=5)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.legend(bbox_to_anchor=(1.50, 1.01),loc = "upper right",fontsize=8)
    # plt.savefig(f'SOM Clustering Map of Concelhos in {altura_do_ano} Filled',dpi=100, bbox_inches = 'tight')
    # plt.show()
    # st.pyplot(fig)  # to be able to display this image in the streamlit web app
    return fig


"""
Function that plots incidences for each of the neurons in one of the specific times
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_raw_incidências_per_neuron(
    raw_data_Covid19, data_incidences, altura_do_ano, colors_per_neuron
):
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    # importing the necessary incidence data standardized and processed with respect to the time of the year chosen
    data = data_incidences.values
    # load and test the saved trained SOM
    som = joblib.load(f"Trained_Models_Incidences/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data, data_incidences.index)
    # contrução de data_incidences por cada grupo de concelhos mapped to the same neuron(de cada key do colors_per_neuron),
    # com base na raw data de Covid inicial
    # fazer plot de cada um dos dfs criados em cada uma das iterações do ciclo for
    fig = plt.figure(figsize=(100, 60))
    ax = fig.gca()
    ax.set_title(
        f"Incidences of Group of Concelhos Mapped to each Neuron,in the {altura_do_ano} period",
        fontweight="bold",
        fontsize=90,
    )
    for neuron_coordinates in list(
        colors_per_neuron.keys()
    ):  # (specific neuron coordinates): [specific color]
        concelhos = list(
            dict(dic_neuron_labels)[neuron_coordinates].keys()
        )  # list of concelhos mapped to the neuron
        concelhos_group = [
            "-".join(concelhos)
        ]  # just to join all the concelhos mapped in the index of the on going df being created at each
        # iteration of the for cycle
        Covid19_filtrada = raw_data_Covid19.loc[concelhos, :].copy()
        data_incidences = pd.DataFrame(
            concelhos_group, columns=["Concelhos"]
        ).set_index(
            "Concelhos"
        )  # df being always created
        n_firstday = Covid19_filtrada.columns.get_loc(stages[altura_do_ano][0])
        n_lastday = Covid19_filtrada.columns.get_loc(stages[altura_do_ano][1])
        for firstday, lastday in zip(
            range(n_firstday - 13, n_lastday + 1, 1),
            range(n_firstday, n_lastday + 1, 1),
        ):
            Total_Cases = (
                Covid19_filtrada.loc[
                    concelhos,
                    Covid19_filtrada.columns[firstday] : Covid19_filtrada.columns[
                        lastday
                    ],
                ]
                .sum(axis=1)
                .sum()
            )
            data_incidences[f"{Covid19_filtrada.columns[lastday]}"] = (
                ((Total_Cases / Covid19_filtrada["Populacao"].sum()) * 100000)
            ).round(decimals=4)
        ax.plot(
            data_incidences.columns,
            data_incidences.loc[concelhos_group].values.tolist()[0],
            color=colors_per_neuron[neuron_coordinates],
            label=[neuron_coordinates],
            linewidth=15,
        )
    # contrução de df correspondente à Incidência Nacional com base na raw data de Covid inicial
    # plot da incidência nacional
    Covid19 = raw_data_Covid19.copy()
    national_incidences = pd.DataFrame(
        ["Incidência Nacional"], columns=["Concelhos"]
    ).set_index("Concelhos")
    n_firstday = Covid19.columns.get_loc(stages[altura_do_ano][0])
    n_lastday = Covid19.columns.get_loc(stages[altura_do_ano][1])
    for firstday, lastday in zip(
        range(n_firstday - 13, n_lastday + 1, 1), range(n_firstday, n_lastday + 1, 1)
    ):
        Nacional_Cases = (
            Covid19.loc[:, Covid19.columns[firstday] : Covid19.columns[lastday]]
            .sum(axis=1)
            .sum()
        )
        national_incidences[f"{Covid19.columns[lastday]}"] = (
            ((Nacional_Cases / Covid19["Populacao"].sum()) * 100000)
        ).round(decimals=4)
    ax.plot(
        national_incidences.columns,
        national_incidences.loc["Incidência Nacional"].values.tolist(),
        color="black",
        label="Incidência Nacional",
        linewidth=40,
    )
    # Plot das Linhas de Risco Definidas pela DGS
    ax.plot(
        data_incidences.columns,
        [240] * data_incidences.columns.shape[0],
        color="palegreen",
        linewidth=30,
        label="Risco Moderado",
    )
    ax.plot(
        data_incidences.columns,
        [480] * data_incidences.columns.shape[0],
        color="gold",
        linewidth=30,
        label="Risco Elevado",
    )
    ax.plot(
        data_incidences.columns,
        [960] * data_incidences.columns.shape[0],
        color="darkorange",
        linewidth=30,
        label="Risco Muito Elevado",
    )
    ax.set_xlabel("Days", fontsize=80)
    ax.set_ylabel("Cum.Incidence 14Days per 100 thousand inhabitants", fontsize=80)
    # timearray=np.arange('2020-03-28', '2020-06-01',np.timedelta64(5,'D'), dtype='datetime64')
    timearray = data_incidences.columns[::5]  # tick time intervals in the x axis
    plt.xticks(timearray, fontsize=50)
    plt.yticks(fontsize=50)
    ax.tick_params(axis="both", direction="inout", length=60, width=10, color="black")
    # plt.legend(bbox_to_anchor=(1.5, 1.5))
    plt.legend(loc="upper left", fontsize=60)
    plt.grid()
    # plt.savefig(f'Incidences of Group of Concelhos Mapped to each Neuron ,in the {altura_do_ano} period.png',dpi=100,bbox_inches = 'tight')
    # plt.show()
    return fig


"""
Function which plots the geospatial pattern of neurons Above Average in any of the critical times
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_concelhos_in_risk_neurons(all_needed_data, altura_do_ano, colors_per_neuron):
    """
    Auxiliar Funtion- allows to use the ax.fill function to color the region defined by the concelho being considered
    """

    def getPolyCoords(gp_concelhos_format, concelho, coord_type):
        """Returns the coordinates ('x|y') of edges/vertices of a Polygon or 'MultiPolygon geometry
        given by a specific concelho, allows to use the ax.fill function to color the region defined by the
        concelho being considered'"""
        # Parse the geometries and grab the coordinate
        concelho_geometry = gp_concelhos_format.loc[concelho].geometry
        if concelho_geometry.type == "Polygon":
            if coord_type == "x":
                # Get the x coordinates of the points building the exterior format of the polygon
                return list(concelho_geometry.exterior.coords.xy[0])
            elif coord_type == "y":
                # Get the y coordinates of the exterior
                return list(concelho_geometry.exterior.coords.xy[1])
        elif concelho_geometry.type == "MultiPolygon":
            all_xy = []
            for polygon in concelho_geometry.geoms:
                if coord_type == "x":
                    all_xy.append(list(polygon.exterior.coords.xy[0]))
                elif coord_type == "y":
                    all_xy.append(list(polygon.exterior.coords.xy[1]))
            return all_xy

    # importing all the necessary data standardized and processed with respect to the time of the year chosen
    data = all_needed_data.values
    # load and test the saved trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data, all_needed_data.index)
    dic_neuron_concelhos = {}
    # importing the saved dic_neurons_heat_map
    dic_neurons_above_average = load_obj("dic_neurons_above_average")
    # to build a dic with (each neuron):[list of concelhos mapped to it]
    for neuron_coordinates in dic_neurons_above_average[altura_do_ano]:
        if neuron_coordinates in list(dic_neuron_labels.keys()):
            dic_neuron_concelhos[neuron_coordinates] = list(
                dict(dic_neuron_labels)[neuron_coordinates].keys()
            )
    # import the geodataframes necessary for the plotting
    concelhos_format, concelhos_lat_long_geo_data = importing_lat_long_concelho_data()
    # start making the desired map plot
    fig = plt.figure(figsize=(10, 15))
    ax = fig.gca()
    ax.set_title(
        f"Concelhos in High Risk Incidence Neurons in {altura_do_ano}",
        fontsize=20,
        fontweight="bold",
        y=1,
        loc="center",
    )
    # Plotting Portugal and its district borders or concelhos borders
    concelhos_format.plot(ax=ax, edgecolor="black", color="white")
    # districts_format.plot(ax = ax, edgecolor='black', color='white')
    # plot de concelhos consoante as suas coordenadas e da cor específica do neurónio onde foram mapeados, i.e tendo em conta qual a key deles no
    # dic_neuron_concelhos, iterar o dic_neuron_concelhos e fazer plot dos concelhos em cada uma das suas keys com a mesma cor, cor específica
    # do neurónio que corresponde a key
    for neuron_coordinates in list(dic_neuron_concelhos.keys()):
        concelhos = dic_neuron_concelhos[
            neuron_coordinates
        ]  # lista de concelhos mapeados nesse neurónio and in high or low risk
        for concelho in concelhos:
            # ploting and filling each concelhos format into the desired color of the neuron to which the concelho was mapped
            if concelhos_format.loc[concelho].geometry.type == "Polygon":
                ax.fill(
                    getPolyCoords(concelhos_format, concelho, "x"),
                    getPolyCoords(concelhos_format, concelho, "y"),
                    color=colors_per_neuron[neuron_coordinates],
                )
            # otherwise we have to fill all the polygons present in the concelhos geometry
            else:
                for polygon_n in range(
                    len(getPolyCoords(concelhos_format, concelho, "x"))
                ):
                    ax.fill(
                        getPolyCoords(concelhos_format, concelho, "x")[polygon_n],
                        getPolyCoords(concelhos_format, concelho, "y")[polygon_n],
                        color=colors_per_neuron[neuron_coordinates],
                    )
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    ax.set_xlabel("Longitude", fontsize=15, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=15, labelpad=5)
    # plt.legend(bbox_to_anchor=(1.35, 0.75),loc = "upper right",fontsize=13 )
    # plt.savefig(f'Concelhos in High Risk Incidence Neurons in {altura_do_ano} Filled',dpi=100,bbox_inches = 'tight')
    # plt.show()
    return fig


"""
Function capable of computing the heat_map for all the features in the other data, for all the concelhos mapped to each of 
the high and low risk incidence neurons
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def additional_features_heat_map(
    all_needed_data, altura_do_ano, dic_neurons_heat_map, abreviation_feature_names_list
):
    data = all_needed_data.values
    # load and test the saved trained SOM
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data, all_needed_data.index)
    dic_neuron_concelhos = (
        {}
    )  # #to build a dic with (each neuron):[list of concelhos mapped to it]
    for neuron_coordinates in dic_neurons_heat_map[altura_do_ano]:
        if neuron_coordinates in list(
            dic_neuron_labels.keys()
        ):  # just to check if they were indeed mappings in this neuron of low and high risk
            dic_neuron_concelhos[neuron_coordinates] = list(
                dict(dic_neuron_labels)[neuron_coordinates].keys()
            )
    # making of the data for the heat map, compute the mean for each of the features of the concelhos mapped to each neuron
    data_heat_map = []
    for neuron_coordinates in list(dic_neuron_concelhos.keys()):
        concelhos = dic_neuron_concelhos[
            neuron_coordinates
        ]  # lista de concelhos mapeados nesse neurónio
        data_heat_map += [
            [neuron_coordinates]
            + list(np.mean(all_needed_data.loc[concelhos, "dens_pop":].values, axis=0))
        ]
    heat_map_df = pd.DataFrame(
        columns=["Neurons"] + abreviation_feature_names_list, data=data_heat_map
    ).set_index("Neurons")
    # Make the Heat Map
    # make the heat_map
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title(
        f"Average Feature Values of the Highest \n and Lowest 6 Risk Incidence Neurons in the {altura_do_ano}",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    # above its principal diagonal
    cmap = sns.diverging_palette(
        230, 20, as_cmap=True
    )  # just the color padron that will be applied in the heat map
    heat_map = sns.heatmap(
        data=heat_map_df,
        cmap=cmap,
        vmax=heat_map_df.max().max() + 0.1,
        vmin=heat_map_df.min().min() - 0.1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.5},
    )
    # heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    # heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0)
    plt.xlabel("Socioeconomic and Demographic Features", fontsize=14, labelpad=13)
    plt.ylabel(
        "6 Low Risk Incidence Neurons                  6 High Risk Incidence Neurons",
        fontsize=14,
        labelpad=8,
    )
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.savefig(f'Additional Features Heat Map in {altura_do_ano}.png',dpi=100,bbox_inches = 'tight')
    # plt.show()
    return fig


"""
Function which generates a heat map with the correlation of the additional features between eac hother
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def correlation_between_additional_features_heat_map(
    all_data, abreviation_feature_names_list
):
    # All Data standardization
    all_data_standardized = (
        (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
    ).copy()
    # slice only the wanted data from the all_data_standardized
    the_other_data_standardized = all_data_standardized.iloc[
        :,
        all_data_standardized.columns.get_loc("2021-02-06")
        + 1 : all_data_standardized.shape[1],
    ].copy()
    the_other_data_standardized.columns = abreviation_feature_names_list
    corr = the_other_data_standardized.corr().copy()
    # make the heat_map
    figure, ax = plt.subplots(figsize=(11, 9))
    plt.title(f"Additional Features Correlation Among Themselves", fontsize=15)
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )  # good to apply in order to not reapeat the same numbers, return matrix full of 0
    # above its principal diagonal
    cmap = sns.diverging_palette(
        230, 20, as_cmap=True
    )  # just the color padron that will be applied in the heat map
    heat_map = sns.heatmap(
        data=corr.iloc[1:, :-1],
        mask=mask[1:, :-1],
        cmap=cmap,
        vmax=corr.iloc[1:, :-1].max().max() + 0.1,
        vmin=corr.iloc[1:, :-1].min().min() - 0.1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.5},
    )
    # heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    # plt.show()
    return figure
