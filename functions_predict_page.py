# Importe de Todas as Bibliotecas Necessárias
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from functions_explore_page import importing_lat_long_concelho_data


# auxiliary function used by new_input_assignment function
def contemQ(l1, el):
    if l1 == []:
        return False
    return next((True for element in l1 if element == el), False)


def neuron_assignment_total(all_data):
    # df com concelhos e neurons para onde cada concelho seria mapeado em cada uma das epocas críticas
    concelhos = all_data.index
    Concelhos_Clusters = pd.DataFrame(concelhos, columns=["Concelhos"])
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    for epoca in stages:
        # calculo de neuron de cada concelho, com data_incidences a ter a info das incidencias e other_data dos outras features
        # de todos os concelhos nesta epoca de tempo
        data_incidences_df = all_data.loc[:, stages[epoca][0] : stages[epoca][1]].copy()
        the_other_data = all_data.iloc[
            :, all_data.columns.get_loc("2021-02-06") + 1 : all_data.shape[1]
        ].copy()
        all_data_needed = pd.merge(
            data_incidences_df, the_other_data, how="inner", on="Concelhos"
        )
        # data Standardization(mean of 0, standard deviation of 1) before applying SOM
        data_filtrada = (all_data_needed - np.mean(all_data_needed, axis=0)) / np.std(
            all_data_needed, axis=0
        )
        data = data_filtrada.values
        # load and test the saved trained SOM
        som = joblib.load(f"Trained_Models/SOM_{epoca}")
        neurons_predicted = [som.winner(d) for d in data]
        Concelhos_Clusters[f" Cluster in {epoca} "] = neurons_predicted
    Concelhos_Clusters.set_index("Concelhos", inplace=True)
    return Concelhos_Clusters


""" 
Function which predicts the neuron in which the new input would be mapped in all of the timeframes
"""


def forecast_new_data_input_mappings(all_data, new_data_name):
    Concelho_Clusters = pd.DataFrame(["New_Input"], columns=["Concelho"]).set_index(
        "Concelho"
    )
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    for epoca in stages:
        # calculo de neuron para um concelho em específico, com new_data_input a ter a info das incidencias e da other data
        # desse nesta epoca de tempo
        # data Standardization(mean of 0, standard deviation of 1) before splicing the data and then applying SOM
        all_data_standardized = (
            (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
        ).copy()
        new_data_input = all_data_standardized.loc[new_data_name].copy()
        new_data_input_incidences = new_data_input.loc[
            stages[epoca][0] : stages[epoca][1]
        ].copy()
        new_data_input_other_data = new_data_input.loc["dens_pop":].copy()
        # new_data_input = new_data_input_incidences.append(
        # new_data_input_other_data
        # ).copy()
        new_data_input = pd.concat(
            [new_data_input_incidences, new_data_input_other_data]
        ).copy()
        new_data = new_data_input.values
        som = joblib.load(f"Trained_Models/SOM_{epoca}")
        neuron_predicted = som.winner(new_data)
        Concelho_Clusters[f" Cluster in {epoca} "] = [neuron_predicted]
    return Concelho_Clusters


""" 
Function which computes the concelhos that were mapped in the majority of time in the same manner as the new input would be mapped
"""


def new_input_assignment(all_data, Concelhos_Neurons_Over_Time, concelho_name):
    """
    Generic Function which tries to return the group of concelhos which were mapped to the same neurons as a
    new input(in this case the Lagoa Concelho) is predict to be mapped"""
    Concelho_Clusters = pd.DataFrame([concelho_name], columns=["Concelho"]).set_index(
        "Concelho"
    )
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )
    for epoca in stages:
        # calculo de neuron de cada concelho, com new_data_input a ter a info das incidencias e da other data
        # do concelho em específico nesta epoca de tempo
        # data Standardization(mean of 0, standard deviation of 1) before applying SOM
        all_data_standardized = (
            (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
        ).copy()
        # here the new_data_input corresponds to Lagoa, but ideally it would be other random concelho
        new_data_input = all_data_standardized.loc[concelho_name].copy()
        new_data_input_incidences = new_data_input.loc[
            stages[epoca][0] : stages[epoca][1]
        ].copy()
        new_data_input_other_data = new_data_input.loc["dens_pop":].copy()
        new_data_input = new_data_input_incidences.append(
            new_data_input_other_data
        ).copy()
        data = new_data_input.values
        som = joblib.load(f"Trained_Models/SOM_{epoca}")
        neuron_predicted = som.winner(data)
        Concelho_Clusters[f" Cluster in {epoca} "] = [neuron_predicted]
        # forming Concelho_Clusters_df with only one row, represeting the new concelho(new_input given to be mapped)
        # and then 5 columns, where are shown the neurons where it was predicted by SOM for the new concelho to be mapped
        # in each of 5 specific time periods
    cols = list(Concelhos_Neurons_Over_Time.columns)
    vals = list(Concelho_Clusters.iloc[0, :].values)  # values of the input concelho
    # goes searching in the Concelhos_Neurons_Over_Time table, for the concelhos(rows) with the same mappings as the new input given, for that
    # it is created a mask_df_input of False and Trues comparing each row in Concelhos_Neurons_Over_Time with the respective value in input
    mask_df_input = (Concelhos_Neurons_Over_Time[cols] == vals).copy()
    # sum all the rows values, above 4, it means has 4 or more True values, ie the same mappings in 4 of the 5 times
    concelho_equal_evolution = list(
        Concelhos_Neurons_Over_Time.loc[mask_df_input.sum(axis=1) >= 4].index
    )
    # remove the concelho itself given it has all True values, in an ideal scenario the concelho data given as input would not be present in the
    # Concelhos_Neurons_Over_Time and this step would be unnecessary
    if contemQ(concelho_equal_evolution, concelho_name):
        concelho_equal_evolution.remove(concelho_name)
    return concelho_equal_evolution


"""
Classifies a new input, by returning the neuron coords of the neuron in which it was mapped and the other concelhos already mapped to that neuron
"""


def classify(all_data, concelho, altura_do_ano):
    """Classifies the new_input to one of the neurons defined
    using the method labels_map.
    Returns a list of the "concelhos" assigned to the same neuron and the
    neurons coordinates
    """
    stages = dict(
        [
            ("1st Emergency State", ("2020-03-28", "2020-05-30")),
            ("Verao", ("2020-07-01", "2020-09-10")),
            ("September-October 2020", ("2020-09-01", "2020-10-30")),
            ("2 Wave of Covid19", ("2020-10-01", "2020-12-15")),
            ("Explosão Fim de Ano", ("2020-12-15", "2021-02-06")),
        ]
    )

    # data Standardization(mean of 0, standard deviation of 1) before applying SOM
    all_data_standardized = (
        (all_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)
    ).copy()
    # here the new_data_input corresponds to Lagoa, but ideally it would be other random concelho
    new_data_input = all_data_standardized.loc[concelho].copy()
    new_data_input_incidences = new_data_input.loc[
        stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    new_data_input_other_data = new_data_input.loc["dens_pop":].copy()
    new_data_input = new_data_input_incidences.append(new_data_input_other_data).copy()
    data = new_data_input.values
    # Calculo das restantes incidências para todos os outros concelhos e other data também
    data_incidences_df = all_data_standardized.loc[
        :, stages[altura_do_ano][0] : stages[altura_do_ano][1]
    ].copy()
    the_other_data = all_data_standardized.iloc[
        :,
        all_data_standardized.columns.get_loc("dens_pop") : all_data_standardized.shape[
            1
        ],
    ].copy()
    all_data_needed = pd.merge(
        data_incidences_df, the_other_data, how="inner", on="Concelhos"
    )
    data_SOM = all_data_needed.values
    som = joblib.load(f"Trained_Models/SOM_{altura_do_ano}")
    dic_neuron_labels = som.labels_map(data_SOM, all_data_needed.index)
    # form a dic with neuron_coordinate : [list of concelhos mapped to it]
    dic_neuron_concelhos = {}
    for neuron_coordinates in list(dic_neuron_labels.keys()):
        dic_neuron_concelhos[neuron_coordinates] = list(
            dict(dic_neuron_labels)[neuron_coordinates].keys()
        )
    coordinates_winning_neuron = som.winner(data)
    concelhos_mapped_to_same_neuron = []
    concelhos_mapped_to_same_neuron.append(
        dic_neuron_concelhos[coordinates_winning_neuron]
    )
    return coordinates_winning_neuron, concelhos_mapped_to_same_neuron


"""
Function which plots the concelhos that were mapped to the same neuron as the new input that the user wanted to classify
"""


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_concelhos_classified_together(
    concelhos_mapped_together, colors_per_neuron, neuron_coords
):
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

    # import the geodataframes necessary for the plotting
    # import the geodataframes necessary for the plotting
    concelhos_format, concelhos_lat_long_geo_data = importing_lat_long_concelho_data()
    # start making the desired map plot
    fig = plt.figure(figsize=(5, 10))
    ax = fig.gca()
    ax.set_title(
        "Concelhos Mapped in the Same Neuron",
        fontsize=10,
        fontweight="bold",
        y=1,
        loc="center",
    )
    # Plotting Portugal and its district borders or concelhos borders
    concelhos_format.plot(ax=ax, edgecolor="black", color="white")
    # districts_format.plot(ax = ax, edgecolor='black', color='white')
    # plot de concelhos consoante as suas coordenadas e da cor específica do neurónio onde foram mapeados
    for concelho in concelhos_mapped_together[0]:
        if concelhos_format.loc[concelho].geometry.type == "Polygon":
            ax.fill(
                getPolyCoords(concelhos_format, concelho, "x"),
                getPolyCoords(concelhos_format, concelho, "y"),
                color=colors_per_neuron[neuron_coords],
            )
        # otherwise we have to fill all the polygons present in the concelhos geometry
        else:
            for polygon_n in range(len(getPolyCoords(concelhos_format, concelho, "x"))):
                ax.fill(
                    getPolyCoords(concelhos_format, concelho, "x")[polygon_n],
                    getPolyCoords(concelhos_format, concelho, "y")[polygon_n],
                    color=colors_per_neuron[neuron_coords],
                )
    # Kill the spines...
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="inout", length=10, width=1, color="black")
    ax.set_xlabel("Longitude", fontsize=13, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=13, labelpad=5)
    # plt.legend(bbox_to_anchor=(1.35, 0.75),loc = "upper right",fontsize=13 )
    # plt.savefig(f'Concelhos in High Risk Incidence Neurons in {altura_do_ano} Filled',dpi=100,bbox_inches = 'tight')
    # plt.show()
    return fig
