import streamlit as st

from functions_explore_page import (
    SOM_clustering_grid,
    additional_features_heat_map,
    correlation_between_additional_features_heat_map,
    load_obj,
    plot_concelhos_in_risk_neurons,
    plot_raw_incidências_per_neuron,
    processing_all_needed_data,
    processing_incidence_needed_data,
)


def show_explore_data_page(raw_data_Covid19, Data_incidences, all_data):
    # create a title app
    st.image(r"Images/covid_19_logo.png", width=200)
    st.title("Covid-19 Spacial and Temporal SOM Clustering Analysis")
    # add some text to the app taking advantage of markup language
    st.sidebar.markdown("____")
    st.sidebar.markdown(
        """
    ## __*Want to Explore the Covid-19 Evolution in what Timeframe and Using which Clustering Model?*__
    """
    )
    # create a selectorbox as a sidebar for the user in the app to use, to choose one of the 5 time intervals defined
    time_frame = st.selectbox(
        "Select Time Period",
        (
            "1st Emergency State",
            "Verao",
            "September-October 2020",
            "2 Wave of Covid19",
            "Explosão Fim de Ano",
        ),
    )
    # we also want to have different classifier options for the user to select, so we just resuse the code of creating a selectorbox
    clustering_model_name = st.selectbox("Select Clustering Model", ("SOM", "KMeans"))

    # importing all the necessary data
    all_needed_data = processing_all_needed_data(all_data, time_frame)
    dic_colors_per_neuron = load_obj("dic_colors_per_neuron")
    dic_neurons_heat_map = load_obj("dic_neurons_heat_map")
    Data_incidences = processing_incidence_needed_data(Data_incidences, time_frame)
    abreviation_feature_names = [
        "Pop_Density",
        "Deprivation_Index",
        "Youth_Pop",
        "Eldery_Pop",
        "Primary_sector",
        "Secondary_sector",
        "Tertiary_sector",
        "State_Benefits",
        "Schools_km2",
    ]
    # the plots are beggining to be generated and display
    # first will be display the som clustering grid side by side with the plot of concelhos mapped to risk neurons
    col1, col2 = st.columns((2.5, 2.5))
    with col1:
        # just to make the first grid be in the same horizontal line as the second figure
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.pyplot(
            SOM_clustering_grid(all_needed_data, time_frame, dic_colors_per_neuron)
        )
    with col2:
        st.pyplot(
            plot_concelhos_in_risk_neurons(
                all_needed_data, time_frame, dic_colors_per_neuron
            )
        )
    # now we will plot the scatter plot of incidences per neuron
    st.pyplot(
        plot_raw_incidências_per_neuron(
            raw_data_Covid19, Data_incidences, time_frame, dic_colors_per_neuron
        )
    )
    st.write("")
    # heat_map with the additional features correlations between each other
    user_wants_correlation_between_additional_features_heat_map = st.checkbox(
        "Display Additional Features Correlation Between Each Other"
    )
    if user_wants_correlation_between_additional_features_heat_map:
        st.pyplot(
            correlation_between_additional_features_heat_map(
                all_data, abreviation_feature_names
            )
        )

    st.write("")
    # followed by the heat_map with the additional features correlations with the low and high risk neurons
    user_wants_additional_features_heat_map = st.checkbox(
        "Display Additional Features Correlation with Respect with the Low and High Risk Neurons"
    )
    if user_wants_additional_features_heat_map:
        figure = additional_features_heat_map(
            all_needed_data, time_frame, dic_neurons_heat_map, abreviation_feature_names
        )
        st.pyplot(figure)
