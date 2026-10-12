import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf



class DataExplorationVisualisation():

    """
    Visualising the data before forecasting it.
    """

    def __init__(self, features: pd.DataFrame):

        """
        Constructor / Initialising a "DataExplorationVisualisation" object.

        :param features: The pandas DataFrame of features being visualised.
        """

        self.features = features



    def acf_vis(self):

        """
        Visualising the ACF values for a specific feature column.
        """

        feature = "log_return"
        df = pd.DataFrame(self.features[feature].copy())

        acf_plot = plot_acf(df[feature], lags=100)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/ACF x PACF/acf_plot.png")
        # plt.show()

        acf_vals = acf(df[feature])
        num_lags = 20
        plt.bar(range(num_lags), acf_vals[:num_lags])
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/ACF x PACF/acf_plot2.png")
        # plt.show()




    def pacf_vis(self):

        """
        Visualising the PACF values for our specific feature column.
        """

        feature = "log_return"
        df = pd.DataFrame(self.features[feature].copy())

        pacf_plot = plot_pacf(df[feature], lags=100)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/ACF x PACF/pacf_plot.png")
        # plt.show()

        pacf_vals = pacf(df[feature])
        num_lags = 20
        plt.bar(range(num_lags), pacf_vals[:num_lags])
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/ACF x PACF/pacf_plot2.png")
        # plt.show()