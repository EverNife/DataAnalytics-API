import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

from myownapi.MainAPI import MainAPI

class AnalyticsProphet(MainAPI):

    # ---------------------------------------------------------------------------
    #   Prophet
    # ---------------------------------------------------------------------------

    nomeDaColunaObjetivo = None;
    nomeDaColunaDeDatas = None;
    calculationDF : pd.DataFrame;
    model : Prophet;

    def definirColunaObjetivo(self, nomeDaColunaObjetivo, nomeDaColunaDeDatas, funcaoDeConversaDeDatas=None):
        self.nomeDaColunaObjetivo = nomeDaColunaObjetivo;
        self.nomeDaColunaDeDatas = nomeDaColunaDeDatas;

        if funcaoDeConversaDeDatas is not None:
            self.df[nomeDaColunaDeDatas] = self.df[nomeDaColunaDeDatas].apply(funcaoDeConversaDeDatas);
        else:
            self.df[nomeDaColunaDeDatas] = pd.to_datetime(self.df[nomeDaColunaDeDatas])

        self.df = self.df.groupby(nomeDaColunaDeDatas)[nomeDaColunaObjetivo].sum().reset_index()
        self.df = self.df.set_index(nomeDaColunaDeDatas)

        self.calculationDF = self.df.copy(deep=True);
        self.calculationDF.rename(columns={'deaths': 'y'}, inplace=True) #Prophet obriga a ser chamado Y
        self.calculationDF['ds'] = self.calculationDF.index                   #Prophet obriga a ser chamado ds

    #Atenção, o dataFrame precisa estar AGRUPADO e ORDENADO
    def plotarDecomposicao(self, dataFrame=None, theModel='addtive', theFigsize = None, theFreq = None):
        from pylab import rcParams;
        import statsmodels.api as sm;
        rcParams['figure.figsize'] = 18, 8;

        if dataFrame is None:
            dataFrame = self.df;

        # Dois tipos de modelos possíves, Aditivo e Multiplicativo (Necessário testar as diferenças)
        if theFreq is not None:
            decomposicao = sm.tsa.seasonal_decompose(dataFrame, model=theModel, freq=theFreq);
        else:
            decomposicao = sm.tsa.seasonal_decompose(dataFrame, model=theModel);
        if theFigsize is not None:
            plt.figure(figsize=theFigsize)
        decomposicao.plot();


    def aplicarPROPHET(self):
        self.model = Prophet();
        self.model.fit(self.calculationDF);

    def PROPHETForecast(self, steps):
        future_df = self.model.make_future_dataframe(periods=steps)
        forecast_df = self.model.predict(future_df)
        return forecast_df;

    def plotarForecast(self, steps):
        import plotly.offline as py
        from fbprophet.plot import plot_plotly
        py.init_notebook_mode()
        fig = plot_plotly(self.model, self.PROPHETForecast(steps));
        py.iplot(fig)