import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from django.core.validators import URLValidator
from statsmodels.tsa.statespace.mlemodel import MLEResults, PredictionResultsWrapper

from myownapi.MainAPI import MainAPI

class AnalyticsARIMA(MainAPI):

    # ---------------------------------------------------------------------------
    #   DataFrame
    # ---------------------------------------------------------------------------



    # ---------------------------------------------------------------------------
    #   Arima
    # ---------------------------------------------------------------------------

    modelFit:MLEResults = None;
    nomeDaColunaObjetivo = None;
    nomeDaColunaDeDatas = None;

    def arimaDefinirColunaObjetivo(self, nomeDaColunaObjetivo, nomeDaColunaDeDatas, funcaoDeConversaDeDatas=None):
        self.nomeDaColunaObjetivo = nomeDaColunaObjetivo;
        self.nomeDaColunaDeDatas = nomeDaColunaDeDatas;

        if funcaoDeConversaDeDatas is not None:
            self.df[nomeDaColunaDeDatas] = self.df[nomeDaColunaDeDatas].apply(funcaoDeConversaDeDatas);
        else:
            self.df[nomeDaColunaDeDatas] = pd.to_datetime(self.df[nomeDaColunaDeDatas])

        self.df = self.df.groupby(nomeDaColunaDeDatas)[nomeDaColunaObjetivo].sum().reset_index()
        self.df = self.df.set_index(nomeDaColunaDeDatas)

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


    # ARIMA_SASONALIDADE == 12 meses, no caso, 1 ano;
    def aplicarARIMA(self, ARIMA_SASONALIDADE = 12, verbose = False):

        if verbose:
            print('\n# ==============================================================================================================');
            print('# Preparando quantidade de treino.');
            print('# ==============================================================================================================\n\n');

        import itertools;
        p = sazonalidade = range(0, 2);  # Arima P == auto-regressive part of the model
        d = tendencia = range(0, 2);  # Arima D == integrated part of the model
        q = ruido = range(0, 2);  # Arima Q == moving average part of the model

        # itertools.product basicamente relaciona todas as variáveis com todas as varíaveis... como já diz, PRODUCT
        pdq = list(itertools.product(sazonalidade, tendencia, ruido));

        # Criando agora as variações de calculos para o arima usar.
        # (Similar ao 'grid search' de machine learning)
        seasonal_pdq = [(x[0], x[1], x[2], ARIMA_SASONALIDADE) for x in list(itertools.product(p, d, q))];

        if verbose == True:
            print(seasonal_pdq);

        if verbose:
            print('# ==============================================================================================================');
            print('# Escolhendo a melhor combinação de parametros arima.');
            print('# ==============================================================================================================\n\n');

        import warnings;
        warnings.filterwarnings("ignore")  # Negócio chato pacas...

        menorCombinacao = None;
        menorCombinacaoValor = 99999999999999999;  # Mesma coisa que Integer.MAX_VALUE

        import statsmodels.api as sm;

        for parametro in pdq:
            for parametro_sasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.df,
                                                    order=parametro,
                                                    seasonal_order=parametro_sasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    resultado = mod.fit(disp=0)#disp == 0 Oculta log indesejado que trava o programa....

                    if resultado.aic < menorCombinacaoValor:
                        menorCombinacao = [parametro, parametro_sasonal, ARIMA_SASONALIDADE];
                        menorCombinacaoValor = resultado.aic;

                    if verbose == True:
                        print('ARIMA{}x{}x{} - AIC:{}'.format(parametro, parametro_sasonal, ARIMA_SASONALIDADE, resultado.aic))

                except:
                    # Algumas combinações são NaN (Não são possíveis! por isso tem esse TryCath)
                    continue

        if verbose == True:
            print('\n\n')
            print('O menor valor encontrado par ao AIC é: {}'.format(menorCombinacaoValor))
            print('Utilizando a combinação: ARIMA{}x{}x{}'.format(menorCombinacao[0], menorCombinacao[1], menorCombinacao[2]))

        theOrder = menorCombinacao[0];
        theSeasonal_order = menorCombinacao[1];

        if verbose:
            print('# ==============================================================================================================');
            print('# Ajustando Modelo.');
            print('# ==============================================================================================================\n\n');

        mod = sm.tsa.statespace.SARIMAX(self.df,
                                        order=theOrder,
                                        seasonal_order=theSeasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        self.modelFit = mod.fit(disp=0)#disp == 0 Oculta log indesejado que trava o programa....
        print('\n\n');
        print(self.modelFit.summary().tables[1])

    def diagnostico(self):
        self.modelFit.plot_diagnostics(figsize=(15, 12))
        plt.show()

    def ARIMAPrediction(self, forecastStartingDate = None ,datasetStartDate = None, theFigsize = (14, 7)):

        if datasetStartDate == None:
            datasetStartDate = self.df.index[0];

        if forecastStartingDate == None:
            forecastStartingDate = self.df.index[0];

        # Predição propriamente dita
        pred = self.modelFit.get_prediction(start=pd.to_datetime(forecastStartingDate), dynamic=False)
        pred_ci = pred.conf_int()

        ax = self.df[datasetStartDate:].plot(label='Observado')
        pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7, figsize=theFigsize)


        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)

        ax.set_xlabel(self.nomeDaColunaDeDatas)
        ax.set_ylabel(self.nomeDaColunaObjetivo)
        plt.legend()
        plt.show()

    def ARIMAForecast(self, steps, datasetStartDate = None, theFigsize = (14, 7), verbose = False):

        if datasetStartDate == None:
            datasetStartDate = self.df.index[0];

        pred_uc = self.modelFit.get_forecast(steps=steps)

        if verbose is True:
            print(pred_uc.predicted_mean)

        pred_ci = pred_uc.conf_int()

        ax = self.df[datasetStartDate:].plot(label='Observado')
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=theFigsize)

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)

        ax.set_xlabel(self.nomeDaColunaDeDatas)
        ax.set_ylabel(self.nomeDaColunaObjetivo)
        plt.legend()
        plt.show()

    def ARIMAPredictionToPred(self, forecastStartingDate = None ,datasetStartDate = None, theFigsize = (14, 7)):

        if datasetStartDate == None:
            datasetStartDate = self.df.index[0];

        if forecastStartingDate == None:
            forecastStartingDate = self.df.index[0];

        # Predição propriamente dita
        pred = self.modelFit.get_prediction(start=pd.to_datetime(forecastStartingDate), dynamic=False)
        return pred;

    def ARIMAForecastToPred(self, steps, datasetStartDate = None, theFigsize = (14, 7), verbose = False):

        if datasetStartDate == None:
            datasetStartDate = self.df.index[0];

        pred:PredictionResultsWrapper = self.modelFit.get_forecast(steps=steps)

        if verbose is True:
            print(pred.predicted_mean)

        return pred;

    def ARIMAForecastToJson(self, steps, datasetStartDate = None, theFigsize = (14, 7), verbose = False):

        if datasetStartDate == None:
            datasetStartDate = self.df.index[0];

        pred:PredictionResultsWrapper = self.modelFit.get_forecast(steps=steps)

        if verbose is True:
            print(pred.predicted_mean)

        return pred.predicted_mean.to_json();



    # ---------------------------------------------------------------------------
    #   ARIMA Tratamentos - Tecnicas de Regularização
    # ---------------------------------------------------------------------------

    def pegarDataframeAgrupadoPor(self, nomeDoAtributoDeTempo, nomeDoAtributoValorado, replaceInside = False):
        groupedDataframe = self.df.groupby(nomeDoAtributoDeTempo)[nomeDoAtributoValorado].sum().reset_index()
        groupedDataframe = groupedDataframe.set_index(nomeDoAtributoDeTempo)
        if (replaceInside == True):
            self.df = groupedDataframe;
        return groupedDataframe

    def ordenarDataframePor(self, nomeDoAtributoValorado, unidadeDeTempo = 'MS', replaceInside = False): # ‘M’ Mensal # 'S' Inicio
        orderedDataframe = self.df[nomeDoAtributoValorado].resample(unidadeDeTempo).mean();
        if (replaceInside == True):
            self.df = orderedDataframe;
        return orderedDataframe;