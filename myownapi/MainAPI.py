import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from django.core.validators import URLValidator
from sklearn import preprocessing


class MainAPI:

    # ---------------------------------------------------------------------------
    #   Basic Functions
    # ---------------------------------------------------------------------------
    apiVersion = "2.0.1z"
    debug = False;  # Define se a aplicação irá relatar os debugs!
    info = True;  # Define se a aplicação irá relatar os passos!

    @staticmethod
    def getVersion():
        return MainAPI.apiVersion;

    # Dorme por X Segundos.
    def sleepFor(self, seconds):
        if (self.info):
            time.sleep(seconds)

    def getDataframe(self):
        return self.df;

    def debug(self, text):
        if (self.debug == True):
            print(text);

    def info(self, text):
        if (self.info):
            print(text);

    def is_url(self, url_name):
        try:
            URLValidator()(url_name)
            return True;
        except:
            return False;

    def clone(self):
        otherSelf = self.__class__(); #Cria uma nova instancia da classe self, no caso, os sub_módulos!
        otherSelf.setDataframe(self.df.copy(deep=True));
        return otherSelf;

    def copy(self):
        return self.clone();

    def printIMGFromURL(self, url):
        from PIL import Image
        import requests
        from io import BytesIO
        from IPython.display import display
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        display(img)

    # ---------------------------------------------------------------------------
    #   Panda
    # ---------------------------------------------------------------------------
    df = None;  # DataFrame

    def setDataframe(self, newDataFrame):
        self.df = newDataFrame;

    def read_csv(self, filePath):
        self.info("Iniciando leitura do arquivo:\n --> " + filePath);
        self.df = pd.read_csv(filePath);
        self.info("Arquivo lido com sucesso!");
        return self.df;

    def read_xls(self, filePath):
        self.info("Iniciando leitura do arquivo:\n --> " + filePath);
        self.df = pd.read_excel(filePath);
        self.info("Arquivo lido com sucesso!");
        return self.df;

    #   Verifica se o DataFrame possui valores nulos, se tiver,
    #     cria uma cópia do dataset, dropa as variáveis nulas
    #     e ai sim descreve ele!
    def descreverDataFrame(self, safeMode=None):

        if (safeMode == None):
            foundNull = self.df.isnull().values.any();
            if foundNull == True:
                self.info("Variáveis nulas foram encontradas!!");
                self.info("[Criando uma cópia do DataFrame sem as variáveis nulas]");
            return self.descreverDataFrame(safeMode=foundNull);

        if (safeMode == True):
            dataframeASerDescrito = self.df.copy(deep=True);
        else:
            dataframeASerDescrito = self.df;

        thePercentiles = [.20, .40, .60, .80];
        includedTypes = ['object', 'float', 'int'];
        desc = dataframeASerDescrito.describe(percentiles=thePercentiles, include=includedTypes);
        return desc;

    def descreverAtributo(self, nomeDoAtributo):
        thePercentiles = [.20, .40, .60, .80];
        includedTypes = ['object', 'float', 'int'];
        colunaSelecionada = self.df[nomeDoAtributo];
        desc = colunaSelecionada.describe(percentiles=thePercentiles, include=includedTypes);
        valoresNulos = colunaSelecionada.isnull().sum();
        if valoresNulos != 0:
            print("Variáveis nulas: " + str(valoresNulos));
        valoresNaN = colunaSelecionada.isna().sum();
        if valoresNaN != 0:
            print("Variáveis NaN: " + str(valoresNaN));
        return desc;

    def plotarSelf(self, theFigsize = None):
        if theFigsize != None:
            plt.figure(figsize=theFigsize)
        return sns.distplot(self.df);

    def plotarSimpleSelf(self, theFigsize = None):
        if theFigsize != None:
            return self.df.plot(figsize=theFigsize)
        else:
            return self.df.plot()

    def plotarSimple(self, nomeDoAtributo, safeMode = True, theFigsize = None):  # Plotar Gráfico para dados numéricos
        foundNull = self.df[nomeDoAtributo].isnull().values.any();
        if (safeMode == True and foundNull == True):
            self.info("Variáveis nulas foram encontradas!!");
            self.info("[Criando uma cópia do DataFrame sem as variáveis nulas]");
            dfCopy = self.df[nomeDoAtributo].copy(deep=True);
            dfCopy = dfCopy.apply(pd.to_numeric,errors='coerce')  # Transforma todos os dados não numéricos para NaN
            columASerPlotada = dfCopy.dropna();                   # Remove linhas com variáveis NaN
        else:
            columASerPlotada = self.df[nomeDoAtributo];

        if theFigsize != None:
            return columASerPlotada.plot(figsize=theFigsize);
        else:
            return columASerPlotada.plot();

    def plotar(self, nomeDoAtributo, safeMode = True, theFigsize = None):  # Plotar Gráfico para dados numéricos

        foundNull = self.df[nomeDoAtributo].isnull().values.any();
        if (safeMode == True and foundNull == True):
            self.info("Variáveis nulas foram encontradas!!");
            self.info("[Criando uma cópia do DataFrame sem as variáveis nulas]");
            dfCopy = self.df[nomeDoAtributo].copy(deep=True);
            dfCopy = dfCopy.apply(pd.to_numeric,errors='coerce')  # Transforma todos os dados não numéricos para NaN
            columASerPlotada = dfCopy.dropna();                   # Remove linhas com variáveis NaN
        else:
            columASerPlotada = self.df[nomeDoAtributo];

        if theFigsize != None:
            plt.figure(figsize=theFigsize)
        return sns.distplot(columASerPlotada);

    def autoCorrelacao(self, nomeDoAtributo=None):  # Plotar Gráfico de AutoCorrelação
        from pandas.plotting import autocorrelation_plot;
        if nomeDoAtributo != None:
            return autocorrelation_plot(self.df[nomeDoAtributo]);
        else:
            return autocorrelation_plot(self.df);

    def pegarDataframeCondicionalmente(self, nomeDoAtributo, conteudoDoAtributo):  # Plotar Gráfico de AutoCorrelação
        return self.df.loc[self.df[nomeDoAtributo] == conteudoDoAtributo];

    def columns(self):
        return self.df.columns;

    def head(self):
        return self.df.head()

    def head(self,amount):
        return self.df.head(amount)


    # ---------------------------------------------------------------------------
    #   DataFrame Tratamentos - Tecnicas de Regularização
    # ---------------------------------------------------------------------------

    def atributosComVariaveisNulas(self):
        return self.df.isnull().any();

    def removerLinhasComVariaveisNaN(self, nomeDoAtributo):
        self.df[nomeDoAtributo] = self.df[nomeDoAtributo].apply(pd.to_numeric, errors='coerce') # Transforma todos os dados não numéricos para NaN
        self.df[nomeDoAtributo].dropna();                                                                       #remove todos os NaN e nulls

    def tratarVariaveisNulasComMediaDasOutras(self, nomeDoAtributo):
        colunaSelecionada = self.df[nomeDoAtributo];
        valoresNulos = colunaSelecionada.isnull().sum();
        if (valoresNulos == 0):
            self.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDoAtributo)
            return None;

        mediaDosDemais = colunaSelecionada.mean();
        colunaSelecionada.fillna(mediaDosDemais,
                                 inplace=True)  # Subistitui nas posições aonde existe um Null, o valor da média dos demais
        self.info("Um total de " + str(valoresNulos) + " valores nulos foram encontrados.")
        self.info(
            "O valor desses atributos nulos foram definidos para a média dos demais, no caso: " + str(mediaDosDemais));

    def tratarVariaveisNulasComValorEspecifico(self, nomeDoAtributo, valorEspecifico):
        colunaSelecionada = self.df[nomeDoAtributo];
        valoresNulos = colunaSelecionada.isnull().sum();
        if (valoresNulos == 0):
            self.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDoAtributo)
            return None;

        colunaSelecionada.fillna(valorEspecifico,
                                 inplace=True)  # Subistitui nas posições aonde existe um Null, o valorEspecifico
        self.info("Um total de " + str(valoresNulos) + " valores nulos foram encontrados.")
        self.info("O valor desses atributos nulos foram definidos para : " + str(valorEspecifico));

    def tratarVariaveisZeradasComMediaDasOutras(self, nomeDaColuna):
        colunaSelecionada = self.df[nomeDaColuna];
        valoresZerados = colunaSelecionada.value_counts().get(0, 0);
        if (valoresZerados == 0):
            self.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDaColuna)
            return None;

        mediaDosDemais = np.mean(colunaSelecionada);
        colunaSelecionada.replace(0,
                                  mediaDosDemais)  # Subistitui nas posições aonde existe um valor ZERO, o valor da média dos demais
        self.info("Um total de " + str(valoresZerados) + " foram encontrados.")
        self.info(
            "O valor desses atributos nulos foi definido para a média dos demais, no caso: " + str(mediaDosDemais));

    def tratarVariaveisCategoricasParaNumericas(self, nomeDaColuna):
        le = preprocessing.LabelEncoder();
        self.df[nomeDaColuna] = le.fit_transform(self.df[nomeDaColuna]);
        return self.df[nomeDaColuna];


