import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from django.core.validators import URLValidator
from sklearn import metrics
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import preprocessing
from xgboost import XGBRegressor

from myownapi.MainAPI import MainAPI

class AnalyticsXGBoost(MainAPI):

    # ---------------------------------------------------------------------------
    #   Linear Regression
    # ---------------------------------------------------------------------------

    dfDeAtributosPrecisao = None;  # DataFrame de atributos de Precisão (Atributos que irão balisar a regressão Linear)
    dfDeAtributoDesejado = None;  # DataFrame de atributos de Precisão (Atributos que irão balisar a regressão Linear)
    porcentagem_teste = 25;  # Porcentagem dos dados que será usada para treinar a regressão linear
    erro_quadratico = 0;  # Qual o erro quadrático resultante da regressão linear (quanto menor melhor)
    r_square = 0;  # Qual o RSquare (Coeficiente de Determinação) da regressão Linear, varia entre 0 e 1, quanto maior melhor (% de explicação dos dados de saida com base nos dados de entrada)

    def definirAtributosDePrecisao(self, arrayDeAtributos):
        self.info("Atributos de precisão definidos para:")
        for atributo in arrayDeAtributos:
            self.info("-->  " + atributo);
        self.info("Total de {0} atributos.".format(len(arrayDeAtributos)));
        self.dfDeAtributosPrecisao = self.df[arrayDeAtributos];  # Criando um dataFrame Auxiliar com as linhas e colunas selecionadas pelo Array

    def definirAtributoDesejado(self, atributoDesejado):
        import warnings
        warnings.filterwarnings("ignore",
                                message="From version 0.21, test_size will always complement train_size unless both are specified.")
        self.info("Definindo atributo desejado para: " + atributoDesejado)
        self.dfDeAtributoDesejado = DataFrame(self.df[atributoDesejado]);  # Criando um dataFrame Auxiliar com a linha de coluna do atributo desejado

    def definirPorcentagemDeTeste(self, novaPorcentagem):
        self.porcentagem_teste = novaPorcentagem;
        porcentagem_de_treino = 100 - novaPorcentagem;
        self.info("Porcentagem de Teste definido para: " + str(self.porcentagem_teste) + "%")
        self.info("Porcentagem de Treino definido para: " + str(porcentagem_de_treino) + "%")


    xGBoostRegressor = None;  # Instancia do LinearRegression()
    x_train = None;  # Variáveis de treino (Xcoord)
    y_train = None;  # Variáveis de treino (Ycoord)
    x_test = None;  # Variáveis de teste (Xcoord)
    y_test = None;  # Variáveis de teste (Ycoord)

    y_pred = None; #Variavel contendo a predição calculada!

    dfDePesos = None;  # DataFrame com todos os coeficientes (pesos de cada um dos atributos)

    def aplicarXGBoost(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfDeAtributosPrecisao,self.dfDeAtributoDesejado, test_size=(self.porcentagem_teste / 100),shuffle=False);
        self.info("Iniciando XGBoostRegressor")
        self.xGBoostRegressor = XGBRegressor(objective='reg:linear');  # Instanciando um novo objeto [sklearn.linear_model]
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.xGBoostRegressor.fit(self.x_train, self.y_train);  # Treinando os X e Y
        self.y_pred = self.xGBoostRegressor.predict(self.x_test);  # Realiza a predição de acordo com o X de teste
        self.info("XGBoost aplicado com sucesso!")

    def calcularCoeficientesDePesos(self):
        self.dfDePesos = DataFrame(self.x_train.columns)  # Criando um dataFrame com os pesos dos coeficientes
        self.dfDePesos['Coefficient'] = Series(self.xGBoostRegressor.coef_)  # O LinearRegression() Salva esses coeficientes na variavel "coef_"
        return self.dfDePesos;

    def compararPredicao(self):
        df = pd.DataFrame({'Atual': self.y_test, 'Predição': self.y_pred})
        return df;

    def compararPredicaoGraficamente(self, dataFrame = None, interval=(0,20)):
        # Pegar posição de 0 até a 20 e plotar ela!
        if dataFrame == None:
            dataFrame = self.compararPredicao();
        dataFrame.iloc[interval[0]:interval[1]].plot(kind='bar', figsize=(10, 8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()

    def calcularErroMedioQuadratico(self, verbose=False):
        result = metrics.mean_squared_error(self.y_test, self.y_pred);
        if verbose is True:
            print("Erro médio Quadrático: " + str(result))
        return result;

    def calcularErroMedioAbsoluto(self, verbose=False):
        result = metrics.mean_absolute_error(self.y_test, self.y_pred);
        if verbose is True:
            print("Erro médio Absoluto: " + str(result))
        return result;

    def calcularRaizQuadradadaDoErroMedioQuadratico(self, verbose=False):
        result = np.sqrt(self.calcularErroMedioQuadratico());
        if verbose is True:
            print("SQRT do Erro médio Quadrático: " + str(result))
        return result;