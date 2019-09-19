import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MainAPI:

    # ---------------------------------------------------------------------------
    #   Basic Functions
    # ---------------------------------------------------------------------------
    apiVersion = "0.2.1d"
    debug = False;          #Define se
    info = True;            #Define se a aplicação irá relatar os passos!

    @staticmethod
    def getVersion():
        return MainAPI.apiVersion;

    @staticmethod   #Dorme por X Segundos.
    def sleepFor(seconds):
        if (MainAPI.info):
            time.sleep(seconds)

    @staticmethod
    def getDataframe():
        return MainAPI.df;

    @staticmethod
    def debug(text):
        if (MainAPI.debug == True):
            print(text);

    @staticmethod
    def info(text):
        if (MainAPI.info):
            print(text);

    #---------------------------------------------------------------------------
    #   Panda
    # ---------------------------------------------------------------------------
    df = None;          #DataFrame

    @staticmethod
    def read_csv(filePath):
        MainAPI.info("Iniciando leitura do arquivo:\n --> " + filePath);
        MainAPI.df = pd.read_csv(filePath);

        MainAPI.info("Arquivo lido com sucesso!");
        return MainAPI.df;


    #   Verifica se o DataFrame possui valores nulos, se tiver,
    #     cria uma cópia do dataset, dropa as variáveis nulas
    #     e ai sim descreve ele!
    @staticmethod
    def descreverDataFrame(safeMode=None):

        if (safeMode==None):
            foundNull = MainAPI.df.isnull().values.any();
            if foundNull == True:
                MainAPI.info("Variáveis nulas foram encontradas!!");
                MainAPI.info("[Criando uma cópia do DataFrame sem as variáveis nulas]");
            return MainAPI.descreverDataFrame(safeMode=foundNull);

        dataframeASerDescrito = MainAPI.df;
        if (safeMode == True):
            dataframeASerDescrito = MainAPI.df.copy(deep=True);
        thePercentiles = [.20, .40, .60, .80];
        includedTypes = ['object', 'float', 'int'];
        desc = dataframeASerDescrito.describe(percentiles=thePercentiles, include=includedTypes);
        return desc;


    @staticmethod
    def descreverAtributo(nomeDoAtributo):
        thePercentiles = [.20, .40, .60, .80];
        includedTypes = ['object', 'float', 'int'];
        desc = MainAPI.df[nomeDoAtributo].describe(percentiles=thePercentiles, include=includedTypes);
        return desc;


    @staticmethod
    def plotar(nomeDoAtributo):                             #Plotar Gráfico para dados numéricos
        return sns.distplot(MainAPI.df[nomeDoAtributo]);

    @staticmethod
    def plotarDadosCategoricos(nomeDoAtributo):             #Plotar Gráfico para dados categóricos
        plt.figure(figsize=(10, 10))
        plt.subplot(311)

        return sns.distplot(MainAPI.df[nomeDoAtributo]);

    # ---------------------------------------------------------------------------
    #   Linear Regression
    # ---------------------------------------------------------------------------

    dfDeAtributosPrecisao = None;   #DataFrame de atributos de Precisão (Atributos que irão balisar a regressão Linear)
    dfDeAtributoDesejado = None;    #DataFrame de atributos de Precisão (Atributos que irão balisar a regressão Linear)
    porcentagem_teste = 0.25;       #Porcentagem dos dados que será usada para treinar a regressão linear
    erro_quadratico = 0;            #Qual o erro quadrático resultante da regressão linear (quanto menor melhor)
    r_square = 0;                   #Qual o RSquare (Coeficiente de Determinação) da regressão Linear, varia entre 0 e 1, quanto maior melhor (% de explicação dos dados de saida com base nos dados de entrada)


    @staticmethod
    def definirAtributosDePrecisao(arrayDeAtributos):
        MainAPI.info("Atributos de precisão definidos para:")
        for atributo in arrayDeAtributos:
            MainAPI.info("-->  " + atributo);
        MainAPI.info("Total de {0} atributos.".format(len(arrayDeAtributos)));
        MainAPI.dfDeAtributosPrecisao = MainAPI.df[arrayDeAtributos]; #Criando um dataFrame Auxiliar com as linhas e colunas selecionadas pelo Array


    @staticmethod
    def definirAtributoDesejado(atributoDesejado):
        import warnings
        warnings.filterwarnings("ignore", message="From version 0.21, test_size will always complement train_size unless both are specified.")
        MainAPI.info("Definindo atributo desejado para: " + atributoDesejado)
        MainAPI.dfDeAtributoDesejado = MainAPI.df[atributoDesejado]; #Criando um dataFrame Auxiliar com a linha de coluna do atributo desejado


    @staticmethod
    def lregDefinirPorcentagemDeTeste(novaPorcentagem):
        MainAPI.porcentagem_teste = novaPorcentagem;
        porcentagem_de_teste = 100 - novaPorcentagem;
        MainAPI.info("Porcentagem de Treino definido para: " + str(MainAPI.porcentagem_teste) + "%")
        MainAPI.info("Porcentagem de Teste definido para: " + str(porcentagem_de_teste)  + "%" )


    lreg    = None;             #Instancia do LinearRegression()
    x_train = None;             #Variáveis de treino (Xcoord)
    y_train = None;             #Variáveis de treino (Ycoord)
    x_test  = None;             #Variáveis de teste (Xcoord)
    y_test  = None;             #Variáveis de teste (Ycoord)

    dfDePesos = None;    #DataFrame com todos os coeficientes (pesos de cada um dos atributos)

    @staticmethod
    def lregAplicarRegressaoLinear():
        MainAPI.x_train, MainAPI.x_test, MainAPI.y_train, MainAPI.y_test = train_test_split(MainAPI.dfDeAtributosPrecisao, MainAPI.dfDeAtributoDesejado, test_size=(MainAPI.porcentagem_teste / 100));

        MainAPI.info("Iniciando regressão Linear!")
        MainAPI.lreg = LinearRegression()                       #Instanciando um novo objeto [sklearn.linear_model]
        MainAPI.lreg.fit(MainAPI.x_train, MainAPI.y_train);     #Treinando os X e Y
        MainAPI.info("Regressão Linear aplicada com sucesso!")

    @staticmethod
    def calcularErroMedioQuadratico():
        predicao = MainAPI.lreg.predict(MainAPI.x_test);                        #Realiza a predição de acordo com o X de teste
        MainAPI.erro_quadratico = np.mean((predicao - MainAPI.y_test) ** 2)     #Calculando o erro Quadrático
        return MainAPI.erro_quadratico;


    @staticmethod
    def calcularCoeficientesDePesos():
        MainAPI.dfDePesos = DataFrame(MainAPI.x_train.columns)           #Criando um dataFrame com os pesos dos coeficientes
        MainAPI.dfDePesos['Pesos'] = Series(MainAPI.lreg.coef_)          #O LinearRegression() Salva esses coeficientes na variavel "coef_"
        return MainAPI.dfDePesos;


    @staticmethod
    def calcularRSQuare():
        MainAPI.r_square = MainAPI.lreg.score(MainAPI.x_test,MainAPI.y_test)
        return MainAPI.r_square;

    # ---------------------------------------------------------------------------
    #   DataFrame Tratamentos
    # ---------------------------------------------------------------------------

    @staticmethod
    def tratarVariaveisNulasComMediaDasOutras(nomeDoAtributo):
        colunaSelecionada = MainAPI.df[nomeDoAtributo];
        valoresNulos = colunaSelecionada.isnull().sum();
        if (valoresNulos == 0):
            MainAPI.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDoAtributo)
            return None;

        mediaDosDemais = colunaSelecionada.mean();
        colunaSelecionada.fillna(mediaDosDemais, inplace=True)  #Subistitui nas posições aonde existe um Null, o valor da média dos demais
        MainAPI.info("Um total de " + str(valoresNulos) + " valores nulos foram encontrados.")
        MainAPI.info("O valor desses atributos nulos foram definidos para a média dos demais, no caso: " + str(mediaDosDemais));


    @staticmethod
    def tratarVariaveisNulasComValorEspecifico(nomeDoAtributo, valorEspecifico):
        colunaSelecionada = MainAPI.df[nomeDoAtributo];
        valoresNulos = colunaSelecionada.isnull().sum();
        if (valoresNulos == 0):
            MainAPI.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDoAtributo)
            return None;

        colunaSelecionada.fillna(valorEspecifico, inplace=True)  #Subistitui nas posições aonde existe um Null, o valorEspecifico
        MainAPI.info("Um total de " + str(valoresNulos) + " valores nulos foram encontrados.")
        MainAPI.info("O valor desses atributos nulos foram definidos para : " + str(valorEspecifico));


    @staticmethod
    def tratarVariaveisZeradasComMediaDasOutras(nomeDaColuna):
        colunaSelecionada = MainAPI.df[nomeDaColuna];
        valoresZerados = colunaSelecionada.value_counts().get(0,0);
        if (valoresZerados == 0):
            MainAPI.info("Nenhum valor nulo foi encontrado na coluna: " + nomeDaColuna)
            return None;

        mediaDosDemais = np.mean(colunaSelecionada);
        colunaSelecionada.replace(0, mediaDosDemais)            #Subistitui nas posições aonde existe um valor ZERO, o valor da média dos demais
        MainAPI.info("Um total de " + str(valoresZerados) + " foram encontrados.")
        MainAPI.info("O valor desses atributos nulos foi definido para a média dos demais, no caso: " + str(mediaDosDemais));


    # ---------------------------------------------------------------------------
    #   Tecnicas de Regularização
    # ---------------------------------------------------------------------------


    # Ta no progresso ainda, to tentando entender como funciona o LabelEncoder :/