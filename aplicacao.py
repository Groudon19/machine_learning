import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.optimizers import Adam




dados = pd.read_csv(
    "R15.txt", 
    sep="\t", #separa colunas
    comment='#',
    header=None,
    names=["X1", "X2", "label"]
)


print("Etapa 3: Tratamento de dados nulos\n")
print("Verificando dados nulos antes do tratamento:\n")
nulos = dados.isnull().sum()
print(nulos)

# Opção 1: Remover linhas com valores nulos
if nulos.any():
    print("Removendo linhas com valores nulos:\n")
    dados = dados.dropna()
    
x = dados[["X1", "X2"]].values
y = dados["label"].values

y = y - 1 #ajuste necessária para o tensor flow: [1..15] -> [0..14]

print("Etapa 5: Separação dos dados em treino, validação e teste:\n")
# Separando conforme proporção: 70%, 30% 

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, 
    test_size=0.3,  #30% para teste, 70% pra treino
    random_state=42,
    stratify=y
)

# 30%: 15% validação, 15% teste
x_val, x_teste, y_val, y_teste = train_test_split(
    x_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"Tamanho do Treino: {len(x_treino)} amostras\n")
print(f"Tamanho da Validação: {len(x_val)} amostras\n")
print(f"Tamanho do Teste: {len(x_teste)} amostras\n")

