import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
    
X = dados[["X1", "X2"]].values
Y = dados["label"].values


