import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping




dados = pd.read_csv(
    "R15.txt", 
    sep="\t", #separa colunas
    comment='#',
    header=None,
    names=["X1", "X2", "label"]
)


print("====Etapa 3: Tratamento de dados nulos====\n")
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

print("====Etapa 5: Separação dos dados em treino, validação e teste:====\n")
# Separando conforme proporção: 70%, 30% 

x_treino, x_temp, y_treino, y_temp = train_test_split(
    x, y, 
    test_size=0.3,  #30% para teste, 70% pra treino
    random_state=42,
    stratify=y
)

# 30%: 15% validação, 15% teste
x_val, x_teste, y_val, y_teste = train_test_split(
    x_temp, y_temp,
    test_size=0.5, # 30*0.5 = 15%
    random_state=42,
    stratify=y_temp
)

print(f"Tamanho do Treino: {len(x_treino)} amostras\n")
print(f"Tamanho da Validação: {len(x_val)} amostras\n")
print(f"Tamanho do Teste: {len(x_teste)} amostras\n")


print("====Etapa 4: Escalonamento dos dados:====\n")
print("Antes de escalonar:", x_treino[0])

scaler = MinMaxScaler()
x_treino_s = scaler.fit_transform(x_treino) # Escala entre 0 e 1
x_val_s = scaler.transform(x_val)
x_teste_s = scaler.transform(x_teste)


print("Depois de escalonar:", x_treino_s[0])


print("====Etapa 6: Construção da MLP:====\n")

modelo = keras.Sequential([
    layers.Input(shape=(2,)), #Camada de entrada com 2 neuronios. x1 e x2
    
    # Camada oculta com ReLU e regularização L2 
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    
    # Camada oculta com ReLU e regularização L2 
    #layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    
    layers.Dense(15, activation='softmax') # Camada de saída com 15 classes

])

modelo.summary()


print("====Etapa 7: Treinamento do modelo====\n")

TAXA_DE_APRENDIZADO = 0.002
otimizador_adam= Adam(learning_rate=TAXA_DE_APRENDIZADO)

# Compilação do modelo
modelo.compile(
    optimizer=Adam(learning_rate=0.005),
    loss='sparse_categorical_crossentropy', # Ideal para rótulos inteiros (0 a 14)
    metrics=['accuracy']
)

# 2. Configuração do Early Stopping (Etapa 8 - Regularização)
# Para o treino se a val_loss não melhorar por 20 épocas seguidas e restaura os melhores pesos
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# 3. Treinamento do modelo
historico = modelo.fit(
    x_treino_s, y_treino,
    validation_data=(x_val_s, y_val),
    epochs=250,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)
