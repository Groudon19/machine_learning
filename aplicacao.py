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


tf.random.set_seed(42)
np.random.seed(42)

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
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    
    layers.Dense(15, activation='softmax') # Camada de saída com 15 classes

])

modelo.summary()


print("====Etapa 7: Treinamento do modelo====\n")

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
    epochs=70, #ajuste
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

print("Treinamento concluído\n")

print("====Etapa 8: Gerando Curvas de Aprendizado====\n")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
dados_historico = historico.history


# Gráfico da perda
ax[1].plot(dados_historico['loss'], label='Loss - Treino', color='blue', linewidth=2)
ax[1].plot(dados_historico['val_loss'], label='Loss - Validação', color='orange', linestyle='--', linewidth=2)
ax[1].set_title('Evolução da Perda (Loss)')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Perda')
ax[1].legend(loc='upper right')
ax[1].grid(True, linestyle='--', alpha=0.7)

# Gráfico da acuracia (Taxa de Acerto)
ax[0].plot(dados_historico['accuracy'], label='Acurácia - Treino', color='blue', linewidth=2)
ax[0].plot(dados_historico['val_accuracy'], label='Acurácia - Validação', color='orange', linestyle='--', linewidth=2)
ax[0].set_title('Evolução da Acurácia (Convergência)')
ax[0].set_xlabel('Épocas')
ax[0].set_ylabel('Acurácia')
ax[0].legend(loc='lower right')
ax[0].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("imagens/curvas_aprendizado.png", dpi=300)
plt.show()


print("\n==== Etapa 9: Avaliação Final no Conjunto de Teste ====\n")

# 1. Previsões no conjunto de teste (nunca visto durante o treino)
probabilidades = modelo.predict(x_teste_s)
y_pred = np.argmax(probabilidades, axis=1)

# 2. Relatório completo (Precisão, Recall/Sensibilidade, F1-Score e Acurácia)[cite: 6]
nomes_classes = [f"Classe {i+1}" for i in range(15)]
print(classification_report(y_teste, y_pred, target_names=nomes_classes))

# 3. Matriz de Confusão 15x15[cite: 6]
cm = confusion_matrix(y_teste, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1, 16))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Matriz de Confusão - Conjunto de Teste (R15)', fontsize=13, fontweight='bold')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.savefig("imagens/matriz_confusao.png", dpi=300)
plt.show()
