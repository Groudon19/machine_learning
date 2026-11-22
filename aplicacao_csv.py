import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Fixar semente para reprodutibilidade
tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("imagens_csv", exist_ok=True)

# Carregamento da base CSV
dados = pd.read_csv("base_classificacao_binaria.csv")

# ==========================================
# PREPARAÇÃO INICIAL DE FEATURES E TARGET
# ==========================================
# Remove a coluna identificadora inútil (sample_id)[cite: 13]
X_df = dados.drop(columns=["sample_id", "target"])
# O target já está no formato binário [0, 1], não precisa subtrair 1[cite: 13, 18]
y = dados["target"].astype(int).values

# ==========================================
# ETAPA 5: SEPARAÇÃO DOS DADOS (70% / 15% / 15%)
# ==========================================
print("==== Etapa 5: Separação dos dados em treino, validação e teste ====\n")

X_treino_df, X_temp_df, y_treino, y_temp = train_test_split(
    X_df, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val_df, X_teste_df, y_val, y_teste = train_test_split(
    X_temp_df, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Tamanho do Treino: {len(X_treino_df)} amostras")
print(f"Tamanho da Validação: {len(X_val_df)} amostras")
print(f"Tamanho do Teste: {len(X_teste_df)} amostras\n")

# ==========================================
# ETAPA 3: TRATAMENTO DE DADOS FALTANTES + ONE-HOT ENCODING
# ==========================================
print("==== Etapa 3: Imputação de dados faltantes e codificação categórica ====\n")

colunas_num = X_df.select_dtypes(include=[np.number]).columns.tolist()
colunas_cat = ["environment"]

# 3.1 Imputação Numérica (Mediana para resistir aos outliers observados na EDA)
imputer_num = SimpleImputer(strategy="median", add_indicator=True)
# Alternativa do roteiro (descomente se quiser testar KNN):
#imputer_num = KNNImputer(n_neighbors=5)

X_treino_num = imputer_num.fit_transform(X_treino_df[colunas_num])
X_val_num = imputer_num.transform(X_val_df[colunas_num])
X_teste_num = imputer_num.transform(X_teste_df[colunas_num])

# 3.2 Imputação Categórica (Moda) + One-Hot Encoding para 'environment'
imputer_cat = SimpleImputer(strategy="most_frequent")
X_treino_cat_imp = imputer_cat.fit_transform(X_treino_df[colunas_cat])
X_val_cat_imp = imputer_cat.transform(X_val_df[colunas_cat])
X_teste_cat_imp = imputer_cat.transform(X_teste_df[colunas_cat])

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_treino_cat = encoder.fit_transform(X_treino_cat_imp)
X_val_cat = encoder.transform(X_val_cat_imp)
X_teste_cat = encoder.transform(X_teste_cat_imp)

print("Nulos restantes após imputação (Treino):", np.isnan(X_treino_num).sum())
print("Categorias codificadas:", encoder.categories_[0], "\n")

# ==========================================
# ETAPA 4: ESCALONAMENTO DOS DADOS
# ==========================================
print("==== Etapa 4: Escalonamento dos dados (StandardScaler) ====\n")

scaler = StandardScaler()
# scaler = RobustScaler() 
X_treino_num_s = scaler.fit_transform(X_treino_num)
X_val_num_s = scaler.transform(X_val_num)
X_teste_num_s = scaler.transform(X_teste_num)

# Junta as features numéricas escalonadas com as categóricas codificadas (11 + 3 = 14 features)
x_treino_s = np.hstack([X_treino_num_s, X_treino_cat])
x_val_s = np.hstack([X_val_num_s, X_val_cat])
x_teste_s = np.hstack([X_teste_num_s, X_teste_cat])

print(f"Total de features finais de entrada na MLP: {x_treino_s.shape[1]}\n")

# ==========================================
# ETAPA 6: CONSTRUÇÃO DA MLP (CLASSIFICAÇÃO BINÁRIA)
# ==========================================
print("==== Etapa 6: Construção da MLP ====\n")

modelo = keras.Sequential([
    layers.Input(shape=(x_treino_s.shape[1],)),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Camada de saída binária: 1 neurônio com ativação Sigmoid
    layers.Dense(1, activation="sigmoid")
])

modelo.summary()

# ==========================================
# ETAPA 7: TREINAMENTO DO MODELO
# ==========================================
print("\n==== Etapa 7: Treinamento do modelo ====\n")

modelo.compile(
    optimizer=Adam(learning_rate=0.002),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

historico = modelo.fit(
    x_treino_s, y_treino,
    validation_data=(x_val_s, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# ETAPA 8: VALIDAÇÃO E CURVAS DE APRENDIZADO
# ==========================================
print("\n==== Etapa 8: Gerando Curvas de Aprendizado ====\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Curvas de Aprendizado - Classificação Binária (CSV)", fontsize=14, fontweight="bold")

axes[0].plot(historico.history["loss"], label="Loss de Treino", color="steelblue", linewidth=2)
axes[0].plot(historico.history["val_loss"], label="Loss de Validação", color="darkorange", linestyle="--", linewidth=2)
axes[0].set_title("Evolução da Função de Perda (Loss)")
axes[0].set_xlabel("Épocas")
axes[0].set_ylabel("Binary Cross-Entropy")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(historico.history["accuracy"], label="Acurácia de Treino", color="steelblue", linewidth=2)
axes[1].plot(historico.history["val_accuracy"], label="Acurácia de Validação", color="darkorange", linestyle="--", linewidth=2)
axes[1].set_title("Evolução da Acurácia")
axes[1].set_xlabel("Épocas")
axes[1].set_ylabel("Acurácia")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("imagens_csv/curvas_aprendizado.png", dpi=300)
plt.show()

# ==========================================
# ETAPA 9: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ==========================================
print("\n==== Etapa 9: Avaliação Final no Conjunto de Teste ====\n")

# Probabilidade > 0.5 -> Classe 1; caso contrário -> Classe 0
probabilidades = modelo.predict(x_teste_s)
y_pred = (probabilidades > 0.5).astype(int).flatten()

print(classification_report(y_teste, y_pred, target_names=["Classe 0", "Classe 1"]))

cm = confusion_matrix(y_teste, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Classe 0", "Classe 1"])
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Matriz de Confusão - Conjunto de Teste (CSV)", fontsize=13, fontweight="bold")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig("imagens_csv/matriz_confusao.png", dpi=300)
plt.show()