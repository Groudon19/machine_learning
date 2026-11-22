import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("imagens_csv", exist_ok=True)

dados = pd.read_csv("base_classificacao_binaria.csv")

# 1. Distribuição das Classes (Target) e Ambiente
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
contagem_target = dados["target"].value_counts().sort_index()
plt.bar(["Classe 0", "Classe 1"], contagem_target.values, color=["steelblue", "darkorange"], edgecolor="black")
plt.title("Distribuição do Target")
plt.ylabel("Quantidade de Amostras")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
contagem_env = dados["environment"].fillna("Nulo").value_counts()
plt.bar(contagem_env.index.astype(str), contagem_env.values, color="mediumseagreen", edgecolor="black")
plt.title("Distribuição por Ambiente (Environment)")
plt.ylabel("Quantidade de Amostras")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("imagens_csv/distribuicao_classes.png", dpi=300)
plt.show()

# 2. Histogramas das Features Numéricas
colunas_num = dados.select_dtypes(include=[np.number]).columns.drop("target", errors="ignore")

dados[colunas_num].hist(bins=25, figsize=(16, 10), color="steelblue", edgecolor="black")
plt.suptitle("Histogramas das Features Numéricas", fontsize=14)
plt.tight_layout()
plt.savefig("imagens_csv/histogramas_dados.png", dpi=300)
plt.show()

# 3. Boxplots por Feature (Detecção de Outliers)
fig, axs = plt.subplots(3, 4, figsize=(16, 10))
axs = axs.flatten()

for i, col in enumerate(colunas_num):
    dados.boxplot(column=col, ax=axs[i])
    axs[i].set_title(col)

for j in range(len(colunas_num), len(axs)):
    axs[j].axis("off")

plt.suptitle("Boxplot por Feature", fontsize=14)
plt.tight_layout()
plt.savefig("imagens_csv/boxplots_dados.png", dpi=300)
plt.show()

# 4. Validação de dados duplicados, nulos e resumo
dupl = dados.duplicated()
print("Duplicados:", dupl.sum())
print("\nValores nulos por coluna:\n", dados.isnull().sum())
print("\nProporção das classes (%):\n", (dados["target"].value_counts(normalize=True) * 100).round(2))