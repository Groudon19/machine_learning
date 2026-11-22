import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("imagens", exist_ok=True)


dados = pd.read_csv(
    "R15.txt", 
    sep="\t", #separa colunas
    comment='#',
    header=None,
    names=["X1", "X2", "label"]
) 

#grafico de dispersão
plt.figure(figsize=(10, 6))

classes_unicas = np.unique(dados["label"])

cmap = plt.get_cmap('tab20')
cores = [cmap(i) for i in range(len(classes_unicas))]

for i, classe in enumerate(classes_unicas):
    info = dados[dados["label"] == classe]
    
    plt.scatter(info["X1"], info["X2"], label=f"Classe {classe}", color=cores[i], alpha=0.7)
    

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Distribuição dos dados")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig("imagens/distribuicao_dados.png", dpi=300)
plt.show()

#Histograma
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

for classe in classes_unicas:
    info = dados[dados["label"]== classe]
    
    axs[0].hist(info["X1"], bins=15, alpha=0.7, label=f"Classe {classe}")
    
    axs[1].hist(info["X2"], bins=15, alpha=0.7, label=f"Classe {classe}")
    
    
axs[0].set_title("X1 por classe")
axs[1].set_title("X2 por classe")
axs[0].legend()
axs[1].legend()

plt.tight_layout() 
plt.savefig("imagens/histogramas_dados.png", dpi=300)
plt.show()

#boxplot
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

dados.boxplot(column="X1", by="label", ax=axs[0])
dados.boxplot(column="X2", by="label", ax=axs[1])

axs[0].set_title("X1")
axs[1].set_title("X2")

plt.suptitle("Boxplot por classe")
plt.tight_layout()
plt.savefig("imagens/boxplots_dados.png", dpi=300)
plt.show()



# Validação da quantidade d dados duplicados
dupl = dados.duplicated()
print("Duplicados", dupl.sum())
