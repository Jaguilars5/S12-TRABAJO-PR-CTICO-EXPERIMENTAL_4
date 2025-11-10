from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Ejecutando Clustering No Supervisado (KMeans)...")

# 1. Crear datos artificiales
X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

# 2. Entrenar modelo no supervisado
# Se agrega n_init=10 para evitar advertencias futuras
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# 3. Visualizar resultados
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=30)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroides",
)
plt.title("Clustering con KMeans")
plt.legend()
plt.show()
