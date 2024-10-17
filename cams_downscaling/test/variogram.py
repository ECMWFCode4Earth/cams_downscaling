import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from pykrige.variogram_models import spherical_variogram_model
from sklearn.metrics.pairwise import haversine_distances

# Coordenades d'exemple (longitud i latitud de les estacions)
longituds = np.array([2.2045, 2.115, 2.0098, 1.9749, 2.0074, 1.9232, 2.2222, 2.089, 2.1014, 2.1538, 2.2095,
                      2.1534, 2.1254, 2.1874, 2.0425, 2.2121, 2.0138, 2.0821, 2.1151, 1.9996, 1.9905])  # Insereix aquí les longituds
latituds = np.array([41.4039, 41.3705, 41.3922, 41.4508, 41.5561, 41.4756, 41.4256, 41.4768, 41.5612,
                     41.3853, 41.4474, 41.3987, 41.5127, 41.3864, 41.4921, 41.5492, 41.3135, 41.3218, 41.3875, 41.4008, 41.4153])   # Insereix aquí les latituds
no2_vals = np.array([40.0, 37.0, 29.0, 32.0, 38.0, 29.0, 36.0, 18.0, 36.0, 
                     36.0, 37.0, 29.0, 37.0, 36.0, 43.0, 24.0, 31.0, 
                     37.0, 39.0, 33.0, 19.0])

# Càlcul de les distàncies entre estacions (en km, utilitzant la fórmula de Haversine)
coords = np.radians(np.c_[latituds, longituds])
distances = haversine_distances(coords) * 6371.0  # Radi de la Terra en km

# Càlcul del variograma experimental
n = len(no2_vals)
gamma = []
dist_bins = np.linspace(0, np.max(distances), num=15)

for d_min, d_max in zip(dist_bins[:-1], dist_bins[1:]):
    mask = (distances >= d_min) & (distances < d_max)
    if np.any(mask):
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    diffs.append((no2_vals[i] - no2_vals[j]) ** 2)
        gamma.append(np.mean(diffs) / 2)

# Filtrar els bins on es tenen dades de semi-variància
valid_bins = [i for i in range(len(gamma)) if gamma[i] is not None]

# Llistes amb només els bins vàlids
dist_bins_valid = dist_bins[:-1][valid_bins]
gamma_valid = np.array(gamma)[valid_bins]

# Visualitzar el variograma experimental només per als bins vàlids
plt.figure(figsize=(8, 6))
plt.plot(dist_bins_valid, gamma_valid, 'o-', label='Variograma Experimental')
plt.xlabel('Distància (km)')
plt.ylabel('Semi-variància')
plt.title('Variograma Experimental')
plt.legend()
plt.savefig('variograma_experimental.png')

# Ajustar el model de variograma esferic
sill = np.var(no2_vals)
range_guess = np.max(distances) / 2  # Pista inicial del range
nugget = 5  # Valor inicial del nugget  

print(sill, range_guess, nugget)

# Crear l'interpolador Kriging
ok = OrdinaryKriging(longituds, latituds, no2_vals, variogram_model='spherical', 
                     variogram_parameters={'sill': sill, 'range': range_guess, 'nugget': nugget})