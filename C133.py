from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv("star_with_gravity.csv")

planet_masses=df["Mass"].tolist()
planet_radiuses=df["Radius"].tolist()

X = []
for index, planet_mass in enumerate(planet_masses):
  drawf_planet_list = [
                  planet_radiuses[index],
                  planet_mass
              ]
  X.append(drawf_planet_list)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='blue')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()