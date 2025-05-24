import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

# # Load and prepare data (as in your previous steps)
# oecd_bli = pd.read_csv("OECD,DF_BLI,+all.csv", thousands=',')
# gdp_per_capita = pd.read_excel("customers-100.csv", na_values='n/a')

# oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
# oecd_bli = oecd_bli[oecd_bli["Indicator"] == "HS_LEB"]  # Using life expectancy as proxy for now
# oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Observation Value")
# gdp_per_capita = gdp_per_capita[gdp_per_capita["Subject Descriptor"] == "Gross domestic product per capita, current prices"]
# gdp_per_capita = pd.DataFrame({"Country": gdp_per_capita["Country"], "GDP per capita": gdp_per_capita["2015"]})
# country_stats = pd.merge(oecd_bli, gdp_per_capita, on="Country", how="inner")
# country_stats = country_stats.rename(columns={"HS_LEB": "Life satisfaction"})

# # Prepare features (X) and target (y)
# X = np.c_[country_stats["GDP per capita"]]
# y = np.c_[country_stats["Life satisfaction"]]

# # Train k-NN regression model with k=3
# knn_model = KNeighborsRegressor(n_neighbors=3)  # Set k=3
# knn_model.fit(X, y)

# # Make a prediction for Cyprus
# X_new = [[22587]]  # Cyprus's GDP per capita
# prediction = knn_model.predict(X_new)
# print("Predicted Life satisfaction for Cyprus:", prediction[0][0])

# # Optional: Plot the data and prediction
# plt.scatter(X, y, color='blue', label="Data points")
# plt.scatter(X_new, prediction, color='red', label="Cyprus prediction")
# plt.xlabel("GDP per capita")
# plt.ylabel("Life satisfaction")
# plt.legend()
# plt.show()

# Step 1: Simulate the OECD Better Life Index data (as before)
# data_bli = {
#     "Country": ["France", "Germany", "Italy", "Japan", "Canada", "Australia", "United States", "Spain"],
#     "Inequality": ["TOT"] * 8,
#     "Indicator": ["HS_LEB"] * 8,
#     "Observation Value": [82.9, 81.4, 81.7, 86.3, 82.8, 83.0, 78.5, 83.9]
# }
# oecd_bli = pd.DataFrame(data_bli)

# # Step 2: Simulate the GDP per capita data (replace customers-100.csv)
# data_gdp = {
#     "Country": ["France", "Germany", "Italy", "Japan", "Canada", "Australia", "United States", "Spain"],
#     "Subject Descriptor": ["Gross domestic product per capita, current prices"] * 8,
#     "2015": [37000, 45000, 31000, 34000, 41000, 50000, 56000, 29000]
# }
# gdp_per_capita = pd.DataFrame(data_gdp)

# # Proceed with filtering and merging
# oecd_bli = oecd_bli[oecd_bli["Inequality"] == "TOT"]
# oecd_bli = oecd_bli[oecd_bli["Indicator"] == "HS_LEB"]
# oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Observation Value")
# gdp_per_capita = gdp_per_capita[gdp_per_capita["Subject Descriptor"] == "Gross domestic product per capita, current prices"]
# gdp_per_capita = pd.DataFrame({"Country": gdp_per_capita["Country"], "GDP per capita": gdp_per_capita["2015"]})

# country_stats = pd.merge(oecd_bli, gdp_per_capita, on="Country", how="inner")
# country_stats = country_stats.rename(columns={"HS_LEB": "Life satisfaction"})

# # Prepare features (X) and target (y)
# X = np.c_[country_stats["GDP per capita"]]
# y = np.c_[country_stats["Life satisfaction"]]

# # Train k-NN regression model with k=3
# knn_model = KNeighborsRegressor(n_neighbors=3)
# knn_model.fit(X, y)

# # Make prediction for Cyprus (example GDP per capita = 22587)
# X_new = [[22587]]  # Cyprus's GDP per capita
# prediction = knn_model.predict(X_new)
# print("Predicted Life satisfaction for Japan:", prediction[0][0])

# # Optional: Plot the data and prediction
# plt.scatter(X, y, color='blue', label="Data points")
# plt.scatter(X_new, prediction, color='red', label="Cyprus prediction")
# plt.xlabel("GDP per capita")
# plt.ylabel("Life satisfaction")
# plt.title("GDP vs Life Satisfaction (k-NN Prediction)")
# plt.legend()
# plt.grid(True)
# plt.show()












