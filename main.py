"""
Radius-Dependent Plasmonic Transmission in
Gold Nanoparticle-Embedded Hydrogel Systems

This script:
1. Loads transmission data for Au nanoparticles.
2. Analyzes how transmission depends on radius and wavelength.
3. Trains a Random Forest regression model.
4. Evaluates performance and feature importance.

Author: <Your Name>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================
# 1. LOAD DATA
# =============================

df = pd.read_csv("data/transmission_dataset.csv")

print("Dataset preview:")
print(df.head(), "\n")

required_cols = ["radius_nm", "wavelength_nm", "transmission"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# If transmission is in %, convert to 0–1
if df["transmission"].max() > 1.5:
    print("Transmission seems to be in %, converting to 0–1.")
    df["transmission"] = df["transmission"] / 100.0

# Drop any bad rows
df = df.dropna(subset=required_cols).reset_index(drop=True)

print("Dataset size:", df.shape)
print("\nBasic statistics:")
print(df.describe(), "\n")

# =============================
# 2. EXPLORATORY ANALYSIS
# =============================

# Plot transmission vs wavelength for each radius
unique_radii = sorted(df["radius_nm"].unique())
print("Unique radii (nm):", unique_radii)

plt.figure()
for r in unique_radii:
    subset = df[df["radius_nm"] == r].sort_values("wavelength_nm")
    plt.plot(
        subset["wavelength_nm"],
        subset["transmission"],
        label=f"r={r} nm"
    )

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.title("Transmission vs Wavelength for Different Radii (Au NPs)")
plt.legend()
plt.tight_layout()
plt.show()

print("Correlation matrix:")
print(df[required_cols].corr(), "\n")

# =============================
# 3. MACHINE LEARNING MODEL
# =============================

# Features: radius and wavelength
X = df[["radius_nm", "wavelength_nm"]]
y = df["transmission"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest performance:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}\n")

print("Sample Actual vs Predicted (first 10 points):")
for a, p in list(zip(y_test.values, y_pred))[:10]:
    print(f"Actual: {a:.3f} | Predicted: {p:.3f}")

# =============================
# 4. FEATURE IMPORTANCE
# =============================

importances = model.feature_importances_
features = X.columns

print("\nFeature importance:")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.3f}")

plt.figure()
plt.bar(features, importances)
plt.ylabel("Importance")
plt.title("Feature Importance (Radius vs Wavelength)")
plt.tight_layout()
plt.show()

# =============================
# 5. 2D TRANSMISSION MAP
# =============================

pivot = df.pivot_table(
    values="transmission",
    index="radius_nm",
    columns="wavelength_nm"
)

plt.figure()
plt.imshow(
    pivot,
    aspect="auto",
    origin="lower",
    extent=[
        pivot.columns.min(),
        pivot.columns.max(),
        pivot.index.min(),
        pivot.index.max()
    ]
)

plt.colorbar(label="Transmission")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radius (nm)")
plt.title("Transmission Map (Radius vs Wavelength, Au NPs)")
plt.tight_layout()
plt.show()

