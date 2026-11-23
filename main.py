import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# ==========================================
#
# ==========================================
X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# ЗАВДАННЯ 1:
# ==========================================
model_1 = LinearRegression()
model_1.fit(X_train, y_train)

y_pred_1 = model_1.predict(X_test)
r2_1 = r2_score(y_test, y_pred_1)

features = [f'x{i+1}' for i in range(X.shape[1])]
results_1 = pd.DataFrame({
    'Ознака': features,
    'Коефіцієнт': model_1.coef_
})

print("Результати Завдання 1 (Коефіцієнти):")
print(results_1.to_string(index=False))
print(f"\nR² на тестовій вибірці: {r2_1:.4f}")
print("\n" + "="*50 + "\n")


# ==========================================
# ЗАВДАННЯ 2:
# ==========================================
r2_no_scale = r2_1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

results_2 = pd.DataFrame({
    'Масштабування': ['Ні', 'StandardScaler'],
    'R²': [r2_no_scale, r2_scaled]
})

print("Результати Завдання 2 (Масштабування):")
print(results_2.to_string(index=False))
print("\n" + "="*50 + "\n")


# ==========================================
# ЗАВДАННЯ 3
# ==========================================
r2_no_missing = r2_1

X_train_missing = X_train.copy()
mask = np.random.RandomState(42).rand(*X_train_missing[:, 0].shape) < 0.1
X_train_missing[mask, 0] = np.nan

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_missing)

X_test_missing = X_test.copy()
mask_test = np.random.RandomState(42).rand(*X_test_missing[:, 0].shape) < 0.1
X_test_missing[mask_test, 0] = np.nan
X_test_imputed = imputer.transform(X_test_missing)

model_imputed = LinearRegression()
model_imputed.fit(X_train_imputed, y_train)

y_pred_imputed = model_imputed.predict(X_test_imputed)
r2_imputed = r2_score(y_test, y_pred_imputed)

results_3 = pd.DataFrame({
    'Сценарій': ['Без пропусків', "Після ім'ютації"],
    'R²': [r2_no_missing, r2_imputed]
})

print("Результати Завдання 3 (Пропуски та ім'ютація):")
print(results_3.to_string(index=False))