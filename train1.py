import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

df = pd.read_csv('car data.csv')

# 1. Смотрим на базовые статистические метрики
print("--- Статистика числовых признаков ---")
print(df.describe())

# 2. Смотрим на уникальные значения в категориальных признаках
print("\n--- Уникальные значения в текстовых колонках ---")
for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
    print(f"{col}: {df[col].unique()}")
    print(f"\nВсего уникальных марок машин (Car_Name): {df['Car_Name'].nunique()}")

# переведем год в возраст
df['Age'] = 2026 - df['Year']
# удалим ненужные колонки
df = df.drop(['Year', 'Car_Name'], axis=1)

# перекодируем текстовые колонки в цифры
df = pd.get_dummies(df, drop_first=True)
print(df.head(3))

# подготовка к модели: разделим на треин и тестовую выборки
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# внедряем линейную регрессию
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# внедрим метрики чтобы посмотреть как работает наша модель

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("--- Метрики Линейной Регрессии ---")
print(f"MAE: {mae_lr:.2f}")
print(f"R2 Score: {r2_lr:.4f}")

# применим L2 регуляризацию чтобы избежать скрытых связей

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
# считаем метрики для ридж
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("--- Метрики Ridge Регрессии (L2) ---")
print(f"MAE: {mae_ridge:.2f}")
print(f"R2 Score: {r2_ridge:.4f}")

# теперь воспользуемся Лассо
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("--- Метрики Lasso Регрессии (L1) ---")
print(f"MAE: {mae_lasso:.2f}")
print(f"R2 Score: {r2_lasso:.4f}")

# построим табличку с коэфициентами для: линейной регрессии и лассо регул

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Linear_Reg_Weight': lr_model.coef_,
    'Lasso_Weight': lasso_model.coef_
})

print("--- Сравнение весов признаков ---")
print(coefficients.round(4))

# так как Ridge регул оказалась самой правильной, сделаем вывод с ней в виде графика
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, color='blue', alpha=0.6, label='Предсказания Ridge')
# рисуем идеальный прогноз
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2, label='Идеальный прогноз')
# Добавляем названия осей и заголовок графика
plt.xlabel('Реальная цена машины (в лакхах)', fontsize=12)
plt.ylabel('Предсказанная цена моделью (в лакхах)', fontsize=12)
plt.title(f'Качество предсказания Ridge Регрессии (R2 = {r2_ridge:.4f})', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

