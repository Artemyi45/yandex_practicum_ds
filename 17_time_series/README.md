# Прогнозирование заказов такси

## 🎯 Задача
Спрогнозировать количество заказов такси на следующий час. Порог заказчика: RMSE ≤ 48.

## 📊 Данные
- `taxi.csv` — 26 496 записей о заказах такси (март–август 2018).
- Данные агрегированы до часовых интервалов.

## 🧠 Ход работы
1. **Загрузка и ресемплирование** — группировка по часу.
2. **EDA** — тренд, сезонность (суточная/недельная), ACF/PACF, стационарность.
3. **Feature Engineering** — лаги (1-12, 24, 48, 168), скользящие средние и std.
4. **Модели** — LinearRegression, SARIMA, CatBoost.
5. **Подбор гиперпараметров** — CatBoost (grid search по iterations, learning_rate, depth).
6. **Оценка** — RMSE на тестовой выборке.

## 📈 Результаты

| Модель | RMSE (val) |
|:---|---:|
| SARIMA | 92.98 |
| LinearRegression | 30.64 |
| **CatBoost (tuned)** | **30.06** |

**RMSE на тесте: 43.84** (порог 48 пройден).

## 🔧 Стек
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, CatBoost, Statsmodels

## 🚀 Запуск
Открыть `taxi_order_forecasting.ipynb` в Jupyter Notebook.