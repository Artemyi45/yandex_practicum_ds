# Предсказание риска сердечного приступа

## Задача
Разработать модель машинного обучения для предсказания риска сердечного приступа на основе медицинских данных пациентов. Минимизировать пропуск пациентов с реальным риском (максимизация Recall).

## Данные
- **Обучающий датасет:** `heart_train.csv` — 8 685 записей, 28 признаков.
- **Тестовый датасет:** `heart_test.csv` — 966 записей, 27 признаков (без целевого признака).
- **Признаки:** демография, антропометрия, привычки, медицинские показатели, история болезней.

## Модель
- **Метрика:** Recall (важно не пропустить пациентов с риском).
- **Алгоритмы:** LogisticRegression, DecisionTree, RandomForest.
- **Итоговая модель:** RandomForestClassifier (Recall = 0.950, ROC-AUC = 0.949).
- **Файл модели:** `best_heart_model.joblib`.

## Структура проекта
├── README.md
├── requirements.txt
├── app.py
├── test.py
├── Heart_project.ipynb
├── best_heart_model.joblib
├── heart_train.csv
├── heart_test.csv
├── heart_attack_predictions.csv
├── heart_predictor/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── model.py
└── ТЗ.docx

## Запуск
pip install -r requirements.txt
uvicorn app:app --reload

Открыть в браузере: http://localhost:8000/docs

POST /predict — загрузить CSV-файл с тестовыми данными.

Ответ (JSON):
{"predictions": [{"id": 1, "prediction": 0}, {"id": 2, "prediction": 1}]}

## Классы и методы
DataLoader — load_train_data(path), load_test_data(path).
DataPreprocessor — fit(X), transform(X), fit_transform(X).
HeartAttackModel — fit(X, y), predict(X), predict_proba(X), evaluate(X, y).

## Результаты
| Модель | Recall | Precision | F1 | ROC-AUC |
| LogisticRegression | 0.811 | 0.368 | 0.506 | 0.525 |
| DecisionTree | 0.907 | 0.357 | 0.512 | 0.504 |
| RandomForest | 0.950 | 0.684 | 0.795 | 0.949 |

## Стек
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, FastAPI, Uvicorn