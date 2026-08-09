# heart_predictor/data_loader.py
import pandas as pd
from typing import Tuple

class DataLoader:
    """
    Класс для загрузки и проверки данных
    """
    
    def __init__(self):
        self.expected_columns = None
        print("🔄 Создан загрузчик данных")
    
    def load_train_data(self, file_path: str):
        """
        Загружает тренировочные данные
        """
        print(f"📁 Загружаем тренировочные данные из {file_path}")
        
        data = pd.read_csv(file_path)
        print(f"📊 Загружено: {data.shape[0]} строк, {data.shape[1]} колонок")
        
        # ПЕРЕВОДИМ КОЛОНКИ НА РУССКИЙ
        column_translation = {
            'Age': 'Возраст',
            'Cholesterol': 'Холестерин', 
            'Heart rate': 'Пульс',
            'Diabetes': 'Диабет',
            'Family History': 'Семейная история',
            'Smoking': 'Курение',
            'Obesity': 'Ожирение',
            'Alcohol Consumption': 'Алкоголь',
            'Exercise Hours Per Week': 'Тренировки в неделю (часы)',
            'Diet': 'Тип питания',
            'Previous Heart Problems': 'Проблемы с сердцем в прошлом',
            'Medication Use': 'Приём лекарств',
            'Stress Level': 'Уровень стресса',
            'Sedentary Hours Per Day': 'Сидячих часов в день',
            'Income': 'Доход',
            'BMI': 'ИМТ',
            'Triglycerides': 'Триглицериды',
            'Physical Activity Days Per Week': 'Активных дней в неделю',
            'Sleep Hours Per Day': 'Часов сна в день',
            'Blood sugar': 'Уровень сахара в крови',
            'CK-MB': 'КФК-МБ',
            'Troponin': 'Тропонин',
            'Gender': 'Пол',
            'Systolic blood pressure': 'Систолическое давление',
            'Diastolic blood pressure': 'Диастолическое давление'
        }
        
        data = data.rename(columns=column_translation)
        
        # Убираем технические колонки
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        
        # КОДИРУЕМ ТЕКСТ В ЧИСЛА
        if 'Пол' in data.columns:
            data['Пол'] = data['Пол'].map({'Female': 0, 'Male': 1, 'female': 0, 'male': 1})
            data['Пол'] = data['Пол'].fillna(-1)
            print("✅ Пол закодирован в числа")
        
        # Разделяем на X (признаки) и y (целевая переменная)
        X = data.drop(columns=['Heart Attack Risk (Binary)', 'id'])
        y = data['Heart Attack Risk (Binary)']
        
        print("✅ Тренировочные данные успешно загружены и переименованы")
        return X, y
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """
        Загружает тестовые данные
        """
        print(f"📁 Загружаем тестовые данные из {file_path}")
        
        data = pd.read_csv(file_path)
        print(f"📊 Загружено: {data.shape[0]} строк, {data.shape[1]} колонок")
        
        # В тестовых данных убираем id если есть
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
            
        print("✅ Тестовые данные успешно загружены")    
        return data