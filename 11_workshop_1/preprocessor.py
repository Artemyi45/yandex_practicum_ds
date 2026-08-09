# heart_predictor/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Класс для предобработки данных
    """
    
    def __init__(self):
        self.preprocessor = None
        self.is_fitted = False
        print("🔄 Создан препроцессор данных")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Обучает препроцессор на данных
        """
        print("🎯 Обучаем препроцессор...")
        
        # Списки признаков по типам
        numeric_features = [
            'Возраст', 'Холестерин', 'Тренировки в неделю (часы)', 'Доход', 'ИМТ',
            'Триглицериды', 'Часов сна в день', 'Уровень сахара в крови',
            'КФК-МБ', 'Тропонин', 'Систолическое давление', 'Диастолическое давление'
        ]
        
        binary_features = [
            'Диабет', 'Семейная история', 'Курение', 'Ожирение', 'Алкоголь',
            'Проблемы с сердцем в прошлом', 'Приём лекарств', 'Пол'
        ]
        
        categorical_features = ['Тип питания']
        
        # Создаем пайплайны для каждого типа признаков
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Объединяем все пайплайны
        self.preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_features),
            ('binary', binary_pipeline, binary_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        # Обучаем препроцессор
        self.preprocessor.fit(X)
        self.is_fitted = True
        
        print("✅ Препроцессор обучен")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Преобразует данные
        """
        if not self.is_fitted:
            raise ValueError("❌ Препроцессор не обучен! Сначала вызови fit()")
        
        print("🔄 Преобразуем данные...")
        transformed_data = self.preprocessor.transform(X)
        print(f"✅ Данные преобразованы: {transformed_data.shape}")
        
        return transformed_data
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Обучает и преобразует данные за один шаг
        """
        return self.fit(X, y).transform(X)