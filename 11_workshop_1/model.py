# heart_predictor/model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import joblib

class HeartAttackModel:
    """
    Главный класс модели для предсказания риска сердечного приступа
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        print(f"🔄 Создана модель: {model_type}")
    
    def _create_model(self):
        """Создает модель машинного обучения"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError("Поддерживается только random_forest")
    
    def fit(self, X: np.ndarray, y: pd.Series):
        """
        Обучает модель на данных
        """
        print("🎯 Обучаем модель...")
        
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_trained = True
        
        print("✅ Модель обучена")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания
        """
        if not self.is_trained:
            raise ValueError("❌ Модель не обучена! Сначала вызови fit()")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности предсказаний
        """
        if not self.is_trained:
            raise ValueError("❌ Модель не обучена! Сначала вызови fit()")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X: np.ndarray, y: pd.Series) -> dict:
        """
        Оценивает качество модели
        """
        if not self.is_trained:
            raise ValueError("❌ Модель не обучена! Сначала вызови fit()")
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        metrics = {
            'recall': recall_score(y, predictions),
            'precision': precision_score(y, predictions),
            'f1': f1_score(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities[:, 1])
        }
        
        return metrics
    
    def save(self, file_path: str):
        """Сохраняет модель в файл"""
        if not self.is_trained:
            raise ValueError("❌ Нельзя сохранить необученную модель")
        
        joblib.dump(self.model, file_path)
        print(f"💾 Модель сохранена в {file_path}")
    
    def load(self, file_path: str):
        """Загружает модель из файла"""
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"📥 Модель загружена из {file_path}")