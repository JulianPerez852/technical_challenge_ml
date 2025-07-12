import joblib
import numpy as np
import pandas as pd
import os
import sys
try:
    from repositories.title_model_trainer import ModeloHibridoOptimizado
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from repositories.title_model_trainer import ModeloHibridoOptimizado


class TitleProcessor:
    def __init__(self, model_path, scaler_path, encoder_path=None):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Error loading scaler from {scaler_path}: {e}")
        if encoder_path:
            try:
                self.encoder = joblib.load(encoder_path)
            except Exception as e:
                raise RuntimeError(f"Error loading encoder from {encoder_path}: {e}")
        else:
            self.encoder = None
        try:
            self.feature_engineer = ModeloHibridoOptimizado()
        except Exception as e:
            raise RuntimeError(f"Error initializing feature engineer: {e}")

    def predict_title(self, title):
        try:
            df_temp = pd.DataFrame({'title': [title]})
            X_temp, _ = self.feature_engineer.preparar_datos_completos(df_temp)
            X_temp = np.array(X_temp, dtype=np.float32)
            try:
                X_temp = self.scaler.transform(X_temp)
            except Exception:
                pass
            pred_encoded = self.model.predict(X_temp)[0]
            proba = self.model.predict_proba(X_temp)[0]
            if self.encoder is not None:
                categoria = self.encoder.inverse_transform([pred_encoded])[0]
            else:
                categoria = pred_encoded
            return categoria, proba
        except Exception as e:
            raise RuntimeError(f"Error predicting title '{title}': {e}")

    def predict_titles(self, titles):
        try:
            df_temp = pd.DataFrame({'title': titles})
            X_temp, _ = self.feature_engineer.preparar_datos_completos(df_temp)
            X_temp = np.array(X_temp, dtype=np.float32)
            X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=1e6, neginf=-1e6)
            try:
                X_temp = self.scaler.transform(X_temp)
            except Exception:
                pass
            preds_encoded = self.model.predict(X_temp)
            probas = self.model.predict_proba(X_temp)
            if self.encoder is not None:
                categorias = self.encoder.inverse_transform(preds_encoded)
            else:
                categorias = preds_encoded
            return categorias, probas
        except Exception as e:
            raise RuntimeError(f"Error predicting titles: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    processor = TitleProcessor(
        model_path='../../data/modelo_final.joblib',
        scaler_path='../../data/modelo_scaler.joblib',
        encoder_path='../../data/modelo_encoder.joblib'
    )
    titulo = "Samsung Galaxy S21 256GB Nuevo Sellado Garantía"
    categoria, probabilidades = processor.predict_title(titulo)
    print(f"Categoría: {categoria}")
    print(f"Probabilidades: {probabilidades}")
