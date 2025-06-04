import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.keras
from dotenv import load_dotenv
import pickle
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#Carga variables de entorno
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def procesar_dataset_completo(df_original, df_filtrado, nombre_target):
    columnas_modelo = [
        'fecha_embarque','tipo_agrupacion',
        'dia_del_anio_sin','dia_del_anio_cos',
        'dia_embarque_sin','dia_embarque_cos',
        'hora_embarque_sin','hora_embarque_cos',
        'is_weekend','mes_embarque_sin','mes_embarque_cos',
        'week_of_year_sin','week_of_year_cos',
        'season_1','season_2','season_3','season_4',
        'is_festivo_nacional','is_festivo_local',
        'is_eid_aladha','is_eid_aladha_prev','is_eid_aladha_post',
        'is_eid_alfitr','is_eid_alfitr_prev','is_eid_alfitr_post',
        'is_mawlid_nabi',
        'is_monday','is_tuesday','is_wednesday','is_thursday',
        'is_friday','is_saturday','is_sunday',
        'weekday_sin','weekday_cos'
    ]
    df_orig = df_original[columnas_modelo].copy()
    df_filt = df_filtrado[columnas_modelo].copy()
    fechas = pd.date_range(
        start=df_orig['fecha_embarque'].min(),
        end=df_orig['fecha_embarque'].max(),
        freq='D'
    )
    df_daily = (
        df_filt.groupby('fecha_embarque')
               .size()
               .reindex(fechas, fill_value=0)
               .reset_index(name=nombre_target)
               .rename(columns={'index':'fecha_embarque'})
    )
    df_temp = (
        df_orig.drop(columns=['tipo_agrupacion'])
               .drop_duplicates('fecha_embarque')
               .set_index('fecha_embarque')
               .reindex(fechas)
               .reset_index()
               .rename(columns={'index':'fecha_embarque'})
    )
    for col in df_temp.columns:
        if col != 'fecha_embarque':
            df_temp[col] = df_temp[col].fillna(method='ffill').fillna(0)
    df_final = pd.merge(df_temp, df_daily, on='fecha_embarque', how='left').fillna({nombre_target:0})
    df_final = df_final.sort_values('fecha_embarque')
    scaler = MinMaxScaler()
    df_final[f'{nombre_target}_norm'] = scaler.fit_transform(df_final[[nombre_target]])
    return df_final, scaler

def crear_secuencias(df, target_col, lookback=7):
    features = df.columns.difference(['fecha_embarque', target_col, f'{target_col}_norm'])
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[i-lookback:i][features].values.astype('float32'))
        y.append(df.iloc[i][f'{target_col}_norm'])
    return np.array(X), np.array(y)

def split_temporal(X, y, ratios=(0.7, 0.15, 0.15)):
    n = len(X)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])

def crear_modelo_lstm(input_shape, dropout_rate=0.1, lstm_units=64, bidirectional=True):
    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units // 2)))
    else:
        model.add(LSTM(lstm_units // 2))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model

def compilar_modelo(model, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
    return model

def crear_callbacks(model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    early = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
    
    #Checkpoint para guardar pesos
    ckpt_weights = os.path.join(output_dir, f'{model_name}_weights.h5')
    chk = ModelCheckpoint(
        ckpt_weights,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    #Checkpoint para guardar modelo completo
    ckpt_model = os.path.join(output_dir, f'{model_name}.h5')
    chk_model = ModelCheckpoint(
        ckpt_model,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-7, verbose=1)
    return [early, chk, chk_model, reduce_lr]

def calculate_metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-10))) * 100
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - ss_res/ss_tot
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2}

def guardar_configuracion_modelo(config, output_dir, nombre_modelo):
    """Guarda la configuración del modelo para reproducibilidad"""
    config_path = os.path.join(output_dir, f'config_{nombre_modelo}.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f)
    return config_path

def train_lstm(
    input_path="data/processed/dataset_empresa_03_pos_EDA_2.csv",
    output_model_dir="models",
    lookback=7,
    lstm_units=64,
    dropout_rate=0.1,
    learning_rate=0.001,
    epochs=300,
    batch_size=32
):
    mlflow.set_tracking_uri(MLFLOW_URI)
    results = {}
    
    #Guardar configuracion general
    config_general = {
        "lookback": lookback,
        "lstm_units": lstm_units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_path": input_path,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    #Cargar dataset
    df = pd.read_csv(input_path, low_memory=False)
    df['fecha_embarque'] = pd.to_datetime(df['fecha_embarque'], errors='coerce')

    #ENTRENAMIENTO DE PASAJEROS
    with mlflow.start_run(run_name="Training_(P)LSTM_Run"):
        mlflow.set_tags({
            "step": "train_pasajeros",
            "model_type": "LSTM",
            "dataset_version": "v1.0",
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Log parameters
        mlflow.log_params({
            "lookback": lookback,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size
        })
        
        #Procesar dataset pasajeros
        df_p = df[df['tipo_agrupacion'] == 1]
        df_p_daily, scaler_p = procesar_dataset_completo(df, df_p, 'total_pasajeros')
        
        #Guardar columnas y metadatos 
        feature_cols = df_p_daily.columns.difference(['fecha_embarque', 'total_pasajeros', 'total_pasajeros_norm']).tolist()
        config_pasajeros = config_general.copy()
        config_pasajeros.update({
            "feature_columns": feature_cols,
            "input_shape": [lookback, len(feature_cols)],
            "target_column": "total_pasajeros"
        })
        config_path_p = guardar_configuracion_modelo(config_pasajeros, output_model_dir, 'pasajeros')
        mlflow.log_artifact(config_path_p)
        
        #Crear secuencias y split
        Xp, yp = crear_secuencias(df_p_daily, 'total_pasajeros', lookback)
        (Xp_tr, yp_tr), (Xp_val, yp_val), (Xp_ts, yp_ts) = split_temporal(Xp, yp)
        
        #Crear y entrenar modelo
        input_shape = (Xp_tr.shape[1], Xp_tr.shape[2])
        model = crear_modelo_lstm(input_shape, dropout_rate, lstm_units)
        model = compilar_modelo(model, learning_rate)
        callbacks = crear_callbacks("lstm_model_pasajeros", output_model_dir)
        history = model.fit(
            Xp_tr, yp_tr,
            validation_data=(Xp_val, yp_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        #Guardar modelo completo en formato SavedModel (mas completo que h5)
        saved_model_path = os.path.join(output_model_dir, 'lstm_model_pasajeros_savedmodel')
        model.save(saved_model_path, save_format='tf')
        
        #Guardar historico y scaler
        hist_path = os.path.join(output_model_dir, 'history_lstm_pasajeros.pkl')
        with open(hist_path, 'wb') as f:
            pickle.dump(history.history, f)
        scaler_path_p = os.path.join(output_model_dir, 'scaler_pasajeros.pkl')
        joblib.dump(scaler_p, scaler_path_p)
        
        #Guardar feature_columns
        features_path = os.path.join(output_model_dir, 'feature_columns_pasajeros.pkl')
        joblib.dump(feature_cols, features_path)
        
        #Log artifacts con MLflow
        mlflow.log_artifact(hist_path, artifact_path='history')
        mlflow.log_artifact(scaler_path_p, artifact_path='scalers')
        mlflow.log_artifact(features_path, artifact_path='features')
        mlflow.log_artifact(saved_model_path, artifact_path='saved_model')
        
        #Registrar modelo con MLflow
        mlflow.keras.log_model(
            model,
            artifact_path="model_pasajeros",
            registered_model_name="lstm_pasajeros"
        )
        
        #Prediccion y metricas
        y_pred_norm = model.predict(Xp_ts)
        y_pred = scaler_p.inverse_transform(y_pred_norm.reshape(-1,1)).flatten()
        y_true = scaler_p.inverse_transform(yp_ts.reshape(-1,1)).flatten()
        metrics_p = calculate_metrics(y_true, y_pred)
        
        #Log metrics to MLflow
        for metric_name, metric_value in metrics_p.items():
            mlflow.log_metric(metric_name, metric_value)
        
        results['pasajeros'] = metrics_p

    #ENTRENAMIENTO DE VEHICULOS
    with mlflow.start_run(run_name="Training_(V)LSTM_Run"):
        mlflow.set_tags({
            "step": "train_vehiculos",
            "model_type": "LSTM",
            "dataset_version": "v1.0",
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        #Log parameters
        mlflow.log_params({
            "lookback": lookback,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size
        })
        
        #Procesar dataset vehculos
        df_v = df[df['tipo_agrupacion'] == 0]
        df_v_daily, scaler_v = procesar_dataset_completo(df, df_v, 'total_vehiculos')
        
        #Guardar columnas y metadatos 
        feature_cols_v = df_v_daily.columns.difference(['fecha_embarque', 'total_vehiculos', 'total_vehiculos_norm']).tolist()
        config_vehiculos = config_general.copy()
        config_vehiculos.update({
            "feature_columns": feature_cols_v,
            "input_shape": [lookback, len(feature_cols_v)],
            "target_column": "total_vehiculos"
        })
        config_path_v = guardar_configuracion_modelo(config_vehiculos, output_model_dir, 'vehiculos')
        mlflow.log_artifact(config_path_v)
        
        #Crear secuencias y split
        Xv, yv = crear_secuencias(df_v_daily, 'total_vehiculos', lookback)
        (Xv_tr, yv_tr), (Xv_val, yv_val), (Xv_ts, yv_ts) = split_temporal(Xv, yv)
        
        #Crear y entrenar modelo
        input_shape_v = (Xv_tr.shape[1], Xv_tr.shape[2])
        model_v = crear_modelo_lstm(input_shape_v, dropout_rate, lstm_units)
        model_v = compilar_modelo(model_v, learning_rate)
        callbacks_v = crear_callbacks("lstm_model_vehiculos", output_model_dir)
        history_v = model_v.fit(
            Xv_tr, yv_tr,
            validation_data=(Xv_val, yv_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_v,
            verbose=1
        )
        
        #Guardar modelo completo en formato SavedModel
        saved_model_path_v = os.path.join(output_model_dir, 'lstm_model_vehiculos_savedmodel')
        model_v.save(saved_model_path_v, save_format='tf')
        
        #Guardar historico y scaler
        hist_path_v = os.path.join(output_model_dir, 'history_lstm_vehiculos.pkl')
        with open(hist_path_v, 'wb') as f:
            pickle.dump(history_v.history, f)
        scaler_path_v = os.path.join(output_model_dir, 'scaler_vehiculos.pkl')
        joblib.dump(scaler_v, scaler_path_v)
        
        #Guardar feature_columns
        features_path_v = os.path.join(output_model_dir, 'feature_columns_vehiculos.pkl')
        joblib.dump(feature_cols_v, features_path_v)
        
        #Log artifacts con MLflow
        mlflow.log_artifact(hist_path_v, artifact_path='history')
        mlflow.log_artifact(scaler_path_v, artifact_path='scalers')
        mlflow.log_artifact(features_path_v, artifact_path='features')
        mlflow.log_artifact(saved_model_path_v, artifact_path='saved_model')
        
        #Registrar modelo con MLflow
        mlflow.keras.log_model(
            model_v,
            artifact_path="model_vehiculos",
            registered_model_name="lstm_vehiculos"
        )
        
        #Prediccion y metricas
        y_pred_norm_v = model_v.predict(Xv_ts)
        y_pred_v = scaler_v.inverse_transform(y_pred_norm_v.reshape(-1,1)).flatten()
        y_true_v = scaler_v.inverse_transform(yv_ts.reshape(-1,1)).flatten()
        metrics_v = calculate_metrics(y_true_v, y_pred_v)
        
        #Log metrics to MLflow
        for metric_name, metric_value in metrics_v.items():
            mlflow.log_metric(metric_name, metric_value)
            
        results['vehiculos'] = metrics_v

    return results


def run_train_lstm():
    return train_lstm()

if __name__ == '__main__':
    res = run_train_lstm()
    print('Metricas Pasajeros:', res['pasajeros'])
    print('Metricas Vehículos:', res['vehiculos'])