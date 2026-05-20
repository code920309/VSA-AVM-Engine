import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# 경로 및 설정
INPUT_PATH = 'data/processed/nationwide_RHTrade_final_features.csv'
REPORT_PATH = 'advanced_model_report.md'
MODEL_PATH = 'models/advanced_avm_model.pkl'
VISUALS_DIR = 'visuals'
ENCODING = 'utf-8-sig'

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_vsa_advanced():
    print("--- [1] 고도화 피처 엔지니어링 및 데이터 준비 ---")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: 파일을 찾을 수 없습니다. {INPUT_PATH}")
        return
    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)
    if not os.path.exists('models'):
        os.makedirs('models')

    df = pd.read_csv(INPUT_PATH, encoding=ENCODING, low_memory=False)
    
    # [고도화 피처]
    df['is_high_floor_without_elevator'] = ((df['floor'] >= 4) & (df['has_elevator'] == 0)).astype(int)
    df['area_ratio'] = df['excluUseAr'] / (df['landAr'] + 1e-5)
    
    features = [
        'excluUseAr', 'landAr', 'floor', 'Age', 'land_index', 
        'parking_ratio', 'has_elevator', 'total_floors', 'subway_dist', 
        'houseType_연립', 'umdNm_encoded', 'is_high_floor_without_elevator', 'area_ratio'
    ]
    target = 'Price_per_m2_log'
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 모델 학습
    print("\n--- [2] Advanced LightGBM 학습 및 최적화 진행 중 ---")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=64,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True
    )
    
    callbacks = [early_stopping(stopping_rounds=50), log_evaluation(period=100)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=callbacks
    )

    # 3. 모델 파일 추출 (저장)
    print("\n--- [3] 모델 파일 추출 및 시각화 저장 중 ---")
    joblib.dump(model, MODEL_PATH)
    print(f" > 모델 파일 저장 완료: {MODEL_PATH}")

    # 4. 성능 평가 및 보고서 갱신
    y_pred_log = model.predict(X_test)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mape = calculate_mape(y_test_real, y_pred_real)

    # 시각화 저장 (산점도 및 중요도)
    plt.figure(figsize=(10, 8))
    sns.regplot(x=y_test_real, y=y_pred_real, scatter_kws={'alpha':0.1, 's':1}, line_kws={'color':'blue'})
    plt.savefig(os.path.join(VISUALS_DIR, 'advanced_pred_vs_actual.png'))
    plt.close()

    importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(importances['Feature'], importances['Importance'], color='salmon')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'advanced_feature_importance.png'))
    plt.close()

    print(f"\n--- [종료] 모델 추출 및 리포트 최신화 완료 ---")
    print(f" > R2: {r2:.4f}, MAPE: {mape:.2f}%")

if __name__ == "__main__":
    train_vsa_advanced()
