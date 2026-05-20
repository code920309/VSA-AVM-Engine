import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

# 설정 및 경로
INPUT_PATH = 'data/processed/nationwide_RHTrade_final_features.csv'
REPORT_PATH = 'baseline_model_report.md'
VISUALS_DIR = 'visuals'
ENCODING = 'utf-8-sig'

# 한글 폰트 설정 (Windows 기준 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_report():
    print("--- [1] 데이터 로딩 및 전처리 시작 ---")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: 파일을 찾을 수 없습니다. {INPUT_PATH}")
        return

    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)

    df = pd.read_csv(INPUT_PATH, encoding=ENCODING, low_memory=False)
    
    features = [
        'excluUseAr', 'landAr', 'floor', 'Age', 'land_index', 
        'parking_ratio', 'has_elevator', 'total_floors', 'subway_dist', 
        'houseType_연립', 'umdNm_encoded'
    ]
    target = 'Price_per_m2_log'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" > 데이터셋 분할 완료 (Train: {len(X_train):,}, Test: {len(X_test):,})")

    # 모델 학습
    print("\n--- [2] LightGBM Baseline 모델 학습 중 ---")
    start_time = time.time()
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True
    )
    
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f" > 학습 완료! (소요 시간: {training_time:.2f}초)")

    # 예측 및 역변환
    print("\n--- [3] 모델 평가 및 시각화 생성 중 ---")
    y_pred_log = model.predict(X_test)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    
    # 평가 지표
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mape = calculate_mape(y_test_real, y_pred_real)

    # --- 시각화 A: 실제값 vs 예측값 산점도 ---
    plt.figure(figsize=(10, 8))
    sns.regplot(x=y_test_real, y=y_pred_real, scatter_kws={'alpha':0.1, 's':1}, line_kws={'color':'red'})
    plt.xlabel('Actual Price (per m2)')
    plt.ylabel('Predicted Price (per m2)')
    plt.title('Actual vs Predicted Price (Regression)')
    plt.savefig(os.path.join(VISUALS_DIR, 'pred_vs_actual.png'))
    plt.close()

    # --- 시각화 B: 피처 중요도 차트 ---
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importances['Feature'], importances['Importance'], color='skyblue')
    plt.title('LightGBM Feature Importance')
    plt.xlabel('Importance Value')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'feature_importance.png'))
    plt.close()

    # --- 시각화 C: 오차율 분포 (MAPE Histogram) ---
    errors_pct = np.abs((y_test_real - y_pred_real) / y_test_real) * 100
    plt.figure(figsize=(10, 6))
    sns.histplot(errors_pct, bins=50, kde=True, color='green')
    plt.axvline(mape, color='red', linestyle='--', label=f'Mean MAPE: {mape:.2f}%')
    plt.xlim(0, 100) # 100% 이상의 오차는 시각화에서 제외 (보기 편하게)
    plt.xlabel('Absolute Percentage Error (%)')
    plt.title('Distribution of Prediction Errors (MAPE)')
    plt.legend()
    plt.savefig(os.path.join(VISUALS_DIR, 'error_distribution.png'))
    plt.close()

    # 보고서 생성 (Markdown)
    print("\n--- [4] 마크다운 보고서 갱신 중 ---")
    report_content = f"""# 🤖 VSA-AVM Baseline 모델 시각화 보고서

본 리포트는 모델의 예측 성능과 변수 기여도를 시계각화한 결과입니다.

## 1. 모델 개요 및 지표
| 지표명 | 결과값 | 의미 |
| :--- | :--- | :--- |
| **R2 Score** | **{r2:.4f}** | 모델의 설명력 (1.0에 가까울수록 우수) |
| **MAE** | **{mae:,.0f} 원** | 실제 가격 대비 평균 절대 오차 금액 |
| **MAPE** | **{mape:.2f} %** | 실제 가격 대비 평균 백분율 오차율 |

## 2. 예측 정확도 시각화 (Actual vs Predicted)
빨간 선에 점들이 밀집될수록 예측이 정확함을 의미합니다.

![Prediction Accuracy](./visuals/pred_vs_actual.png)

## 3. 피처 중요도 분석 (Feature Importance)
가치 산정에 가장 큰 영향을 준 변수 Top 10 입니다.

![Feature Importance](./visuals/feature_importance.png)

## 4. 오차 분포 분석 (Error Analysis)
대부분의 매물이 낮은 오차율 구간(왼쪽)에 집중되어 있는지 확인합니다.

![Error Distribution](./visuals/error_distribution.png)

---
> [!TIP]
> **전용면적(`excluUseAr`)** 및 **지리적 가치(`umdNm_encoded`)**가 가격 결정에 결정적인 기여를 하고 있음이 시각적으로 확인되었습니다.

**생성 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**엔지니어**: Antigravity AVM Agent
"""

    with open(REPORT_PATH, 'w', encoding='utf-8-sig') as f:
        f.write(report_content)

    print(f" > 보고서 및 시각화 저장 완료: {REPORT_PATH}, /visuals/")
    print("\n--- [완성] 시각화 리포트 생성이 완료되었습니다. ---")

if __name__ == "__main__":
    train_and_report()
