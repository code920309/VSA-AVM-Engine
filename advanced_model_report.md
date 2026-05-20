# 💎 VSA-AVM 고도화 학습 보고서 (Advanced Model)

본 리포트는 도메인 피처 추가 및 머신러닝 튜닝을 통해 완성도 높은 가치 산정 성능을 도출한 결과입니다.

## 1. 고도화 주요 사항
- **도메인 피처 추가**: 
  - `is_high_floor_without_elevator`: 고층(4층 이상)이면서 엘리베이터가 없는 감가 요소 정밀화
  - `area_ratio`: 토지 지분 대비 전용면적 효율성 반영
- **튜닝 기법**: Early Stopping (50 rounds), n_estimators 확장(2000), Max Depth 제한(10)

## 2. 모델 성능 지표 (Original Scale)
Baseline 대비 향상된 성능 지표입니다.

| 지표명 | 결과값 | 의미 |
| :--- | :--- | :--- |
| **R2 Score** | **0.8824** | 예측 모델의 설명력 |
| **MAE** | **691,694 원** | 평균 오차 금액 |
| **MAPE** | **14.57 %** | 평균 오차율 |

## 3. 고도화 피처의 영향도 시각화
새로운 피처가 가격 산정에 얼마나 기여했는지 확인합니다.

![Advanced Feature Importance](./visuals/advanced_feature_importance.png)

## 4. 예측 정확도 분석 (Scatter Plot)
실제 가격 선(대각선)을 따라 예측 포인트가 얼마나 밀집되었는지 확인합니다.

![Advanced Prediction Analysis](./visuals/advanced_pred_vs_actual.png)

---
> [!IMPORTANT]
> 신규 피처인 **`area_ratio`**가 상위권 중요도를 차지하며 대지지분 효율성이 가격에 큰 영향을 미침이 증명되었습니다. 또한 **`is_high_floor_without_elevator`** 피처가 모델에 학습되어 주거 편의성에 따른 가치 감가가 본격적으로 반영되기 시작했습니다.

**생성 일시**: 2026-05-21 02:23:34
**엔지니어**: Antigravity AVM Agent
