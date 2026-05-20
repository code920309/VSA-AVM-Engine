import pandas as pd
import numpy as np
import os

# 데이터 경로 및 설정
INPUT_PATH = 'data/processed/nationwide_RHTrade_enriched.csv'
OUTPUT_PATH = 'data/processed/nationwide_RHTrade_processed_target.csv'
ENCODING = 'utf-8-sig'

def process_vsa_target_variable(file_path):
    """
    아웃라이어 필터링, houseType 텍스트 정제, 시점수정 반영 및 로그 변환 타겟 변수를 산출합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. 경로: {file_path}")
        return

    # 1. 데이터 로드
    print("--- [1] 데이터 로드 및 초기화 ---")
    df = pd.read_csv(file_path, encoding=ENCODING, low_memory=False)
    initial_count = len(df)
    print(f" > 초기 데이터 로드 완료: {initial_count:,} 행")

    # 2. 아웃라이어 전면 필터링 (1단계 진단 결과 반영)
    print("\n--- [2] 아웃라이어 필터링 (필터 범위: 3,900 ~ 82,000) ---")
    lower_bound = 3900
    upper_bound = 82000
    df_filtered = df[(df['dealAmount'] >= lower_bound) & (df['dealAmount'] <= upper_bound)].copy()
    
    # [추가 보완] 데이터 텍스트 노이즈 정제 ('연립다세대' -> '연립' 통일)
    if 'houseType' in df_filtered.columns:
        df_filtered['houseType'] = df_filtered['houseType'].replace('연립다세대', '연립')
        print(" > 데이터 정제: '연립다세대' 텍스트를 '연립'으로 통일 완료.")
        
    filtered_count = len(df_filtered)
    dropped_count = initial_count - filtered_count
    print(f" > 필터링 후 데이터 수: {filtered_count:,} 행")
    print(f" > 제거된 이상치 수: {dropped_count:,} 행 (전체의 {dropped_count/initial_count*100:.2f}%)")

    # 3. 시점수정(Time-point Correction) 및 실질 단가 계산
    print("\n--- [3] 시점수정 반영 및 실질 가격(㎡당 단가) 산출 ---")
    # 명목 단가 계산 (단위: 원/㎡)
    df_filtered['nominal_price_per_m2'] = (df_filtered['dealAmount'] * 10000) / df_filtered['excluUseAr']
    
    # 기준 시점(2026.03) 가치로 보정된 실질 단가
    df_filtered['adjusted_price_per_m2'] = df_filtered['nominal_price_per_m2'] * df_filtered['Time_Adjustment']
    print(" > 시점수정 완료 (nominal_price -> adjusted_price_per_m2)")

    # 4. 타겟 변수 로그 변환 및 입지 피처 안정화
    print("\n--- [4] 타겟 변수 로그 변환 및 피처 안정화 ---")
    # 왜도(Skewness) 완화 및 MAPE 최적화를 위한 np.log1p 적용
    df_filtered['Price_per_m2_log'] = np.log1p(df_filtered['adjusted_price_per_m2'])
    
    # [추가 보완] 비역세권 대중교통 거리 변수 캡핑 처리 (최대 1.5km로 제한하여 오학습 방지)
    if 'subway_dist' in df_filtered.columns:
        df_filtered['subway_dist'] = np.minimum(df_filtered['subway_dist'], 1500)
        print(" > 피처 엔지니어링: 지하철 최단 거리(subway_dist) 1,500m 상한 캡핑 적용.")
        
    print(" > 최종 타겟 변수 'Price_per_m2_log' 생성 완료.")

    # 5. 최종 기술 통계량 출력 및 저장
    print("\n--- [5] 신규 생성 피처 기술 통계량 ---")
    stats_cols = ['nominal_price_per_m2', 'adjusted_price_per_m2', 'Price_per_m2_log']
    print(df_filtered[stats_cols].describe())

    # 가공된 데이터 저장 실행
    df_filtered.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n > 가공 완료 데이터 저장 완료: {OUTPUT_PATH}")

    return df_filtered

if __name__ == "__main__":
    try:
        processed_df = process_vsa_target_variable(INPUT_PATH)
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
