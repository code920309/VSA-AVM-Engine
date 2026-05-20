import pandas as pd
import numpy as np
import os

# 데이터 경로 설정
INPUT_PATH = 'data/processed/nationwide_RHTrade_processed_target.csv'
OUTPUT_PATH = 'data/processed/nationwide_RHTrade_final_features.csv'
ENCODING = 'utf-8-sig'

def engineer_vsa_features(file_path):
    """
    3단계: 도메인 기반 결측치 대체 및 범주형 변수 인코딩을 수행합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. 경로: {file_path}")
        return

    # 1. 데이터 로드
    print("--- [1] 2단계 데이터 로드 ---")
    df = pd.read_csv(file_path, encoding=ENCODING, low_memory=False)
    print(f" > 로드 완료: {len(df):,} 행")

    # 2. 도메인 지식 기반 결측치 처리 (Imputation)
    print("\n--- [2] 도메인 기반 결측치 정밀 대체 진행 중 ---")
    
    # [주차비율] 법정동(umdNm)별 평균값으로 채우고, 데이터가 없는 경우 0.5(기본값) 적용
    if 'parking_ratio' in df.columns:
        df['parking_ratio'] = df.groupby('umdNm')['parking_ratio'].transform(lambda x: x.fillna(x.mean()))
        df['parking_ratio'] = df['parking_ratio'].fillna(0.5)
        print(" > parking_ratio: 법정동별 평균 대치 및 기본값(0.5) 보정 완료.")

    # [엘리베이터] 건축년도(buildYear) 기준 2015년 이후 의무 설치 비중 고려
    if 'has_elevator' in df.columns:
        # has_elevator를 수치형으로 변환 시도 후 결측치 처리
        df['has_elevator'] = pd.to_numeric(df['has_elevator'], errors='coerce')
        # 2015년 이상이면 1, 미만이면 0 (건축물관리법 및 도메인 관례 반영)
        df.loc[df['has_elevator'].isnull(), 'has_elevator'] = \
            np.where(df.loc[df['has_elevator'].isnull(), 'buildYear'] >= 2015, 1, 0)
        print(" > has_elevator: 건축년도(2015) 기준 논리적 대치 완료.")

    # [총 층수] 누락된 경우 해당 매물의 해당 층(floor)으로 대체하여 정합성 유지
    if 'total_floors' in df.columns:
        df['total_floors'] = df['total_floors'].fillna(df['floor'])
        # 만약 floor도 결측인 경우 1층으로 기본 설정
        df['total_floors'] = df['total_floors'].fillna(1)
        print(" > total_floors: 매물 층수(floor) 기반 정합성 보정 완료.")

    # 3. 범주형 데이터 변환 (Encoding)
    print("\n--- [3] 범주형 피처 인코딩 (One-Hot & Target Encoding) ---")
    
    # [houseType] One-Hot Encoding 적용 (다세대/연립 등 유형이 적음)
    if 'houseType' in df.columns:
        df = pd.get_dummies(df, columns=['houseType'], drop_first=True)
        print(" > houseType: One-Hot Encoding 적용 성공.")

    # [umdNm] Target Encoding 적용 (차원의 저주 방지)
    # ※ 주의: 실제 운영 단계에서는 Test set의 타겟값이 노출되지 않도록 Train set 기준으로만 
    #   평균을 구해 매핑하는 것이 Data Leakage(데이터 누수)를 방지하는 정석입니다.
    if 'umdNm' in df.columns and 'Price_per_m2_log' in df.columns:
        umd_target_mean = df.groupby('umdNm')['Price_per_m2_log'].mean()
        df['umdNm_encoded'] = df['umdNm'].map(umd_target_mean)
        print(" > umdNm: Target Encoding (법정동별 평균 가격) 매핑 완료.")

    # 4. 결측치 완전 제거 확인 및 최종 저장
    print("\n--- [4] 최종 데이터셋 정합성 검증 및 저장 ---")
    
    # 분석에 필수적인 수치형/피처 컬럼들 위주로 결측치 확인
    null_summary = df.isnull().sum()
    # 주요 분석 컬럼들만 요약 출력 (전체 확인)
    important_cols = [
        'parking_ratio', 'has_elevator', 'total_floors', 
        'umdNm_encoded', 'Price_per_m2_log', 'adjusted_price_per_m2'
    ]
    # 실제 존재하는 컬럼들만 체크
    check_cols = [c for c in important_cols if c in df.columns]
    null_check = df[check_cols].isnull().sum()
    
    if null_check.sum() == 0:
        print(" > [검증 결과] 모든 주요 입전 피처의 결측치가 완전히 제거되었습니다.")
    else:
        print(f" > [주의] 아직 결측치가 남은 주요 컬럼이 있습니다:\n{null_check[null_check > 0]}")

    # 최종 컬럼 리스트 및 상위 데이터 출력
    print(f"\n[최종 데이터 사양] 행: {len(df):,}, 컬럼: {len(df.columns)}")
    print(f"[최종 컬럼 목록]: {df.columns.tolist()[:10]} ... 외 {len(df.columns)-10}개")

    # CSV 저장
    df.to_csv(OUTPUT_PATH, index=False, encoding=ENCODING)
    print(f"\n--- [종료] 피처 엔지니어링 완료! 저장 경로: {OUTPUT_PATH} ---")

    return df

if __name__ == "__main__":
    try:
        final_df = engineer_vsa_features(INPUT_PATH)
    except Exception as e:
        print(f"3단계 처리 중 오류 발생: {e}")
