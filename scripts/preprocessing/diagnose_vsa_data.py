import pandas as pd
import numpy as np
import os

# G2B 및 프롭테크 데이터 환경 설정
INPUT_PATH = 'data/processed/nationwide_RHTrade_enriched.csv'
# 이전 작업에서 확인된 BOM(Byte Order Mark) 대응을 위해 utf-8-sig 사용
ENCODING = 'utf-8-sig'

def diagnose_vsa_dataset(file_path):
    """
    VSA-AVM 데이터셋의 스키마를 정렬하고 결측치 및 아웃라이어를 정밀 진단합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. 경로: {file_path}")
        return

    print("--- [1] 데이터 로드 및 타입 강제 정의 시작 ---")
    
    # 데이터 로드
    df = pd.read_csv(file_path, encoding=ENCODING, low_memory=False)

    # 1. 수치형 변수 정의 및 강제 형변환
    num_cols = [
        'excluUseAr', 'landAr', 'floor', 'dealAmount', 'Age', 
        'land_index', 'current_land_index', 'total_floors', 
        'lat', 'lng', 'subway_dist'
    ]
    
    # 데이터셋에 존재하지 않는 컬럼 예외 처리 및 타입 변환
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # 예: Age가 없고 buildYear만 있는 경우 연산 시도
            if col == 'Age' and 'buildYear' in df.columns and 'dealYear' in df.columns:
                df['Age'] = df['dealYear'] - df['buildYear']
            else:
                print(f"Warning: 수치형 컬럼 {col}이 데이터셋에 존재하지 않습니다.")

    # 2. 범주형/문자열 변수 정의
    cat_cols = [
        'sggCd', 'umdNm', 'jibun', 'mhouseNm', 'houseType', 
        'subway_name', 'building_structure', 'has_elevator', 'is_commercial_villa'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)
        else:
            print(f"Warning: 범주형 컬럼 {col}이 데이터셋에 존재하지 않습니다.")

    # 3. 날짜/시간 변수 처리
    date_cols = ['dealYear', 'dealMonth', 'dealDay', 'approval_date']
    for col in date_cols:
        if col in df.columns:
            if col == 'approval_date':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    print("--- [2] 결측치 및 논리적 노이즈 스크리닝 진행 ---")

    # 결측치 요약 리포트 생성
    missing_report = []
    
    for col in df.columns:
        # 기본 Null/NaN 확인
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        # 문자열 공백 또는 빈 값 스크리닝
        empty_str_count = 0
        if df[col].dtype == 'object':
            empty_str_count = df[col].apply(lambda x: str(x).strip() == '' if pd.notnull(x) else False).sum()
            
        # 수치형 논리 오류 (면적, 금액 등이 0인 경우)
        logical_zero_count = 0
        if col in ['excluUseAr', 'landAr', 'dealAmount', 'total_floors']:
            logical_zero_count = (df[col] == 0).sum()

        missing_report.append({
            'Column': col,
            'Missing(Null)': null_count,
            'Missing(%)': round(null_pct, 2),
            'EmptyStr': empty_str_count,
            'LogicalZeroError': logical_zero_count
        })

    missing_df = pd.DataFrame(missing_report)

    print("--- [3] 타겟 변수(dealAmount) 아웃라이어 진단 ---")
    
    if 'dealAmount' in df.columns:
        target = df['dealAmount'].dropna()
        stats = {
            'Mean': target.mean(),
            'Median': target.median(),
            'Min': target.min(),
            'Max': target.max(),
            'Lower_2pct': np.percentile(target, 2),
            'Upper_2pct': np.percentile(target, 98)
        }
        
        # 가독성을 위한 통계값 출력
        print("\n[Target: dealAmount 기술 통계 및 경계값]")
        for k, v in stats.items():
            print(f"- {k:12}: {v:15,.0f}")
    
    print("\n--- [최종 출력] 데이터 스키마 및 결측치 리포트 ---")
    print("\n[1. Data Schema Summary]")
    print(df.info())
    
    print("\n[2. Missing Value & Noise Report]")
    # 결측치가 있거나 노이즈가 있는 컬럼만 필터링하여 출력 가능
    print(missing_df.sort_values(by='Missing(%)', ascending=False).to_string(index=False))

    return df, missing_df

if __name__ == "__main__":
    # 작업 디렉토리 기준 경로 확인 후 실행
    try:
        final_df, report = diagnose_vsa_dataset(INPUT_PATH)
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
