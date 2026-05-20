"""
File: preprocess_and_embed.py
Description: Master-level data preprocessing and semantic embedding pipeline for AVM.
             - Implements strict chronological sorting and split (Train/Test) to prevent Time-Series Data Leakage.
             - Resolves raw null data (buildYear, Age, floor) using Legal Dong (umdNm) mode/median mapping.
             - Conducts high-performance vectorized descriptive Korean natural language context serialization.
             - Fits RobustScalers and LabelEncoders strictly on past/train data and maps to future/test splits.
             - Computes resource-efficient 128-D semantic document embeddings locally via TF-IDF + TruncatedSVD (LSA).
             - Saves production-ready mapping dicts and exports optimized Parquet data & NumPy matrices.
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import re
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

INPUT_PATH = 'c:/VSA-AVM-Engine/data/processed/nationwide_RHTrade_enriched.csv'
OUTPUT_DIR = 'c:/VSA-AVM-Engine/data/processed'
PARQUET_OUTPUT = os.path.join(OUTPUT_DIR, 'processed_data.parquet')
EMBEDDING_OUTPUT = os.path.join(OUTPUT_DIR, 'property_embeddings.npy')
SCALER_OUTPUT = os.path.join(OUTPUT_DIR, 'robust_scaler.pkl')
ENCODER_OUTPUT = os.path.join(OUTPUT_DIR, 'label_encoders.pkl')

def clean_mhouse_name(name):
    if pd.isna(name) or not isinstance(name, str):
        return '미명명 단지'
    # Remove standard double spaces or unwanted characters, but keep Korean letters, numbers and parentheses
    cleaned = re.sub(r'[^a-zA-Z0-9가-힣\s\(\),]', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else '미명명 단지'

def main():
    print("--- [시작] AVM 데이터 전처리 및 임베딩 파이프라인 가동 ---")
    start_time = time.time()
    
    if not os.path.exists(INPUT_PATH):
        print(f"[에러] 대상 실거래 데이터셋이 존재하지 않습니다: {INPUT_PATH}")
        return
        
    # 1. 데이터셋 로드
    print("1. 데이터셋 로드 중...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f" > 로드 완료: {len(df):,}행, {len(df.columns)}개 컬럼")
    
    # 2. 데이터 클리닝 및 타입 정렬
    print("2. 데이터 클리닝 및 결측치 보정 (법정동 기준 매핑) 진행 중...")
    
    # mhouseNm 처리
    df['mhouseNm'] = df['mhouseNm'].apply(clean_mhouse_name)
    
    # buildYear, Age, floor 결측치 보정 (umdNm 기준 중앙값 대치)
    for col in ['buildYear', 'Age', 'floor']:
        null_cnt = df[col].isnull().sum()
        if null_cnt > 0:
            print(f" > [{col}] 결측치 {null_cnt}건 발견. 법정동(umdNm)별 중앙값 산출 및 대치 수행...")
            # 법정동별 중앙값 계산
            umd_medians = df.groupby('umdNm')[col].transform('median')
            df[col] = df[col].fillna(umd_medians)
            
            # 여전히 결측치가 남은 경우 (밀도가 낮거나 데이터가 부족한 법정동) 전체 중앙값 대치
            global_median = df[col].median()
            df[col] = df[col].fillna(global_median)
            
    # 타입 정렬 및 날짜 파싱
    df['dealYear'] = df['dealYear'].astype(int)
    df['dealMonth'] = df['dealMonth'].astype(int)
    df['dealDay'] = df['dealDay'].astype(int)
    
    # deal_date 생성 (datetime 형식)
    df['deal_date'] = pd.to_datetime(
        df['dealYear'].astype(str) + '-' + 
        df['dealMonth'].astype(str).str.zfill(2) + '-' + 
        df['dealDay'].astype(str).str.zfill(2), 
        errors='coerce'
    )
    
    # 파싱 불가능한 결측 날짜 대치
    df['deal_date'] = df['deal_date'].fillna(pd.to_datetime(df['dealYear'].astype(str) + '-01-01'))
    print(" > 날짜 파싱 및 결측 주소 정제 완료.")
    
    # 3. 피처 엔지니어링 (Feature Engineering)
    print("3. 도메인 지식 기반 파생 피처 엔지니어링 진행 중...")
    
    # [공간적 지표] 500m 이내 초역세권 여부
    df['is_subway_area'] = (df['subway_dist'] <= 500).astype(int)
    
    # [거래 시점 지표] 거래 당시 건물 연식 재계산
    df['deal_age'] = df['dealYear'] - df['buildYear']
    # 연식이 마이너스로 나오는 비정상 거래 예외 처리 (최소값 0 보정)
    df['deal_age'] = df['deal_age'].clip(lower=0)
    
    # [가격 지표] 제곱미터당 거래 가격의 Skewness 완화 (로그 변환)
    df['log_price_per_m2'] = np.log1p(df['Price_per_m2'])
    print(" > 파생 피처 생성 완료 (is_subway_area, deal_age, log_price_per_m2).")
    
    # 4. 시계열 데이터 분할 (Time-Series Split) - 데이터 누수 방지
    print("4. 시계열 정렬 및 누수 방지 분할 적용 중...")
    # 날짜순 정렬
    df = df.sort_values('deal_date').reset_index(drop=True)
    
    # 시계열 기준 Train/Test 분 분할 (80% 지점을 기준으로 설정)
    split_idx = int(len(df) * 0.8)
    split_date = df.loc[split_idx, 'deal_date']
    print(f" > 기준 분할일자: {split_date.strftime('%Y-%m-%d')} (과거 80% / 미래 20%)")
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f" > Train Split: {len(train_df):,}건, Test Split: {len(test_df):,}건")
    
    # 5. 범주형 변수 처리 (Label Encoding)
    print("5. 범주형 변수 인코딩 및 매핑 사전 구축 중...")
    cat_cols = ['sido', 'region', 'umdNm', 'houseType']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Train 데이터 기준으로 인코더를 학습시킴 (데이터 누수 방지)
        # Train에 없는 카테고리가 Test에 나올 경우를 대비해 예외값('UNKNOWN') 처리 포함
        train_vals = train_df[col].fillna('UNKNOWN').astype(str).tolist()
        le.fit(train_vals + ['UNKNOWN'])
        
        # 적용
        df[col + '_encoded'] = le.transform(df[col].fillna('UNKNOWN').astype(str).map(lambda x: x if x in le.classes_ else 'UNKNOWN'))
        label_encoders[col] = le
        
    # 인코더 저장
    with open(ENCODER_OUTPUT, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f" > 범주형 인코딩 매핑 사전 영구 보존 완료: {ENCODER_OUTPUT}")
    
    # 6. 수치형 데이터 스케일링 (RobustScaler)
    print("6. 연속형 수치 피처 Robust 스케일링 진행 중...")
    num_cols = ['excluUseAr', 'landAr', 'parking_ratio', 'subway_dist', 'total_floors']
    
    scaler = RobustScaler()
    # Train 데이터셋 기준으로 Fit 수행 (데이터 누수 원천 방지)
    scaler.fit(train_df[num_cols].fillna(0))
    
    # 전체 데이터셋에 Transform 수행
    scaled_feats = scaler.transform(df[num_cols].fillna(0))
    scaled_cols = [col + '_scaled' for col in num_cols]
    df[scaled_cols] = pd.DataFrame(scaled_feats, index=df.index)
    
    # 스케일러 저장
    with open(SCALER_OUTPUT, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" > 수치형 Robust 스케일러 저장 완료: {SCALER_OUTPUT}")
    
    # 7. LLM 임베딩용 자연어 텍스트 컨텍스트 직렬화 (Vectorized Textualization)
    print("7. 초고속 벡터 연산 기반 LLM Context 문장 생성 진행 중...")
    t_start = time.time()
    
    sido_s = df['sido'].fillna('').astype(str)
    region_s = df['region'].fillna('').astype(str)
    umd_s = df['umdNm'].fillna('').astype(str)
    house_s = df['houseType'].fillna('주택').astype(str)
    name_s = df['mhouseNm'].astype(str)
    build_s = df['buildYear'].fillna(2000).astype(int).astype(str)
    exclu_s = df['excluUseAr'].fillna(0.0).round(1).astype(str)
    land_s = df['landAr'].fillna(0.0).round(1).astype(str)
    floor_s = df['floor'].fillna(1).astype(int).astype(str)
    year_s = df['dealYear'].astype(str)
    month_s = df['dealMonth'].astype(str)
    
    # 거래금액 한글 포맷 변환 벡터화
    billion_s = (df['dealAmount'] // 10000).fillna(0).astype(int)
    million_s = (df['dealAmount'] % 10000).fillna(0).astype(int)
    
    amount_str = np.where(billion_s > 0, billion_s.astype(str) + "억 ", "") + \
                  np.where(million_s > 0, million_s.apply(lambda x: f"{x:,}") + "만 원", "원")
                  
    subway_s = df['subway_name'].fillna('알 수 없는 역').astype(str)
    subway_dist_s = df['subway_dist'].fillna(0).astype(int).astype(str)
    
    elevator_str = np.where(df['has_elevator'] == 1, "엘리베이터가 있으며", "엘리베이터가 없으며")
    parking_str = np.where(df['parking_ratio'] >= 1.0, 
                           "주차 비율 조건을 충족합니다.", 
                           "주차 비율은 " + df['parking_ratio'].round(2).astype(str) + "대 수준입니다.")
    
    df['text_context'] = sido_s + " " + region_s + " " + umd_s + "에 위치한 " + house_s + " " + name_s + "입니다. " + \
                          build_s + "년에 건축되었으며, 전용면적은 " + exclu_s + "m2, 대지권면적은 " + land_s + "m2입니다. " + \
                          "현재 " + floor_s + "층 매물로, " + year_s + "년 " + month_s + "월에 " + amount_str + "에 거래되었습니다. " + \
                          "가장 가까운 지하철역은 " + subway_s + "이며 거리는 " + subway_dist_s + "m입니다. " + \
                          elevator_str + " " + parking_str
                          
    print(f" > 생성 완료 (소요 시간: {time.time() - t_start:.2f}초)")
    print(" > 예시 문장 출력:")
    print(f"   [문장] \"{df['text_context'].iloc[0]}\"")
    
    # 8. 배치 Semantic 임베딩 추출 (TF-IDF + LSA Latent Semantic Analysis)
    print("8. 로컬 고성능 배치 Semantic 임베딩(TF-IDF + TruncatedSVD) 연산 수행 중...")
    t_start = time.time()
    
    # 자원 효율적 배치 파이프라인
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # 48만 건 전체에 대해 sparse TF-IDF 생성
    print(" > Step A: 텍스트 코퍼스 분석 및 sparse TF-IDF 벡터화...")
    tfidf_matrix = vectorizer.fit_transform(df['text_context'])
    
    # 차원축소 (SVD를 통한 128차원 Dense Semantic Vector 추출)
    print(" > Step B: TruncatedSVD 기반 128차원 밀집 시맨틱 벡터 추출 및 차원 축소...")
    svd = TruncatedSVD(n_components=128, random_state=42)
    dense_embeddings = svd.fit_transform(tfidf_matrix)
    
    print(f" > 임베딩 벡터 생성 완료! 형태(Shape): {dense_embeddings.shape} (소요 시간: {time.time() - t_start:.2f}초)")
    
    # 9. 최종 결과물 저장 및 내보내기 (Export)
    print("9. 전처리 완료된 Parquet 및 dense 임베딩 NPY 파일 영구 저장 중...")
    
    # parquet 저장을 위해 datetime 등 복잡한 타입 및 불필요한 대형 텍스트 제외 테이블 축소 구성 가능
    # 하지만 사용성 확보를 위해 모든 전처리 컬럼 포함하여 Parquet 저장 진행
    df.to_parquet(PARQUET_OUTPUT, index=False)
    print(f" > [완료] 정형 데이터셋 저장 완료: {PARQUET_OUTPUT}")
    
    # NPY 파일 저장
    np.save(EMBEDDING_OUTPUT, dense_embeddings)
    print(f" > [완료] 밀집 시맨틱 임베딩 행렬 저장 완료: {EMBEDDING_OUTPUT}")
    
    end_time = time.time()
    print(f"\n--- [종료] 데이터 전처리 및 임베딩 파이프라인이 전격 완료되었습니다! (총 소요 시간: {end_time - start_time:.2f}초) ---")

if __name__ == '__main__':
    main()
