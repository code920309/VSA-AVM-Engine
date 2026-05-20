import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 경로 설정
INPUT_PATH = 'data/processed/nationwide_RHTrade_final_features.csv'
OUTPUT_EMBEDDING = 'data/processed/final_similarity_matrix.npy'
ENCODING = 'utf-8-sig'

def build_similarity_index():
    print("--- [시작] 유사도 검색을 위한 최종 임베딩 인덱스 구축 ---")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: 파일을 찾을 수 없습니다. {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding=ENCODING, low_memory=False)
    print(f" > 데이터 로드 완료: {len(df):,} 행")

    # 1. 시각적/물리적 특징을 포함한 텍스트 컨텍스트 생성 (유사도 판단의 근거)
    print(" > 텍스트 컨텍스트 생성 중...")
    
    # 누락된 값 기본 처리
    umd_s = df['umdNm'].fillna('').astype(str)
    name_s = df['mhouseNm'].fillna('미명명 단지').astype(str)
    exclu_s = df['excluUseAr'].round(1).astype(str)
    year_s = df['buildYear'].fillna(2000).astype(int).astype(str)
    
    # 유사도 판단에 핵심적인 자연어 문장 생성
    df['search_context'] = (
        umd_s + " " + name_s + " " + 
        exclu_s + "m2 " + year_s + "년 건축 " + 
        df['total_floors'].astype(str) + "층"
    )

    # 2. TF-IDF + SVD (128차원) 벡터화
    print(" > 128차원 시맨틱 벡터 연산 중 (LSA)...")
    t_start = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['search_context'])
    
    svd = TruncatedSVD(n_components=128, random_state=42)
    final_embeddings = svd.fit_transform(tfidf_matrix)
    
    # 3. 인덱스 매칭을 위해 저장
    np.save(OUTPUT_EMBEDDING, final_embeddings)
    print(f" > [완료] 심층 유사도 인덱스 구축 완료: {OUTPUT_EMBEDDING}")
    print(f" > 총 소요 시간: {time.time() - t_start:.2f}초")

if __name__ == "__main__":
    build_similarity_index()
