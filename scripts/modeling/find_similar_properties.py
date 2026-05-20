import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# 경로 설정
DATA_PATH = 'data/processed/nationwide_RHTrade_final_features.csv'
MATRIX_PATH = 'data/processed/final_similarity_matrix.npy'
ENCODING = 'utf-8-sig'

def find_top_comparables(target_index, top_n=3):
    """
    특정 매물(인덱스 기준)과 가장 유사한 상위 N개의 거래 사례를 찾습니다.
    """
    print(f"--- [조회] {target_index}번 매물 기준 유사 사례 분석 시작 ---")
    
    if not os.path.exists(DATA_PATH) or not os.path.exists(MATRIX_PATH):
        print("Error: 필수 데이터(CSV/NPY)가 존재하지 않습니다.")
        return

    # 데이터 및 매트릭스 로드
    df = pd.read_csv(DATA_PATH, encoding=ENCODING, low_memory=False)
    embeddings = np.load(MATRIX_PATH)

    if target_index >= len(df):
        print(f"Error: 인덱스 범위 초과 (최대: {len(df)-1})")
        return

    # 1. 타겟 벡터와 전체 벡터 공간 간의 코사인 유사도 계산
    target_vector = embeddings[target_index].reshape(1, -1)
    similarities = cosine_similarity(target_vector, embeddings).flatten()

    # 2. 자기 자신을 제외하고 유사도 높은 순으로 인덱스 추출
    # argsort는 오름차순이므로 뒤에서부터 슬라이싱
    similar_indices = similarities.argsort()[-(top_n+1):-1][::-1]

    # 3. 결과 출력 및 비교 리포트 생성
    target_info = df.iloc[target_index]
    print("\n[대상 매물 정보]")
    print(f"- 위치: {target_info['sido']} {target_info['region']} {target_info['umdNm']}")
    print(f"- 건물: {target_info['mhouseNm']} ({target_info['buildYear']}년 건축)")
    print(f"- 면적: {target_info['excluUseAr']}m2 / 거래가: {target_info['dealAmount']:,}만원")
    
    print("\n" + "="*50)
    print(f"   AI가 선정한 유사 실거래 사례 Top {top_n}")
    print("="*50)

    for i, idx in enumerate(similar_indices, 1):
        comp = df.iloc[idx]
        sim_score = similarities[idx]
        print(f"사례 {i}. (유사도: {sim_score:.4f})")
        print(f"  - 주소: {comp['sido']} {comp['region']} {comp['umdNm']} {comp['jibun']}")
        print(f"  - 건물: {comp['mhouseNm']} ({comp['buildYear']}년 건축)")
        print(f"  - 조건: {comp['excluUseAr']}m2, {comp['floor']}층")
        print(f"  - 실거래가: {comp['dealAmount']:,}만원 ({comp['dealYear']}-{comp['dealMonth']})")
        print("-" * 30)

if __name__ == "__main__":
    # 샘플로 100번 매물에 대한 유사 사례를 조회해봅니다.
    # 궁금하신 특정 인덱스가 있다면 값을 변경하여 실행 가능합니다.
    find_top_comparables(target_index=100)
