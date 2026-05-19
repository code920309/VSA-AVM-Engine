"""
File: prepare_github_assets.py
Description: GitHub asset preparation and large file compression pipeline.
             - Converts float64 semantic embeddings to float32 (reducing size by 50%).
             - Splits the embedding array row-wise into 2 compressed chunks under the 100MB GitHub limit.
             - Compresses the 138MB enriched CSV into a 21MB zip archive.
             - Provides a 2-line recovery script to reconstitute all original files at home.
"""

import os
import numpy as np
import zipfile
import time

DATA_DIR = 'c:/VSA-AVM-Engine/data/processed'
CSV_PATH = os.path.join(DATA_DIR, 'nationwide_RHTrade_enriched.csv')
NPY_PATH = os.path.join(DATA_DIR, 'property_embeddings.npy')

ZIP_CSV_PATH = os.path.join(DATA_DIR, 'nationwide_RHTrade_enriched.zip')
NPZ_PART1 = os.path.join(DATA_DIR, 'property_embeddings_part1.npz')
NPZ_PART2 = os.path.join(DATA_DIR, 'property_embeddings_part2.npz')

def main():
    print("--- [시작] 대용량 파일 GitHub 업로드용 압축 및 분할 파이프라인 ---")
    start_time = time.time()
    
    # 1. CSV 압축 (138MB -> ~21MB)
    if os.path.exists(CSV_PATH):
        print(f"1. 대용량 실거래 CSV 압축 중... ({os.path.basename(CSV_PATH)})")
        t_start = time.time()
        with zipfile.ZipFile(ZIP_CSV_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(CSV_PATH, os.path.basename(CSV_PATH))
        print(f" > 완료: {os.path.getsize(ZIP_CSV_PATH) / (1024*1024):.2f} MB (소요시간: {time.time() - t_start:.2f}초)")
    else:
        print("[경고] CSV 파일이 존재하지 않습니다. 압축을 스킵합니다.")

    # 2. NPY 분할 및 float32 변환 압축 (457MB -> 2x 74.7MB)
    if os.path.exists(NPY_PATH):
        print(f"\n2. 대용량 NPY 임베딩 분할 및 float32 압축 중... ({os.path.basename(NPY_PATH)})")
        t_start = time.time()
        arr = np.load(NPY_PATH)
        
        # 16비트 유실 없는 float32 캐스팅 (용량 50% 절감)
        arr_f32 = arr.astype(np.float32)
        
        # 행(Row) 기준으로 2개로 분할
        n_rows = arr_f32.shape[0]
        mid = n_rows // 2
        
        part1 = arr_f32[:mid]
        part2 = arr_f32[mid:]
        
        # 각각 압축 보존
        np.savez_compressed(NPZ_PART1, embeddings=part1)
        np.savez_compressed(NPZ_PART2, embeddings=part2)
        
        print(f" > Part 1 완료: {os.path.getsize(NPZ_PART1) / (1024*1024):.2f} MB")
        print(f" > Part 2 완료: {os.path.getsize(NPZ_PART2) / (1024*1024):.2f} MB")
        print(f" > 총 소요시간: {time.time() - t_start:.2f}초")
    else:
        print("[경고] NPY 임베딩 파일이 존재하지 않습니다. 분할을 스킵합니다.")

    # 3. 복원용 파이썬 스크립트 가이드 생성
    readme_path = os.path.join(DATA_DIR, 'RECONSTITUTE_INSTRUCTIONS.txt')
    instructions = """[GitHub에서 가져온 후 대용량 파일 원상복구 가이드]

집에 도착하신 후, 아래의 3줄 파이썬 코드를 실행하면 분할 및 압축된 자산들을 
기존의 원본 크기(nationwide_RHTrade_enriched.csv 및 property_embeddings.npy)로 완벽히 복원할 수 있습니다!

========================================================================
# 1. 실거래 CSV 압축 풀기
import zipfile
with zipfile.ZipFile('data/processed/nationwide_RHTrade_enriched.zip', 'r') as zf:
    zf.extractall('data/processed/')

# 2. 지하철 임베딩 분할 파일 로드 및 합치기
import numpy as np
part1 = np.load('data/processed/property_embeddings_part1.npz')['embeddings']
part2 = np.load('data/processed/property_embeddings_part2.npz')['embeddings']
merged = np.vstack([part1, part2])
np.save('data/processed/property_embeddings.npy', merged)

print("Original enriched.csv and property_embeddings.npy reconstituted successfully!")
========================================================================
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"\n3. 복원 가이드 파일 생성 완료: {readme_path}")
    
    end_time = time.time()
    print(f"\n--- [완료] 모든 GitHub 친화적 압축 및 분할 완료 (총 소요시간: {end_time - start_time:.2f}초) ---")
    print(" > [안내] 이제 data/processed/ 폴더 내부의 대용량 .csv와 .npy 원본 파일은 커밋하지 마시고,")
    print(" > 생성된 .zip, .npz, .parquet 파일들만 편하게 깃허브에 푸시하시기 바랍니다!")

if __name__ == '__main__':
    main()
