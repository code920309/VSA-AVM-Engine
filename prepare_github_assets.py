import os
import numpy as np
import zipfile
import time
import math

# 경로 설정
DATA_DIR = 'data/processed'
LIMIT_MB = 90  # 안전하게 90MB를 한도로 설정

def compress_csv(file_path):
    zip_path = file_path.replace('.csv', '.zip')
    print(f" > CSV 압축 중: {os.path.basename(file_path)} -> {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, os.path.basename(file_path))
    print(f"   완료: {os.path.getsize(zip_path)/(1024*1024):.1f} MB")

def split_npy(file_path):
    print(f" > NPY 분할 중: {os.path.basename(file_path)}")
    arr = np.load(file_path)
    # float32 변환 (정밀도 유지 및 용량 50% 절감)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    
    # 예상 파일당 용량 계산 및 분할 수 결정
    # float32는 4바이트/원소. 128차원 검색 행렬 기준
    total_size_mb = arr.nbytes / (1024*1024)
    num_parts = math.ceil(total_size_mb / LIMIT_MB)
    rows_per_part = math.ceil(len(arr) / num_parts)
    
    base_name = os.path.basename(file_path).replace('.npy', '')
    for i in range(num_parts):
        start = i * rows_per_part
        end = min((i + 1) * rows_per_part, len(arr))
        part_arr = arr[start:end]
        part_name = f"{base_name}_part{i+1}.npz"
        part_path = os.path.join(DATA_DIR, part_name)
        np.savez_compressed(part_path, embeddings=part_arr)
        print(f"   Part {i+1} 저장: {part_name} ({os.path.getsize(part_path)/(1024*1024):.1f} MB)")

def main():
    print("--- [시작] GitHub 대규모 자산 분할 및 압축 (v2.0) ---")
    if not os.path.exists(DATA_DIR):
        print(f"경로 없음: {DATA_DIR}")
        return

    files = os.listdir(DATA_DIR)
    large_files = []
    
    for f in files:
        f_path = os.path.join(DATA_DIR, f)
        if os.path.isfile(f_path) and os.path.getsize(f_path) > 100 * 1024 * 1024:
            large_files.append(f_path)
            
    if not large_files:
        print("100MB 초과 파일을 찾을 수 없습니다.")
        return

    print(f"발견된 대용량 파일: {len(large_files)}개")
    
    for f_path in large_files:
        ext = os.path.splitext(f_path)[1].lower()
        if ext == '.csv':
            compress_csv(f_path)
        elif ext == '.npy':
            split_npy(f_path)
        else:
            print(f"스킵 (지원하지 않는 확장자): {f_path}")

    print("\n--- [완료] 모든 자산의 GitHub 규격화 성공 ---")
    print(" > 안내: 이제 생성된 .zip 및 .npz 파일들을 Git 스테이징에 추가(git add)하세요.")
    print(" > 주의: 100MB 초과 원본 .csv와 .npy 파일은 .gitignore에 포함하거나 수동으로 제외해야 합니다.")

if __name__ == '__main__':
    main()
