import pandas as pd
import requests
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 제공해주신 3개의 카카오 API 키 리스트
KAKAO_API_KEYS = [
    "545d14c4a406db675503b6d170297d2c",
    "7d0c035ea00a86e3b630572e01efd616",
    "f3f3717a5289107d6bbd7c21aeea68a8"
]

DATA_PATH = 'c:/VSA-AVM-Engine/data/processed/nationwide_RHTrade_processed.csv'
MAPPING_PATH = 'c:/VSA-AVM-Engine/data/processed/kakao_mapping.csv'
FINAL_PATH = 'c:/VSA-AVM-Engine/data/processed/nationwide_RHTrade_enriched.csv'

class KakaoKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_idx = 0
        self.lock = threading.Lock()

    def get_headers(self):
        with self.lock:
            return {"Authorization": f"KakaoAK {self.keys[self.current_idx]}"}

    def rotate_key(self, failed_key):
        with self.lock:
            # 실패한 키가 현재 활성화된 키일 경우에만 다음 키로 교환 (여러 스레드가 동시에 바꾸는 것 방지)
            if self.keys[self.current_idx] == failed_key:
                if self.current_idx + 1 < len(self.keys):
                    self.current_idx += 1
                    print(f"\n[알림] API 키 한도 도달. 다음 키로 교체합니다. (Key Index: {self.current_idx + 1}/{len(self.keys)})")
                    return True
                else:
                    print("\n[알림] 모든 API 키가 한도에 도달했습니다.")
                    return False
            return True # 이미 다른 스레드에 의해 키가 교체된 경우

def get_kakao_info(sido, region, umdNm, jibun, key_manager):
    address = f"{sido} {region} {umdNm} {jibun}".strip()
    result = {
        'sido': sido, 'region': region, 'umdNm': umdNm, 'jibun': jibun,
        'lat': None, 'lng': None,
        'subway_name': None, 'subway_dist': None,
        'error': None
    }
    
    max_retries = 3
    
    # 1. 주소 -> 위경도 변환
    for attempt in range(max_retries):
        headers = key_manager.get_headers()
        current_key = headers["Authorization"].split()[-1]
        addr_url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
        
        try:
            res = requests.get(addr_url, headers=headers, timeout=5)
            if res.status_code == 429: # Too Many Requests (한도 초과)
                if key_manager.rotate_key(current_key):
                    continue # 키 교체 후 재시도
                else:
                    result['error'] = 'API_LIMIT_REACHED'
                    return result
            elif res.status_code != 200:
                result['error'] = f'ADDR_API_ERROR_{res.status_code}'
                return result
                
            data = res.json()
            if not data.get('documents'):
                result['error'] = 'ADDRESS_NOT_FOUND'
                return result
                
            doc = data['documents'][0]
            lng, lat = doc['x'], doc['y']
            result['lat'], result['lng'] = lat, lng
            break # 성공 시 루프 탈출
        except Exception as e:
            result['error'] = f"EXCEPTION_ADDR: {str(e)}"
            return result
            
    if result['lat'] is None:
        return result
        
    # 2. 지하철역 검색 (SW8) - 반경 1km 이내
    lat, lng = result['lat'], result['lng']
    for attempt in range(max_retries):
        headers = key_manager.get_headers()
        current_key = headers["Authorization"].split()[-1]
        subway_url = f"https://dapi.kakao.com/v2/local/search/category.json?category_group_code=SW8&y={lat}&x={lng}&radius=1000&sort=distance"
        
        try:
            sub_res = requests.get(subway_url, headers=headers, timeout=5)
            if sub_res.status_code == 429:
                if key_manager.rotate_key(current_key):
                    continue # 키 교체 후 재시도
                else:
                    result['error'] = 'API_LIMIT_REACHED'
                    return result
            elif sub_res.status_code == 200:
                sub_data = sub_res.json()
                if sub_data.get('documents'):
                    nearest_subway = sub_data['documents'][0]
                    result['subway_name'] = nearest_subway['place_name']
                    result['subway_dist'] = nearest_subway['distance']
                break
        except Exception as e:
            # 지하철 검색 오류는 주소 변환 성공 시 넘어가도록 예외처리만 수행
            break
            
    return result

def main():
    print("1. 실거래가 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)
    
    print("2. 고유 주소 추출 중...")
    unique_addrs = df.dropna(subset=['sido', 'region', 'umdNm', 'jibun'])[['sido', 'region', 'umdNm', 'jibun']].drop_duplicates()
    total_unique = len(unique_addrs)
    print(f"총 고유 주소 수: {total_unique}건")
    
    # 기존 매핑 데이터 확인
    if os.path.exists(MAPPING_PATH):
        mapped_df = pd.read_csv(MAPPING_PATH)
        merged = pd.merge(unique_addrs, mapped_df, on=['sido', 'region', 'umdNm', 'jibun'], how='left', indicator=True)
        to_process = merged[merged['_merge'] == 'left_only'][['sido', 'region', 'umdNm', 'jibun']]
    else:
        to_process = unique_addrs
        pd.DataFrame(columns=[
            'sido', 'region', 'umdNm', 'jibun', 'lat', 'lng', 
            'subway_name', 'subway_dist', 'error'
        ]).to_csv(MAPPING_PATH, index=False, encoding='utf-8-sig')
        
    remain_count = len(to_process)
    print(f"처리할 주소 수 (이미 완료된 건 제외): {remain_count}건")
    
    if remain_count == 0:
        print("모든 주소 매핑이 완료되었습니다! 원본 데이터와 병합을 시작합니다.")
        merge_data(df)
        return
        
    # KeyManager 초기화
    key_manager = KakaoKeyManager(KAKAO_API_KEYS)
    
    print("3. 카카오 API로 위경도/지하철 정보 수집 시작...")
    tasks = to_process.to_dict('records')
    completed = 0
    
    write_lock = threading.Lock()
    
    with open(MAPPING_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {
                executor.submit(get_kakao_info, task['sido'], task['region'], task['umdNm'], task['jibun'], key_manager): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                res = future.result()
                
                if res['error'] == 'API_LIMIT_REACHED':
                    # 모든 키가 한도를 넘었다면 루프 중단
                    print("\n[알림] 제공된 모든 API 키의 일일 용량이 초과되었습니다.")
                    break
                    
                with write_lock:
                    pd.DataFrame([res]).to_csv(f, header=False, index=False, encoding='utf-8-sig')
                    f.flush()
                    
                completed += 1
                if completed % 100 == 0:
                    print(f"[진행률: {completed}/{remain_count}] 데이터 수집 중...")
                    time.sleep(0.3)
                    
    print("\n--- 데이터 수집 단계 완료 ---")
    
    # 4. 원본과 병합
    merge_data(df)

def merge_data(df):
    if not os.path.exists(MAPPING_PATH):
        print("매핑 파일이 없습니다.")
        return
        
    print("4. 원본 데이터와 수집된 정보 병합 중...")
    mapping_df = pd.read_csv(MAPPING_PATH)
    mapping_df = mapping_df.drop_duplicates(subset=['sido', 'region', 'umdNm', 'jibun'], keep='last')
    
    mapping_df.loc[mapping_df['error'].notnull(), ['lat', 'lng', 'subway_name', 'subway_dist']] = None
    mapping_df = mapping_df.drop(columns=['error'])
    
    final_df = pd.merge(df, mapping_df, on=['sido', 'region', 'umdNm', 'jibun'], how='left')
    
    print(f"5. 최종 파일 저장 중... ({FINAL_PATH})")
    final_df.to_csv(FINAL_PATH, index=False, encoding='utf-8-sig')
    print("완료!")

if __name__ == '__main__':
    main()
