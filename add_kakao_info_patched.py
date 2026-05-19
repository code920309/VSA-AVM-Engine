"""
File: add_kakao_info_patched.py
Description: Patched version of add_kakao_info.py that fixes Windows socket port exhaustion (WinError 10048).
             - Cleans kakao_mapping.csv of previous EXCEPTION_ADDR and transient API_ERROR_400 / ADDR_API_ERROR_400 / API_FORBIDDEN_401 limit errors.
             - Implements requests.Session with connection pooling (HTTPAdapter) to reuse sockets.
             - Excludes the disabled 2nd Kakao API key (403 Forbidden) and manages active Keys 1 and 3 (fixed typo!).
             - Safely merges results directly into nationwide_RHTrade_enriched.csv without overwriting building features.
             - Detects custom Kakao API limit responses (HTTP 400 with "limit exceeded") and rotates keys immediately.
             - Configures max_workers to 5 for extremely safe, non-invasive execution.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Exclude 2nd key (7d0c03...) due to 403 Forbidden error (App key does not exist / deleted)
# Fixed typo in 3rd key: 'f3f3717a5289107d6bbd7c21aeea68a8' (the '7' was missing in previous sessions!)
ACTIVE_KAKAO_API_KEYS = [
    "545d14c4a406db675503b6d170297d2c",
    "f3f3717a5289107d6bbd7c21aeea68a8"
]

PROCESSED_PATH = 'c:/VSA-AVM-Engine/data/processed/nationwide_RHTrade_processed.csv'
ENRICHED_PATH = 'c:/VSA-AVM-Engine/data/processed/nationwide_RHTrade_enriched.csv'
MAPPING_PATH = 'c:/VSA-AVM-Engine/data/processed/kakao_mapping.csv'

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
            if self.keys[self.current_idx] == failed_key:
                if self.current_idx + 1 < len(self.keys):
                    self.current_idx += 1
                    print(f"\n[알림] API 키 한도 도달. 다음 키로 교체합니다. (Key Index: {self.current_idx + 1}/{len(self.keys)})")
                    return True
                else:
                    return False
            return True

def clean_mapping_checkpoint():
    """
    Cleans kakao_mapping.csv of previous transient socket exhaustion errors
    and API limit errors so we can re-query them.
    """
    print("\n--- [시작] 기존 매핑 체크포인트 정화 작업 ---")
    if not os.path.exists(MAPPING_PATH):
        print("기존 매핑 파일이 없습니다. 새롭게 시작합니다.")
        return
        
    df_map = pd.read_csv(MAPPING_PATH)
    initial_len = len(df_map)
    
    # Filter out rows with socket exceptions or API limit/auth errors.
    # Permanent errors (like ADDRESS_NOT_FOUND) are kept.
    is_network_or_limit_error = df_map['error'].astype(str).str.contains(
        'EXCEPTION_ADDR|WinError|Connection|API_ERROR_400|ADDR_API_ERROR_400|API_FORBIDDEN_401', 
        case=False, 
        na=False
    )
    
    cleaned_df = df_map[~is_network_or_limit_error]
    final_len = len(cleaned_df)
    removed_count = initial_len - final_len
    
    print(f" > 기존 기록 수: {initial_len}건")
    print(f" > 소켓 및 한도 초과 정화 대상: {removed_count}건 제거")
    print(f" > 유효 기록 보존 수: {final_len}건")
    
    cleaned_df.to_csv(MAPPING_PATH, index=False, encoding='utf-8-sig')
    print("--- [완료] 기존 체크포인트 정화 완료 ---\n")

def call_kakao_api_with_retry(session, url, key_manager):
    """
    Calls Kakao API with a robust retry mechanism to handle transient 429 Rate Limits
    and rotates keys only when the daily quota is actually exhausted.
    """
    max_attempts = 5
    for attempt in range(max_attempts):
        headers = key_manager.get_headers()
        current_key = headers["Authorization"].split()[-1]
        try:
            res = session.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                return res, None
            elif res.status_code == 429:
                # Sleep 1.0 second to back off from a temporary rate limit
                time.sleep(1.0)
                # If we get 429 multiple times (>= 2 retries), assume the daily quota is exhausted
                if attempt >= 2:
                    if key_manager.rotate_key(current_key):
                        continue
                    else:
                        return None, 'API_LIMIT_REACHED'
                continue
            elif res.status_code in [401, 403]:
                # Unauthorized or Forbidden, rotate key immediately if others are available
                if key_manager.rotate_key(current_key):
                    continue
                else:
                    return None, f'API_FORBIDDEN_{res.status_code}'
            elif res.status_code == 400:
                # Kakao returns 400 Bad Request with code -10 when daily limit is reached
                try:
                    data = res.json()
                    msg_str = str(data.get("message", "")) or str(data.get("msg", ""))
                    if "limit" in msg_str.lower():
                        print(f"\n[알림] Kakao API 400 한도 초과 감지. 키를 교체합니다.")
                        if key_manager.rotate_key(current_key):
                            continue
                        else:
                            return None, 'API_LIMIT_REACHED'
                except Exception:
                    pass
                return None, f'API_ERROR_{res.status_code}'
            else:
                return None, f'API_ERROR_{res.status_code}'
        except Exception as e:
            # Network exception, wait 0.5s and retry
            time.sleep(0.5)
            if attempt == max_attempts - 1:
                return None, f"EXCEPTION_CONN: {str(e)}"
    return None, 'MAX_RETRIES_EXCEEDED'

def get_kakao_info_session(session, sido, region, umdNm, jibun, key_manager):
    address = f"{sido} {region} {umdNm} {jibun}".strip()
    result = {
        'sido': sido, 'region': region, 'umdNm': umdNm, 'jibun': jibun,
        'lat': None, 'lng': None,
        'subway_name': None, 'subway_dist': None,
        'error': None
    }
    
    # 1. Address to Lat/Lng Geocoding
    addr_url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    res, err = call_kakao_api_with_retry(session, addr_url, key_manager)
    
    if err:
        result['error'] = err
        return result
        
    try:
        data = res.json()
        if not data.get('documents'):
            result['error'] = 'ADDRESS_NOT_FOUND'
            return result
            
        doc = data['documents'][0]
        lng, lat = doc['x'], doc['y']
        result['lat'], result['lng'] = float(lat), float(lng)
    except Exception as e:
        result['error'] = f"EXCEPTION_PARSE: {str(e)}"
        return result
        
    # 2. Subway Category Search (SW8) - 1km Radius
    lat, lng = result['lat'], result['lng']
    subway_url = f"https://dapi.kakao.com/v2/local/search/category.json?category_group_code=SW8&y={lat}&x={lng}&radius=1000&sort=distance"
    
    sub_res, sub_err = call_kakao_api_with_retry(session, subway_url, key_manager)
    if sub_err:
        # If it's a critical daily API limit, propagate it
        if sub_err == 'API_LIMIT_REACHED':
            result['error'] = 'API_LIMIT_REACHED'
        # Otherwise, swallow subway errors so we don't discard successful coordinates
        return result
        
    try:
        sub_data = sub_res.json()
        if sub_data.get('documents'):
            nearest_subway = sub_data['documents'][0]
            result['subway_name'] = nearest_subway['place_name']
            result['subway_dist'] = float(nearest_subway['distance'])
    except Exception:
        pass # Swallow parse exceptions for subway info
        
    return result

def merge_data():
    if not os.path.exists(MAPPING_PATH):
        print("[경고] 매핑 파일이 없어 병합을 스킵합니다.")
        return
        
    print("\n--- [시작] 데이터 최종 병합 작업 ---")
    
    # 1. Determine base dataset (Preserve building features if enriched.csv exists)
    if os.path.exists(ENRICHED_PATH):
        print(f" > 기존 건축 정보가 포함된 {ENRICHED_PATH} 파일을 기준으로 위경도를 갱신합니다.")
        base_df = pd.read_csv(ENRICHED_PATH, low_memory=False)
        # Drop old Kakao columns if they exist to avoid duplication conflict
        for col in ['lat', 'lng', 'subway_name', 'subway_dist']:
            if col in base_df.columns:
                base_df = base_df.drop(columns=[col])
    else:
        print(f" > {ENRICHED_PATH}가 없어 {PROCESSED_PATH}를 기본 데이터셋으로 사용합니다.")
        base_df = pd.read_csv(PROCESSED_PATH, low_memory=False)
        
    # 2. Process Mapping File
    mapping_df = pd.read_csv(MAPPING_PATH, low_memory=False)
    mapping_df = mapping_df.drop_duplicates(subset=['sido', 'region', 'umdNm', 'jibun'], keep='last')
    
    # Set coordinates to None if there was an actual geocoding error
    mapping_df.loc[mapping_df['error'].notnull(), ['lat', 'lng', 'subway_name', 'subway_dist']] = None
    mapping_df = mapping_df.drop(columns=['error'], errors='ignore')
    
    # 3. Perform Left Join
    final_df = pd.merge(base_df, mapping_df, on=['sido', 'region', 'umdNm', 'jibun'], how='left')
    
    # 4. Save Final Output
    print(f" > 최종 정제된 융합 데이터 저장 중... ({ENRICHED_PATH})")
    final_df.to_csv(ENRICHED_PATH, index=False, encoding='utf-8-sig')
    print("--- [완료] 위경도 및 지하철 정보 최종 결합 완료 ---\n")

def main():
    print("1. 체크포인트 클리닝 진행...")
    clean_mapping_checkpoint()
    
    print("2. 원본 실거래 데이터 로드 중...")
    if not os.path.exists(PROCESSED_PATH):
        print(f"[에러] 기본 처리 파일이 없습니다: {PROCESSED_PATH}")
        return
    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    
    print("3. 고유 주소 추출 중...")
    unique_addrs = df.dropna(subset=['sido', 'region', 'umdNm', 'jibun'])[['sido', 'region', 'umdNm', 'jibun']].drop_duplicates()
    total_unique = len(unique_addrs)
    print(f"총 고유 주소 수: {total_unique}건")
    
    # Reload mapping checkpoint
    if os.path.exists(MAPPING_PATH):
        mapped_df = pd.read_csv(MAPPING_PATH, low_memory=False)
    else:
        mapped_df = pd.DataFrame(columns=['sido', 'region', 'umdNm', 'jibun', 'lat', 'lng', 'subway_name', 'subway_dist', 'error'])
        mapped_df.to_csv(MAPPING_PATH, index=False, encoding='utf-8-sig')
        
    merged = pd.merge(unique_addrs, mapped_df, on=['sido', 'region', 'umdNm', 'jibun'], how='left', indicator=True)
    to_process = merged[merged['_merge'] == 'left_only'][['sido', 'region', 'umdNm', 'jibun']]
    
    remain_count = len(to_process)
    print(f"추가 수집이 필요한 잔여 주소 수: {remain_count}건")
    
    if remain_count == 0:
        print("이미 모든 주소가 정상 수집/에러 처리 완료되었습니다. 최종 결합을 진행합니다.")
        merge_data()
        return
        
    # KeyManager and Session Pool Initialization
    key_manager = KakaoKeyManager(ACTIVE_KAKAO_API_KEYS)
    
    # Configure requests.Session with connection pooling (prevents WinError 10048)
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=Retry(total=3, backoff_factor=0.2))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    print("4. 커넥션 풀 기반 비집중식 멀티스레드 API 수집 시작 (max_workers=5)...")
    tasks = to_process.to_dict('records')
    completed = 0
    write_lock = threading.Lock()
    
    # Open mapping file in append mode safely
    with open(MAPPING_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(get_kakao_info_session, session, task['sido'], task['region'], task['umdNm'], task['jibun'], key_manager): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                res = future.result()
                
                if res['error'] == 'API_LIMIT_REACHED':
                    print("\n[알림] 잔여 활성 API 키의 일일 용량이 전체 한도 초과되었습니다.")
                    break
                    
                with write_lock:
                    pd.DataFrame([res]).to_csv(f, header=False, index=False, encoding='utf-8-sig')
                    f.flush()
                    
                completed += 1
                if completed % 50 == 0:
                    print(f"[수집 진행률: {completed}/{remain_count}] 데이터 적재 중... (현재 키 인덱스: {key_manager.current_idx + 1})")
                    time.sleep(0.05)
                    
    print("\n--- API 수집 세션 완료 ---")
    
    # 5. Integrate final results
    merge_data()

if __name__ == '__main__':
    main()
