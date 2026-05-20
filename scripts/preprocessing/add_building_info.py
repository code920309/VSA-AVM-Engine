"""
File: add_building_info.py
Description: Asynchronously queries the public building register API (국토교통부_건축HUB_건축물대장정보 서비스)
             for building metadata, parses and derives key features, applies checkpointing every 5,000 records,
             and performs a Left Join with the nationwide trade transactions dataset.

Step-by-Step Processing Logic:
1. Load nationwide_RHTrade_enriched.csv and extract unique sggCd, umdNm, and jibun combinations.
2. Check for an existing data/processed/building_checkpoint.csv file to load already completed records,
   preventing duplicate API queries and allowing seamless resumption of aborted/limited runs.
3. Build a dynamic legal dong lookup cache by querying the grpc-proxy-server in parallel for all
   unique 5-digit sigungu codes present in the dataset.
4. Prepare asynchronous API queries utilizing a Semaphore of 30 to manage concurrency and load safely.
5. Parse results defensively to handle missing registers or API limit messages (saving progress immediately
   and stopping if daily traffic limits are exceeded).
6. Impute missing columns with safe regional defaults (e.g. 0.8 parking ratio) and match YYYYMMDD dates
   or fall back to buildYear.
7. Merge all enriched architectural features back into nationwide_RHTrade_enriched.csv.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

API_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
SERVICE_KEY = os.getenv("MOLIT_SERVICE_KEY")

DATA_PATH = "data/processed/nationwide_RHTrade_enriched.csv"
CHECKPOINT_PATH = "data/processed/building_checkpoint.csv"

# Safe Imputation Defaults
DEFAULT_PARKING_RATIO = 0.8
DEFAULT_HAS_ELEVATOR = 0
DEFAULT_BUILDING_STRUCTURE = 1  # 1 = Reinforce Concrete (RC), 0 = Others
DEFAULT_IS_COMMERCIAL_VILLA = 0
DEFAULT_TOTAL_FLOORS = 4

def fetch_bjdong_mapping(unique_sgg_codes):
    """
    Dynamically builds a legal dong code cache mapping: (sigunguCd, umdNm) -> bjdongCd (5 digits).
    Uses the public grpc-proxy-server to fetch legal dongs for all unique sigungus in the dataset.
    """
    print("\n--- [시작] 법정동 코드 캐시 작성 중 ---")
    mapping = {}
    
    def fetch_sgg(sgg):
        url = f"https://grpc-proxy-server-mkvo6j4wsq-du.a.run.app/v1/regcodes?regcode_pattern={sgg}*"
        try:
            import requests
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                results = []
                for item in data.get('regcodes', []):
                    code = item['code']
                    name = item['name']
                    parts = name.split()
                    if len(parts) >= 2:
                        umd = parts[-1]
                        bjdong = code[5:10]
                        results.append((umd, bjdong))
                return sgg, results
        except Exception:
            pass
        return sgg, []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_sgg, sgg): sgg for sgg in unique_sgg_codes}
        for i, future in enumerate(as_completed(futures)):
            sgg, results = future.result()
            for umd, bjdong in results:
                mapping[(str(sgg), umd)] = bjdong
            if (i+1) % 30 == 0 or (i+1) == len(unique_sgg_codes):
                print(f" > 법정동 코드 수집 진행률: {i+1}/{len(unique_sgg_codes)}")
                
    print("--- [완료] 법정동 코드 캐시 작성 완료 ---\n")
    return mapping

def parse_jibun(jibun):
    """
    Splits address jibun into 4-digit zero-padded bun and ji string components.
    Example: '5-10' -> ('0005', '0010')
             '112'  -> ('0112', '0000')
    """
    if pd.isna(jibun):
        return None, None
    jibun_str = str(jibun).strip()
    if not jibun_str:
        return None, None
    if '-' in jibun_str:
        parts = jibun_str.split('-')
        bun = parts[0].strip().zfill(4)
        ji = parts[1].strip().zfill(4)
    else:
        bun = jibun_str.strip().zfill(4)
        ji = '0000'
    return bun, ji

async def fetch_building_register(session, sem, sgg, umd, jibun, bjdong_cache):
    """
    Asynchronously queries the Public Building Register API for a single address.
    """
    bun, ji = parse_jibun(jibun)
    if not bun or not ji:
        return {
            'sggCd': sgg, 'umdNm': umd, 'jibun': jibun,
            'parking_ratio': DEFAULT_PARKING_RATIO,
            'has_elevator': DEFAULT_HAS_ELEVATOR,
            'building_structure': DEFAULT_BUILDING_STRUCTURE,
            'is_commercial_villa': DEFAULT_IS_COMMERCIAL_VILLA,
            'total_floors': DEFAULT_TOTAL_FLOORS,
            'approval_date': ''
        }
        
    sigunguCd = str(int(float(sgg))).zfill(5)
    bjdongCd = bjdong_cache.get((sigunguCd, umd))
    
    params = {
        'serviceKey': SERVICE_KEY,
        'sigunguCd': sigunguCd,
        'bun': bun,
        'ji': ji,
        '_type': 'json'
    }
    if bjdongCd:
        params['bjdongCd'] = bjdongCd
    else:
        params['bjdongNm'] = umd

    async with sem:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(API_URL, params=params, timeout=12) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except Exception:
                            text = await response.text()
                            if "LIMITED" in text or "TRAFFIC" in text:
                                return 'LIMIT_EXCEEDED'
                            return 'PARSE_ERROR'
                            
                        res_obj = data.get('response', {})
                        header = res_obj.get('header', {})
                        result_code = header.get('resultCode')
                        result_msg = header.get('resultMsg', '')
                        
                        if result_code not in ('00', '000'):
                            if "LIMITED" in str(result_msg).upper() or "TRAFFIC" in str(result_msg).upper():
                                return 'LIMIT_EXCEEDED'
                            break
                            
                        body = res_obj.get('body', {})
                        if not body or 'items' not in body or not body['items'] or 'item' not in body['items']:
                            break
                            
                        items_data = body['items']['item']
                        if not items_data:
                            break
                            
                        if isinstance(items_data, list):
                            item = items_data[0]
                        else:
                            item = items_data
                            
                        # Feature engineering and extraction
                        indrMech = float(item.get('indrMechUtcnt') or 0)
                        indrAuto = float(item.get('indrAutoUtcnt') or 0)
                        oudrMech = float(item.get('oudrMechUtcnt') or 0)
                        oudrAuto = float(item.get('oudrAutoUtcnt') or 0)
                        hhld = float(item.get('hhldCnt') or 0)
                        parking_count = indrMech + indrAuto + oudrMech + oudrAuto
                        parking_ratio = parking_count / hhld if hhld > 0 else DEFAULT_PARKING_RATIO
                        
                        rideLft = float(item.get('rideLftCnt') or item.get('rideUseElvtCnt') or 0)
                        has_elevator = 1 if rideLft > 0 else 0
                        
                        strct_nm = str(item.get('strctCdNm') or '')
                        building_structure = 1 if '철근콘크리트' in strct_nm or 'RC' in strct_nm.upper() else 0
                        
                        purps_nm = str(item.get('mainPurpsCdNm') or '')
                        is_commercial_villa = 1 if '근린생활시설' in purps_nm else 0
                        
                        try:
                            total_floors = int(float(item.get('grndFlrCnt') or DEFAULT_TOTAL_FLOORS))
                        except Exception:
                            total_floors = DEFAULT_TOTAL_FLOORS
                            
                        approval_date = str(item.get('useAprvDt') or item.get('useAprDay') or '').strip()
                        if len(approval_date) > 8:
                            approval_date = approval_date[:8]
                            
                        return {
                            'sggCd': sgg, 'umdNm': umd, 'jibun': jibun,
                            'parking_ratio': parking_ratio,
                            'has_elevator': has_elevator,
                            'building_structure': building_structure,
                            'is_commercial_villa': is_commercial_villa,
                            'total_floors': total_floors,
                            'approval_date': approval_date
                        }
                    elif response.status == 429:
                        await asyncio.sleep(2)
                        continue
                    else:
                        break
            except Exception:
                await asyncio.sleep(1)
                continue
                
        return {
            'sggCd': sgg, 'umdNm': umd, 'jibun': jibun,
            'parking_ratio': DEFAULT_PARKING_RATIO,
            'has_elevator': DEFAULT_HAS_ELEVATOR,
            'building_structure': DEFAULT_BUILDING_STRUCTURE,
            'is_commercial_villa': DEFAULT_IS_COMMERCIAL_VILLA,
            'total_floors': DEFAULT_TOTAL_FLOORS,
            'approval_date': ''
        }

def save_checkpoint(data_list):
    """
    Appends intermediate results to the checkpoint CSV file.
    """
    new_df = pd.DataFrame(data_list)
    file_exists = os.path.exists(CHECKPOINT_PATH)
    if file_exists:
        try:
            if os.path.getsize(CHECKPOINT_PATH) == 0:
                file_exists = False
        except Exception:
            pass
    new_df.to_csv(CHECKPOINT_PATH, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

def merge_and_save():
    """
    Merges gathered checkpoint results back into the main nationwide dataset.
    Performs standard left join and applies fallback default imputations.
    """
    print("\n--- [시작] 최종 데이터 병합 및 세정 작업 ---")
    df = pd.read_csv(DATA_PATH)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print("[경고] 체크포인트 파일이 존재하지 않아 임베딩 병합을 건너뜁니다.")
        return
        
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
    # Ensure no duplicates in join keys from checkpoint
    checkpoint_df = checkpoint_df.drop_duplicates(subset=['sggCd', 'umdNm', 'jibun'], keep='last')
    
    # Remove older building columns if they already exist
    for col in ['parking_ratio', 'has_elevator', 'building_structure', 'is_commercial_villa', 'total_floors', 'approval_date']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Perform Left Join
    final_df = pd.merge(df, checkpoint_df, on=['sggCd', 'umdNm', 'jibun'], how='left')
    
    # Impute missing values for addresses that failed or had no records
    final_df['parking_ratio'] = final_df['parking_ratio'].fillna(DEFAULT_PARKING_RATIO)
    final_df['has_elevator'] = final_df['has_elevator'].fillna(DEFAULT_HAS_ELEVATOR).astype(int)
    final_df['building_structure'] = final_df['building_structure'].fillna(DEFAULT_BUILDING_STRUCTURE).astype(int)
    final_df['is_commercial_villa'] = final_df['is_commercial_villa'].fillna(DEFAULT_IS_COMMERCIAL_VILLA).astype(int)
    final_df['total_floors'] = final_df['total_floors'].fillna(DEFAULT_TOTAL_FLOORS).astype(int)
    
    # Precise age fallback matching buildYear
    def fill_approval_date(row):
        date_val = str(row['approval_date']).strip() if not pd.isna(row['approval_date']) else ''
        if len(date_val) >= 4:
            return date_val
        by = row['buildYear']
        if not pd.isna(by):
            try:
                year_int = int(float(by))
                return f"{year_int}0101"
            except Exception:
                pass
        return "20100101"
        
    final_df['approval_date'] = final_df.apply(fill_approval_date, axis=1)
    
    print(f" > 최종 본 파일 저장 중... ({DATA_PATH})")
    final_df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')
    print("--- [완료] 건축물대장 데이터 통합 완료 ---\n")

async def main_async():
    print("1. 실거래가 데이터 로드 중...")
    if not os.path.exists(DATA_PATH):
        print(f"[오류] 데이터 파일이 존재하지 않습니다: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    print("2. 고유 주소 추출 중...")
    unique_addrs = df.dropna(subset=['sggCd', 'umdNm', 'jibun'])[['sggCd', 'umdNm', 'jibun']].drop_duplicates()
    total_unique = len(unique_addrs)
    print(f" > 총 고유 주소 수: {total_unique}건")
    
    processed_keys = set()
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
            checkpoint_df = checkpoint_df.dropna(subset=['sggCd', 'umdNm', 'jibun'])
            for _, r in checkpoint_df.iterrows():
                processed_keys.add((r['sggCd'], r['umdNm'], str(r['jibun'])))
            print(f" > 체크포인트 로드 완료: 이미 완료된 고유 건수 {len(processed_keys)}건")
        except Exception as e:
            print(f" > 체크포인트 파일 읽기 실패 (처음부터 시작): {e}")
            
    to_process = []
    for _, r in unique_addrs.iterrows():
        key = (r['sggCd'], r['umdNm'], str(r['jibun']))
        if key not in processed_keys:
            to_process.append(r.to_dict())
            
    print(f" > 신규 처리할 주소 수: {len(to_process)}건")
    
    if len(to_process) == 0:
        print(" > 추가 수집 대상 주소가 없습니다. 병합 및 정규화를 최종 진행합니다.")
        merge_and_save()
        return
        
    unique_sgg = unique_addrs['sggCd'].map(lambda x: str(int(float(x))).zfill(5)).unique().tolist()
    bjdong_cache = fetch_bjdong_mapping(unique_sgg)
    
    sem = asyncio.Semaphore(30)
    connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        buffer = []
        completed = 0
        limit_reached = False
        start_time = time.time()
        
        batch_size = 1000
        total_to_process = len(to_process)
        already_collected = len(processed_keys)
        
        for i in range(0, total_to_process, batch_size):
            if limit_reached:
                break
                
            batch = to_process[i:i+batch_size]
            print(f"\n⚡ [{i // batch_size + 1}회차 배치] {len(batch)}건 API 호출 중...")
            
            tasks = [
                fetch_building_register(session, sem, item['sggCd'], item['umdNm'], item['jibun'], bjdong_cache)
                for item in batch
            ]
            
            results = await asyncio.gather(*tasks)
            
            batch_success = 0
            batch_failed = 0
            
            for res in results:
                if res == 'LIMIT_EXCEEDED':
                    print("\n🛑 [알림] 공공데이터 포털 트래픽 한도(LIMITED/TRAFFIC)에 도달했습니다. 금일 수집을 중단하고 진행된 분량을 저장합니다.")
                    limit_reached = True
                    break
                elif res == 'PARSE_ERROR':
                    batch_failed += 1
                    continue
                elif isinstance(res, dict):
                    buffer.append(res)
                    batch_success += 1
                    completed += 1
            
            if limit_reached:
                break
                
            # Save checkpoint immediately at the end of each batch to protect against Ctrl+C data loss
            if len(buffer) > 0:
                save_checkpoint(buffer)
                buffer = []
                
            elapsed = time.time() - start_time
            overall_collected = already_collected + completed
            overall_pct = (overall_collected / total_unique) * 100
            speed = completed / elapsed if elapsed > 0 else 0
            est_remaining = (total_to_process - completed) / speed if speed > 0 else 0
            
            print(f"📊 [전체 진행률: {overall_collected:,} / {total_unique:,} ({overall_pct:.2f}%)]")
            print(f"   └─ 신규 배치 결과 - 성공: +{batch_success}건 | 실패/결측: +{batch_failed}건")
            print(f"   └─ 통계 - 소요: {elapsed:.1f}초 | 평균 속도: {speed:.1f}건/초 | 남은 시간: {est_remaining/60:.1f}분")
            print(f"   └─ 체크포인트(building_checkpoint.csv) 안전하게 저장 완료!")
            
        if len(buffer) > 0:
            save_checkpoint(buffer)
            print(f" > 최종 잔여 {len(buffer)}건 체크포인트 저장 완료.")
            
    print("\n수집 프로세스가 완료되었습니다. 전체 병합 및 정리 단계로 진입합니다.")
    merge_and_save()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge-only":
        merge_and_save()
    else:
        asyncio.run(main_async())
