import requests
import time
import xml.etree.ElementTree as ET
import csv
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def get_lawd_codes():
    url = "https://grpc-proxy-server-mkvo6j4wsq-du.a.run.app/v1/regcodes?regcode_pattern=*00000"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        codes = {}
        for item in data['regcodes']:
            code5 = item['code'][:5]
            name = item['name']
            if code5.endswith('000'):
                continue
            codes[code5] = name
            
        parent_codes = []
        for code in codes:
            if code.endswith('0'):
                prefix = code[:4]
                has_children = any(c != code and c.startswith(prefix) for c in codes)
                if has_children:
                    parent_codes.append(code)
                    
        for pc in parent_codes:
            del codes[pc]
            
        return codes
    except Exception as e:
        print(f"지역코드 수집 실패: {e}")
        return {}

def fetch_month_data(lawd_cd, region_name, deal_ymd, service_key_dec):
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade"
    params = {
        'serviceKey': service_key_dec,
        'LAWD_CD': lawd_cd,
        'DEAL_YMD': deal_ymd,
        'numOfRows': '9999'
    }
    
    retry_count = 0
    max_retries = 3
    
    columns = [
        'sggCd', 'umdNm', 'jibun', 'mhouseNm', 'buildYear', 
        'houseType', 'excluUseAr', 'landAr', 'floor', 
        'dealYear', 'dealMonth', 'dealDay', 'dealAmount',
        'buyerGbn', 'slerGbn', 'dealingGbn', 'estateAgentSggNm', 
        'cdealType', 'cdealDay', 'rgstDate'
    ]
    
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                result_code = root.find('.//resultCode')
                result_msg = root.find('.//resultMsg')
                
                if result_code is not None and result_code.text not in ('00', '000'):
                    if "LIMITED" in result_msg.text.upper():
                        time.sleep(3)
                        retry_count += 1
                        continue
                    else:
                        print(f"[{region_name} {lawd_cd} - {deal_ymd}] API 오류: {result_msg.text}")
                        return [], f"{lawd_cd}_{deal_ymd}", False
                    
                items = root.findall('.//item')
                rows = []
                for item in items:
                    row = {}
                    for col in columns:
                        node = item.find(col)
                        row[col] = node.text.strip() if node is not None and node.text else ''
                    rows.append(row)
                return rows, f"{lawd_cd}_{deal_ymd}", True
            else:
                retry_count += 1
                time.sleep(2)
        except Exception as e:
            retry_count += 1
            time.sleep(2)
            
    print(f"[{region_name} {lawd_cd} - {deal_ymd}] 최종 요청 실패")
    return [], f"{lawd_cd}_{deal_ymd}", False

def fetch_nationwide():
    service_key_dec = "GF0Lq9LWPlZV7Ga1tMaCqZDhb06lzroW4fwEwQy9BfDy82xa3bPReEfNfTUBi/g4mCd/PfHGZu1Djjs4VdP0iQ=="
    
    print("--- 전국 지역코드 조회 중 ---")
    regions = get_lawd_codes()
    if not regions:
        print("지역코드를 가져오지 못했습니다. 종료합니다.")
        return
        
    print(f"대상 시군구 수: {len(regions)}개")
    
    now = datetime.datetime.now()
    deal_ymds = []
    for i in range(60):
        y = now.year + (now.month - i - 1) // 12
        m = (now.month - i - 1) % 12 + 1
        deal_ymds.append(f"{y}{m:02d}")
    deal_ymds.reverse()
            
    os.makedirs('data/raw', exist_ok=True)
    start_ym = deal_ymds[0]
    end_ym = deal_ymds[-1]
    output_file = f'data/raw/nationwide_RHTrade_{start_ym}_{end_ym}.csv'
    log_file = 'data/raw/fetch_progress.log'
    
    completed_tasks = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            completed_tasks = set(line.strip() for line in f)
            
    columns = [
        'sggCd', 'umdNm', 'jibun', 'mhouseNm', 'buildYear', 
        'houseType', 'excluUseAr', 'landAr', 'floor', 
        'dealYear', 'dealMonth', 'dealDay', 'dealAmount',
        'buyerGbn', 'slerGbn', 'dealingGbn', 'estateAgentSggNm', 
        'cdealType', 'cdealDay', 'rgstDate'
    ]
    
    file_exists = os.path.exists(output_file)
    total_new_records = 0
    start_time = time.time()
    
    write_lock = threading.Lock()
    
    # 1. 큐에 모든 작업 생성
    tasks = []
    for lawd_cd, region_name in regions.items():
        for deal_ymd in deal_ymds:
            task_id = f"{lawd_cd}_{deal_ymd}"
            if task_id not in completed_tasks:
                tasks.append((lawd_cd, region_name, deal_ymd))
                
    print(f"\n--- 전국 데이터 수집 시작 (남은 작업: {len(tasks)}개) ---")
    
    with open(output_file, 'a', newline='', encoding='utf-8-sig') as f_out, \
         open(log_file, 'a', encoding='utf-8') as f_log:
         
        writer = csv.DictWriter(f_out, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
            
        # 2. 멀티스레딩 (5개 동시 실행)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_month_data, t[0], t[1], t[2], service_key_dec): t for t in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                rows, task_id, success = future.result()
                
                if success:
                    with write_lock:
                        for row in rows:
                            writer.writerow(row)
                        f_log.write(task_id + '\n')
                        
                        # 버퍼 즉시 플러시
                        f_out.flush()
                        f_log.flush()
                        
                        total_new_records += len(rows)
                        
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed if elapsed > 0 else 0
                    rem_time = (len(tasks) - (i + 1)) / speed if speed > 0 else 0
                    print(f"[진행률: {i+1}/{len(tasks)}] 수집된 데이터: {total_new_records}건 | 예상 남은 시간: {rem_time/60:.1f}분")

    end_time = time.time()
    duration = end_time - start_time
    print("\n--- 전국 데이터 수집 작업 완료 ---")
    print(f"새로 수집된 데이터(건수): {total_new_records}건")
    print(f"진행 시간: {duration/60:.2f}분")

if __name__ == "__main__":
    fetch_nationwide()
