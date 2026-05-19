import pandas as pd
import numpy as np
import json
import os
import glob

print("1. Loading raw data...")
# Read raw data
raw_path = glob.glob('c:/VSA-AVM-Engine/data/raw/nationwide_RHTrade_*.csv')[0]
df = pd.read_csv(raw_path, low_memory=False)

print("2. Filtering data...")
# cdealType == 'O' 제거 (계약 취소 데이터)
df = df[df['cdealType'].fillna('') != 'O']

# dealingGbn == '직거래' 제거 (이상치 처리)
df = df[df['dealingGbn'].fillna('') != '직거래']

print("3. Feature Engineering...")
# dealAmount: 정수형 변환
df['dealAmount'] = df['dealAmount'].astype(str).str.replace(',', '').astype(int)

# Age: dealYear - buildYear
df['Age'] = df['dealYear'] - pd.to_numeric(df['buildYear'], errors='coerce')

# Price_per_m2: 면적당 단가
df['excluUseAr'] = pd.to_numeric(df['excluUseAr'], errors='coerce')
df['Price_per_m2'] = df['dealAmount'] / df['excluUseAr']

print("4. Loading regcodes for Time_Adjustment mapping...")
with open('c:/VSA-AVM-Engine/regcodes.json', 'r', encoding='utf-8') as f:
    regcodes = json.load(f)['regcodes']

sido_mapping = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "강원도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주"
}

sgg_to_region = {}
for item in regcodes:
    code = int(item['code'][:5])
    name = item['name']
    parts = name.split()
    sido_full = parts[0]
    sido_short = sido_mapping.get(sido_full, sido_full)
    last_token = parts[-1]
    sgg_to_region[code] = {'sido': sido_short, 'region': last_token}

df['sido'] = df['sggCd'].map(lambda x: sgg_to_region.get(x, {}).get('sido'))
df['region'] = df['sggCd'].map(lambda x: sgg_to_region.get(x, {}).get('region'))

# Build sub-district to parent city mapping from regcodes
sub_to_parent = {}
for item in regcodes:
    parts = item['name'].split()
    if len(parts) == 3:
        sido_full = parts[0]
        sido_short = sido_mapping.get(sido_full, sido_full)
        parent = parts[1]
        sub = parts[2]
        sub_to_parent[(sido_short, sub)] = parent

# Create land_region column in main data for merging with land index
df['land_region'] = df.apply(lambda r: sub_to_parent.get((r['sido'], r['region']), r['region']), axis=1)

print("5. Loading and processing land index files...")
land_index_files = glob.glob('c:/VSA-AVM-Engine/data/external/land_index_raw/land_index_*.csv')
land_dfs = []

for f in land_index_files:
    if 'utf8' in f: continue
    
    # Read headers from the first row and drop the "지수" row
    temp = pd.read_csv(f, encoding='cp949', header=0)
    temp = temp.iloc[1:]
    
    # Check for unnamed columns at the end that might be parsed
    temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
    
    cols = list(temp.columns)
    date_cols = [c for c in cols if '년' in c]
    id_vars = [c for c in cols if c not in date_cols]
    
    temp_melt = temp.melt(id_vars=id_vars, value_vars=date_cols, var_name='year_month', value_name='land_index')
    
    # Extract year and month from '2021년 5월'
    extracted = temp_melt['year_month'].str.extract(r'(\d+)년\s*(\d+)월')
    temp_melt['dealYear'] = extracted[0].astype(float)
    temp_melt['dealMonth'] = extracted[1].astype(float)
    
    temp_melt['land_index'] = pd.to_numeric(temp_melt['land_index'].astype(str).str.replace(',', ''), errors='coerce')
    
    if '지역.1' in id_vars:
        temp_melt = temp_melt.rename(columns={'지역': 'sido', '지역.1': 'land_region'})
    else:
        temp_melt = temp_melt.rename(columns={'지역': 'sido'})
        temp_melt['land_region'] = '세종특별자치시'
        
    land_dfs.append(temp_melt[['sido', 'land_region', 'dealYear', 'dealMonth', 'land_index']])

land_df = pd.concat(land_dfs, ignore_index=True)
land_df = land_df.dropna(subset=['land_index'])

# Find the latest index for each region based on land_region
latest_idx = land_df.sort_values(['sido', 'land_region', 'dealYear', 'dealMonth']).groupby(['sido', 'land_region']).last().reset_index()
latest_idx = latest_idx.rename(columns={'land_index': 'current_land_index'})[['sido', 'land_region', 'current_land_index']]

print("6. Merging land index with main data...")

# Build a grid of all dates to propagate (ffill/bfill) indices for future/past months
tx_dates = df[['sido', 'land_region', 'dealYear', 'dealMonth']].drop_duplicates()
combined_dates = pd.concat([land_df[['sido', 'land_region', 'dealYear', 'dealMonth']], tx_dates]).drop_duplicates().sort_values(['sido', 'land_region', 'dealYear', 'dealMonth'])

grid_df = pd.merge(combined_dates, land_df, on=['sido', 'land_region', 'dealYear', 'dealMonth'], how='left')
# Forward-fill (for future dates) and backward-fill (for past dates) within each region
grid_df['land_index'] = grid_df.groupby(['sido', 'land_region'])['land_index'].ffill().bfill()

# Merge onto df
df = pd.merge(df, grid_df, on=['sido', 'land_region', 'dealYear', 'dealMonth'], how='left')
df = pd.merge(df, latest_idx, on=['sido', 'land_region'], how='left')

# Calculate Time_Adjustment
df['Time_Adjustment'] = df['current_land_index'] / df['land_index']

# Drop temporary land_region column
df = df.drop(columns=['land_region'])

output_dir = 'c:/VSA-AVM-Engine/data/processed'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'nationwide_RHTrade_processed.csv')
print(f"7. Saving to {output_path}...")
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("Done!")
