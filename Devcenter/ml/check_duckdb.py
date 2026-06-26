import duckdb

DB_PATH = "Devcenter/data/duckdb/market_data.duckdb"

con = duckdb.connect(DB_PATH, read_only=True)

# 테이블 목록 확인
tables = con.execute("SHOW TABLES").df()
print("Tables:", tables)

# 데이터 확인
if len(tables) > 0:
    table_name = tables['name'].iloc[0]
    print(f"\nChecking table: {table_name}")
    
    # 데이터 범위 확인
    result = con.execute(f"SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time, COUNT(*) as count FROM {table_name}").df()
    print(f"Data range: {result}")
    
    # 샘플 데이터 확인
    sample = con.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
    print(f"\nSample data:\n{sample}")

con.close()
