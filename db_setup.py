# db_setup.py
from pathlib import Path
from sqlalchemy import create_engine, text
import os
import re

DOCS_DIR = "docs"
DB_FILE = "agent_data.db"
DB_URI = f"sqlite:///{DB_FILE}"

def _sanitize_mysql_for_sqlite(sql_script: str) -> str:
    # 1. Remove Comments
    script = re.sub(r'/\*!.*?\*/;', '', sql_script, flags=re.DOTALL)
    script = re.sub(r'^--.*$', '', script, flags=re.MULTILINE)
    
    # 2. Remove Backticks
    script = script.replace('`', '')
    
    # 3. Fix Data Types (Handling spaces: 'int (11)', 'int( 11 )')
    # This is the line that was likely failing before
    script = re.sub(r'(tiny|small|medium|big)?int\s*\(\s*\d+\s*\)', 'INTEGER', script, flags=re.IGNORECASE)
    script = re.sub(r'\bdouble\b', 'REAL', script, flags=re.IGNORECASE)
    script = re.sub(r'\bfloat\b', 'REAL', script, flags=re.IGNORECASE)
    
    # 4. Clean up Table Options
    script = re.sub(r'\)\s*(ENGINE|AUTO_INCREMENT|DEFAULT CHARSET)=[^;]*;', ');', script, flags=re.IGNORECASE)
    script = re.sub(r'(LOCK|UNLOCK) TABLES.*?;', '', script, flags=re.IGNORECASE)
    
    return script

def create_database_from_sql_files(db_uri: str = DB_URI):
    docs_path = Path(DOCS_DIR)
    
    # Reset Primary DB
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"[DB Setup] Deleted old {DB_FILE} to ensure fresh start.")
        except Exception:
            pass

    engine = create_engine(db_uri)
    sql_files = sorted(docs_path.glob("*.sql"))
    
    with engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON;"))
        
        if not sql_files:
             print("[DB Setup] No .sql files. Creating empty DB.")
             return db_uri

        for sql_file in sql_files:
            print(f"[DB Setup] Processing: {sql_file.name}")
            try:
                raw = sql_file.read_text(encoding='utf-8-sig')
                clean = _sanitize_mysql_for_sqlite(raw)
                
                # Split by semicolon
                statements = re.split(r';\s*$', clean, flags=re.MULTILINE)
                
                success = 0
                for stmt in statements:
                    if stmt.strip():
                        try:
                            connection.execute(text(stmt.strip()))
                            success += 1
                        except Exception:
                            pass
                print(f"[DB Setup] Imported {success} statements from {sql_file.name}")
            except Exception as e:
                print(f"[DB Error] {sql_file.name}: {e}")
        connection.commit()
    return db_uri

if __name__ == "__main__":
    create_database_from_sql_files(DB_URI)