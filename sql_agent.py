# sql_agent.py
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine, text
from pathlib import Path

class SQLAgent:
    def __init__(self, db_uri: str, llm_model: str, llm_temperature: float):
        self.db_uri = db_uri
        self.agent_executor = None
        self.attached_dbs = [] 
        self.main_db_status = "OK"

        print(f"[SQL] Connecting to primary database: {db_uri}")
        
        # --- RESILIENT CONNECTION ---
        try:
            self.engine = create_engine(db_uri)
            self.db = SQLDatabase(self.engine)
            # Test connection
            self.db.get_usable_table_names()
            print("[SQL] Primary database connected successfully.")
        except Exception as e:
            print(f"[SQL ERROR] Primary Database Failed: {e}")
            print("[SQL] Switching to In-Memory Fallback...")
            self.main_db_status = "FAILED"
            self.engine = create_engine("sqlite:///:memory:")
            self.db = SQLDatabase(self.engine)

        # --- ATTACH EXTERNAL DBS ---
        self._attach_external_databases()

        self.llm = ChatOllama(model=llm_model, temperature=llm_temperature)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        # --- BUILD AGGRESSIVE SCHEMA MAP ---
        # We build a list of "Table -> specific SQL command" to help the small LLM
        schema_rules = self._get_strict_schema_map()
        
        system_prompt = (
            "You are an SQL Agent. You MUST follow these rules strictly.\n"
            "You have access to the following tables. You MUST use the exact Table Name provided below:\n\n"
            "**AVAILABLE TABLES (Copy these names exactly):**\n"
            f"{schema_rules}\n\n"
            "**RULES:**\n"
            "1. If a table is listed as `chinook.Track`, you MUST write `SELECT ... FROM chinook.Track`.\n"
            "2. DO NOT write `FROM Track`. That will fail.\n"
            "3. If you get a 'no such table' error, look at the list above and correct the prefix.\n"
            "4. Always limit results to 5 rows.\n"
            "5. Output the raw query results at the end labeled 'Raw Query Results:'."
        )

        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            system_prompt=system_prompt,
            verbose=True, 
            handle_parsing_errors=True
        )

    def _attach_external_databases(self):
        docs_dir = Path("docs")
        if not docs_dir.exists(): return

        valid_extensions = ["*.db", "*.sqlite", "*.sqlite3"]
        extra_dbs = []
        for ext in valid_extensions:
            extra_dbs.extend(docs_dir.glob(ext))
        
        # Exclude main DB
        current_db_name = "agent_data.db"
        extra_dbs = [f for f in extra_dbs if f.name != current_db_name]

        if not extra_dbs: return

        print(f"[SQL] Found {len(extra_dbs)} external databases. Attaching...")
        
        with self.engine.connect() as conn:
            for db_file in extra_dbs:
                # Alias cleaning
                alias = db_file.stem.replace(" ", "_").replace("-", "_").replace(".", "_").lower()
                if alias.endswith("_sqlite"): alias = alias.replace("_sqlite", "")
                
                db_path = str(db_file.absolute()).replace("\\", "/")
                try:
                    conn.execute(text(f"ATTACH DATABASE '{db_path}' AS {alias}"))
                    self.attached_dbs.append(alias)
                    print(f"   + Attached '{db_file.name}' as alias '{alias}'")
                except Exception as e:
                    print(f"   ! Failed to attach {db_file.name}: {e}")

    def _get_strict_schema_map(self):
        """Generates a strict list of 'TableName' for the prompt."""
        rules = []
        
        # 1. Main DB Tables
        try:
            if self.main_db_status == "OK":
                for t in self.db.get_usable_table_names():
                    rules.append(f"- Table: '{t}' -> Use SQL: `FROM {t}`")
        except: pass

        # 2. Attached DB Tables
        if self.attached_dbs:
            with self.engine.connect() as conn:
                for alias in self.attached_dbs:
                    try:
                        res = conn.execute(text(f"SELECT name FROM {alias}.sqlite_master WHERE type='table'"))
                        for row in res:
                            t = row[0]
                            if not t.startswith("sqlite_"):
                                rules.append(f"- Table: '{t}' -> Use SQL: `FROM {alias}.{t}`")
                    except Exception:
                        pass
        
        if not rules:
            return "(No tables found)"
        return "\n".join(rules)

    def ask(self, query: str):
        if not self.agent_executor: return "SQL Agent unavailable.", ""
        try:
            result = self.agent_executor.invoke({"input": query})
            answer = result.get("output", "No summary.")
            if "Raw Query Results:" in answer:
                parts = answer.split("Raw Query Results:", 1)
                return parts[0].strip(), parts[1].strip()
            return answer, ""
        except Exception as e:
            return f"Error: {e}", ""