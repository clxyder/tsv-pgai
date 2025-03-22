from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session


class MockPlpy:
    def __init__(self, session: Session) -> None:
        self.session = session
    
    def execute(self, sql: str):
        self.session(text(sql))
    
    def commit(self):
        self.session.commit()
    
    def warning(self, msg: str):
        print(f"WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"ERROR: {msg}")
    
    def notice(self, msg: str):
        print(f"NOTICE: {msg}")

    def prepare(self, sql: str, argv: List[str]):
        pass
