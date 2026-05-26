"""거래 데이터베이스

거래 이벤트를 데이터베이스에 저장하고 조회합니다.

Usage:
    from prediction.trade_database import TradeDatabase
    
    db = TradeDatabase("trades.db")
    db.save_event(event)
    trades = db.query_trades(start_date, end_date)
"""

import logging
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class TradeDatabase:
    """거래 데이터베이스."""
    
    def __init__(self, db_path: str = "trades.db"):
        """초기화.
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        _logger.info("[TRADE_DB] 데이터베이스 초기화: %s", db_path)
    
    def _create_tables(self):
        """테이블 생성."""
        cursor = self.conn.cursor()
        
        # events 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT,
                price REAL,
                size REAL,
                confidence TEXT,
                reason TEXT,
                signal_reason TEXT,
                stop_loss REAL,
                take_profit REAL,
                atr REAL,
                position_id TEXT,
                trailing_stops TEXT,
                atr_snapshots TEXT,
                partial_exits TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # positions 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                size REAL NOT NULL,
                confidence TEXT,
                signal_reason TEXT,
                stop_loss REAL,
                take_profit REAL,
                atr REAL,
                current_stop REAL,
                is_active INTEGER DEFAULT 1,
                max_favorable_excursion REAL DEFAULT 0.0,
                max_adverse_excursion REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # risk_metrics 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                position_id TEXT NOT NULL,
                current_price REAL,
                atr REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                distance_to_stop REAL,
                distance_to_take_profit REAL,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                risk_reward_ratio REAL,
                position_size_pct REAL,
                confidence TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_position_id ON events(position_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_position_id ON positions(position_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_metrics_position_id ON risk_metrics(position_id)")
        
        self.conn.commit()
    
    def save_event(self, event: Dict[str, Any]):
        """이벤트 DB 저장.
        
        Args:
            event: 이벤트 딕셔너리
        """
        cursor = self.conn.cursor()
        
        # 리스트/딕셔너리를 JSON으로 변환
        trailing_stops = json.dumps(event.get("trailing_stops", []))
        atr_snapshots = json.dumps(event.get("atr_snapshots", []))
        partial_exits = json.dumps(event.get("partial_exits", []))
        
        cursor.execute("""
            INSERT INTO events (
                event_type, timestamp, action, price, size, confidence, reason,
                signal_reason, stop_loss, take_profit, atr, position_id,
                trailing_stops, atr_snapshots, partial_exits
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.get("event_type"),
            event.get("timestamp"),
            event.get("action"),
            event.get("price"),
            event.get("size"),
            event.get("confidence"),
            event.get("reason"),
            event.get("signal_reason"),
            event.get("stop_loss"),
            event.get("take_profit"),
            event.get("atr"),
            event.get("position_id"),
            trailing_stops,
            atr_snapshots,
            partial_exits
        ))
        
        self.conn.commit()
        _logger.debug("[TRADE_DB] 이벤트 저장: %s", event.get("event_type"))
    
    def save_position(self, position: Dict[str, Any]):
        """포지션 DB 저장.
        
        Args:
            position: 포지션 딕셔너리
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO positions (
                position_id, action, entry_price, entry_time, size, confidence,
                signal_reason, stop_loss, take_profit, atr, current_stop,
                is_active, max_favorable_excursion, max_adverse_excursion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.get("position_id"),
            position.get("action"),
            position.get("entry_price"),
            position.get("entry_time"),
            position.get("size"),
            position.get("confidence"),
            position.get("signal_reason"),
            position.get("stop_loss"),
            position.get("take_profit"),
            position.get("atr"),
            position.get("current_stop"),
            1 if position.get("is_active", True) else 0,
            position.get("max_favorable_excursion", 0.0),
            position.get("max_adverse_excursion", 0.0)
        ))
        
        self.conn.commit()
        _logger.debug("[TRADE_DB] 포지션 저장: %s", position.get("position_id"))
    
    def save_risk_metrics(self, metrics: Dict[str, Any]):
        """리스크 메트릭 DB 저장.
        
        Args:
            metrics: 리스크 메트릭 딕셔너리
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO risk_metrics (
                timestamp, position_id, current_price, atr, unrealized_pnl,
                unrealized_pnl_pct, distance_to_stop, distance_to_take_profit,
                max_favorable_excursion, max_adverse_excursion, risk_reward_ratio,
                position_size_pct, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.get("timestamp"),
            metrics.get("position_id"),
            metrics.get("current_price"),
            metrics.get("atr"),
            metrics.get("unrealized_pnl"),
            metrics.get("unrealized_pnl_pct"),
            metrics.get("distance_to_stop"),
            metrics.get("distance_to_take_profit"),
            metrics.get("max_favorable_excursion"),
            metrics.get("max_adverse_excursion"),
            metrics.get("risk_reward_ratio"),
            metrics.get("position_size_pct"),
            metrics.get("confidence")
        ))
        
        self.conn.commit()
        _logger.debug("[TRADE_DB] 리스크 메트릭 저장: %s", metrics.get("position_id"))
    
    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        position_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """이벤트 조회.
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            event_type: 이벤트 타입
            position_id: 포지션 ID
        
        Returns:
            이벤트 리스트
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if position_id:
            query += " AND position_id = ?"
            params.append(position_id)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        events = []
        
        for row in rows:
            event = dict(zip(columns, row))
            # JSON 필드 파싱
            event["trailing_stops"] = json.loads(event.get("trailing_stops", "[]"))
            event["atr_snapshots"] = json.loads(event.get("atr_snapshots", "[]"))
            event["partial_exits"] = json.loads(event.get("partial_exits", "[]"))
            events.append(event)
        
        _logger.info("[TRADE_DB] 이벤트 조회: %d개", len(events))
        return events
    
    def query_trades(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """기간별 거래 조회.
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
        
        Returns:
            거래 리스트 (진입/청산 쌍)
        """
        # ENTRY와 EXIT 이벤트 조회
        entries = self.query_events(start_date, end_date, event_type="ENTRY")
        exits = self.query_events(start_date, end_date, event_type="EXIT")
        
        # 진입/청산 매칭
        trades = []
        for entry in entries:
            matching_exits = [e for e in exits if e["timestamp"] > entry["timestamp"]]
            if matching_exits:
                exit = min(matching_exits, key=lambda x: x["timestamp"])
                trades.append({
                    "entry": entry,
                    "exit": exit
                })
        
        _logger.info("[TRADE_DB] 거래 조회: %d개", len(trades))
        return trades
    
    def query_active_positions(self) -> List[Dict[str, Any]]:
        """활성 포지션 조회.
        
        Returns:
            활성 포지션 리스트
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE is_active = 1")
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        positions = [dict(zip(columns, row)) for row in rows]
        
        _logger.info("[TRADE_DB] 활성 포지션 조회: %d개", len(positions))
        return positions
    
    def query_risk_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        position_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """리스크 메트릭 조회.
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            position_id: 포지션 ID
        
        Returns:
            리스크 메트릭 리스트
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM risk_metrics WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if position_id:
            query += " AND position_id = ?"
            params.append(position_id)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        metrics = [dict(zip(columns, row)) for row in rows]
        
        _logger.info("[TRADE_DB] 리스크 메트릭 조회: %d개", len(metrics))
        return metrics
    
    def analyze_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """DB 기반 성능 분석.
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
        
        Returns:
            성능 분석 결과
        """
        trades = self.query_trades(start_date, end_date)
        
        if not trades:
            return {}
        
        # 기본 통계
        total_trades = len(trades)
        profits = []
        
        for trade in trades:
            entry = trade["entry"]
            exit = trade["exit"]
            
            if entry["action"] == "BUY":
                profit = (exit["price"] - entry["price"]) * entry["size"]
            else:
                profit = (entry["price"] - exit["price"]) * entry["size"]
            
            profits.append(profit)
        
        win_trades = sum(1 for p in profits if p > 0)
        loss_trades = sum(1 for p in profits if p <= 0)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        total_profit = sum(profits)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_profit": avg_profit
        }
    
    def close(self):
        """DB 연결 종료."""
        self.conn.close()
        _logger.info("[TRADE_DB] DB 연결 종료")
