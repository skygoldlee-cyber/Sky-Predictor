"""실시간 거래 모니터링 대시보드

웹 기반 실시간 거래 모니터링 대시보드를 제공합니다.

Usage:
    from prediction.trade_dashboard import TradeDashboard
    
    dashboard = TradeDashboard()
    dashboard.run()
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class TradeDashboard:
    """거래 모니터링 대시보드."""
    
    def __init__(self):
        """초기화."""
        self._position_tracker = None
        self._trade_logger = None
        self._trade_database = None
        _logger.info("[DASHBOARD] 대시보드 초기화")
    
    def _get_position_tracker(self):
        """포지션 트래커 인스턴스 가져오기."""
        if self._position_tracker is None:
            try:
                from prediction.trade_logger import get_position_tracker
                self._position_tracker = get_position_tracker()
            except Exception as e:
                _logger.error("[DASHBOARD] 포지션 트래커 로드 실패: %s", e)
        return self._position_tracker
    
    def _get_trade_logger(self):
        """거래 로거 인스턴스 가져오기."""
        if self._trade_logger is None:
            try:
                from prediction.trade_logger import get_trade_logger
                self._trade_logger = get_trade_logger()
            except Exception as e:
                _logger.error("[DASHBOARD] 거래 로거 로드 실패: %s", e)
        return self._trade_logger
    
    def _get_trade_database(self):
        """거래 데이터베이스 인스턴스 가져오기."""
        if self._trade_database is None:
            try:
                from prediction.trade_database import TradeDatabase
                self._trade_database = TradeDatabase("trades.db")
            except Exception as e:
                _logger.error("[DASHBOARD] 거래 데이터베이스 로드 실패: %s", e)
        return self._trade_database
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """활성 포지션 리스트 반환.
        
        Returns:
            활성 포지션 리스트
        """
        tracker = self._get_position_tracker()
        if tracker is None:
            return []
        
        positions = tracker.get_active_positions()
        return [
            {
                "position_id": pos.position_id,
                "action": pos.action,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat(),
                "size": pos.size,
                "confidence": pos.confidence,
                "signal_reason": pos.signal_reason,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "current_stop": pos.current_stop,
                "max_favorable_excursion": pos.max_favorable_excursion,
                "max_adverse_excursion": pos.max_adverse_excursion
            }
            for pos in positions
        ]
    
    def get_daily_pnl(self, days: int = 7) -> Dict[str, Any]:
        """일일 손익 반환.
        
        Args:
            days: 조회할 일수
        
        Returns:
            일일 손익 데이터
        """
        db = self._get_trade_database()
        if db is None:
            return {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            performance = db.analyze_performance(start_date, end_date)
            return {
                "period": f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}",
                "total_trades": performance.get("total_trades", 0),
                "win_rate": performance.get("win_rate", 0),
                "total_profit": performance.get("total_profit", 0),
                "avg_profit": performance.get("avg_profit", 0)
            }
        except Exception as e:
            _logger.error("[DASHBOARD] 일일 손익 조회 실패: %s", e)
            return {}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭 반환.
        
        Returns:
            리스크 메트릭 데이터
        """
        tracker = self._get_position_tracker()
        if tracker is None:
            return {}
        
        positions = tracker.get_active_positions()
        
        if not positions:
            return {
                "active_positions": 0,
                "total_exposure": 0,
                "avg_distance_to_stop": 0,
                "avg_distance_to_take_profit": 0
            }
        
        total_exposure = sum(pos.size * pos.entry_price for pos in positions)
        avg_distance_to_stop = sum(abs(pos.current_stop - pos.entry_price) for pos in positions) / len(positions)
        avg_distance_to_take_profit = sum(abs(pos.take_profit - pos.entry_price) for pos in positions) / len(positions)
        
        return {
            "active_positions": len(positions),
            "total_exposure": total_exposure,
            "avg_distance_to_stop": avg_distance_to_stop,
            "avg_distance_to_take_profit": avg_distance_to_take_profit
        }
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """최근 거래 반환.
        
        Args:
            limit: 반환할 거래 수
        
        Returns:
            최근 거래 리스트
        """
        db = self._get_trade_database()
        if db is None:
            return []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            trades = db.query_trades(start_date, end_date)
            
            # 최근 거래 정렬
            trades.sort(key=lambda x: x["exit"]["timestamp"], reverse=True)
            
            return trades[:limit]
        except Exception as e:
            _logger.error("[DASHBOARD] 최근 거래 조회 실패: %s", e)
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """대시보드 요약 반환.
        
        Returns:
            대시보드 요약 데이터
        """
        return {
            "active_positions": self.get_active_positions(),
            "daily_pnl": self.get_daily_pnl(),
            "risk_metrics": self.get_risk_metrics(),
            "recent_trades": self.get_recent_trades(),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_api(self, host: str = "127.0.0.1", port: int = 8000):
        """FastAPI 서버 실행.
        
        Args:
            host: 호스트 (기본값 127.0.0.1로 변경 - 보안 강화)
            port: 포트
        """
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            import uvicorn
            
            app = FastAPI(title="Trade Dashboard API")
            
            @app.get("/api/summary")
            def get_summary():
                return JSONResponse(content=self.get_summary())
            
            @app.get("/api/active-positions")
            def get_active_positions():
                return JSONResponse(content=self.get_active_positions())
            
            @app.get("/api/daily-pnl")
            def get_daily_pnl(days: int = 7):
                return JSONResponse(content=self.get_daily_pnl(days))
            
            @app.get("/api/risk-metrics")
            def get_risk_metrics():
                return JSONResponse(content=self.get_risk_metrics())
            
            @app.get("/api/recent-trades")
            def get_recent_trades(limit: int = 20):
                return JSONResponse(content=self.get_recent_trades(limit))
            
            _logger.info("[DASHBOARD] API 서버 시작: http://%s:%d", host, port)
            uvicorn.run(app, host=host, port=port)
            
        except ImportError:
            _logger.error("[DASHBOARD] FastAPI 또는 uvicorn 설치 필요: pip install fastapi uvicorn")
        except Exception as e:
            _logger.error("[DASHBOARD] API 서버 실행 실패: %s", e)
    
    def print_summary(self):
        """대시보드 요약 출력."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("거래 모니터링 대시보드")
        print("="*60)
        
        print(f"\n[활성 포지션] ({len(summary['active_positions'])}개)")
        for pos in summary['active_positions']:
            print(f"  {pos['action']} {pos['position_id']} @ {pos['entry_price']:.2f}")
            print(f"    사이즈: {pos['size']:.2f}, 신뢰도: {pos['confidence']}")
            print(f"    손절: {pos['stop_loss']:.2f}, 이익실현: {pos['take_profit']:.2f}")
        
        print(f"\n[일일 손익] {summary['daily_pnl'].get('period', 'N/A')}")
        print(f"  총 거래: {summary['daily_pnl'].get('total_trades', 0)}")
        print(f"  승률: {summary['daily_pnl'].get('win_rate', 0):.2%}")
        print(f"  총 수익: {summary['daily_pnl'].get('total_profit', 0):,.0f}원")
        print(f"  평균 수익: {summary['daily_pnl'].get('avg_profit', 0):,.0f}원")
        
        print("\n[리스크 메트릭]")
        print(f"  활성 포지션: {summary['risk_metrics'].get('active_positions', 0)}")
        print(f"  총 노출: {summary['risk_metrics'].get('total_exposure', 0):,.0f}원")
        print(f"  평균 손절 거리: {summary['risk_metrics'].get('avg_distance_to_stop', 0):.2f}")
        print(f"  평균 이익실현 거리: {summary['risk_metrics'].get('avg_distance_to_take_profit', 0):.2f}")
        
        print(f"\n[최근 거래] (최근 {len(summary['recent_trades'])}건)")
        for trade in summary['recent_trades'][:5]:
            entry = trade['entry']
            exit = trade['exit']
            profit = (exit['price'] - entry['price']) * entry['size'] if entry['action'] == 'BUY' else (entry['price'] - exit['price']) * entry['size']
            emoji = "✅" if profit > 0 else "❌"
            print(f"  {emoji} {entry['action']} @ {entry['price']:.2f} -> {exit['price']:.2f} ({profit:,.0f}원)")
        
        print("="*60)
