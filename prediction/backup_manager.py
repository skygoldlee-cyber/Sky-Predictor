"""백업 및 복구 시스템

거래 로그 파일의 자동 백업, 손상된 로그 복구, 백업 일정 설정 기능을 제공합니다.

Usage:
    from prediction.backup_manager import BackupManager
    
    manager = BackupManager(log_dir=Path("logs/trades"))
    manager.create_backup()
    manager.restore_backup("backup_2026-04-25_120000")
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """백업 정보."""
    name: str
    timestamp: str
    size: int
    file_count: int


class BackupManager:
    """백업 관리자."""
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        max_backups: int = 30
    ):
        """초기화.
        
        Args:
            log_dir: 로그 디렉토리 (기본: logs/trades)
            backup_dir: 백업 디렉토리 (기본: logs/backups)
            max_backups: 최대 백업 보관 수 (기본: 30)
        """
        self.log_dir = log_dir or Path("logs/trades")
        self.backup_dir = backup_dir or Path("logs/backups")
        self.max_backups = max_backups
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_thread: Optional[threading.Thread] = None
        self._backup_running = False
        logger.info("[BackupManager] 초기화 완료: log_dir=%s, backup_dir=%s", 
                   self.log_dir, self.backup_dir)
    
    def create_backup(self) -> Optional[str]:
        """백업 생성.
        
        Returns:
            백업 이름 (실패 시 None)
        """
        try:
            if not self.log_dir.exists():
                logger.warning("[BackupManager] 로그 디렉토리 없음: %s", self.log_dir)
                return None
            
            # 백업 이름 생성
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # 백업 폴더 생성
            backup_path.mkdir(exist_ok=True)
            
            # 로그 파일 복사
            file_count = 0
            total_size = 0
            
            for log_file in self.log_dir.glob("*.jsonl"):
                dest_file = backup_path / log_file.name
                shutil.copy2(log_file, dest_file)
                file_count += 1
                total_size += dest_file.stat().st_size
            
            # 백업 정보 저장
            backup_info = {
                "timestamp": timestamp,
                "file_count": file_count,
                "total_size": total_size,
                "source_dir": str(self.log_dir)
            }
            
            info_file = backup_path / "backup_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            logger.info(
                "[BackupManager] 백업 생성 완료: %s (파일: %d, 크기: %d bytes)",
                backup_name, file_count, total_size
            )
            
            # 오래된 백업 정리
            self._cleanup_old_backups()
            
            return backup_name
            
        except Exception as e:
            logger.error("[BackupManager] 백업 생성 실패: %s", e)
            return None
    
    def restore_backup(self, backup_name: str) -> bool:
        """백업 복구.
        
        Args:
            backup_name: 백업 이름
        
        Returns:
            성공 여부
        """
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.warning("[BackupManager] 백업 없음: %s", backup_name)
                return False
            
            # 현재 로그 백업
            current_backup = self.log_dir / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.log_dir.exists():
                shutil.copytree(self.log_dir, current_backup)
                logger.info("[BackupManager] 현재 로그 백업 완료: %s", current_backup)
            
            # 백업에서 복구
            if self.log_dir.exists():
                shutil.rmtree(self.log_dir)
            
            shutil.copytree(backup_path, self.log_dir)
            
            # 백업 정보 파일 삭제 (복구된 폴더에 있으면)
            info_file = self.log_dir / "backup_info.json"
            if info_file.exists():
                info_file.unlink()
            
            logger.info("[BackupManager] 백업 복구 완료: %s", backup_name)
            return True
            
        except Exception as e:
            logger.error("[BackupManager] 백업 복구 실패: %s", e)
            return False
    
    def list_backups(self) -> List[BackupInfo]:
        """백업 리스트.
        
        Returns:
            백업 정보 리스트
        """
        backups = []
        
        try:
            for backup_path in self.backup_dir.iterdir():
                if not backup_path.is_dir():
                    continue
                
                info_file = backup_path / "backup_info.json"
                if not info_file.exists():
                    continue
                
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    
                    backups.append(BackupInfo(
                        name=backup_path.name,
                        timestamp=info.get("timestamp", ""),
                        size=info.get("total_size", 0),
                        file_count=info.get("file_count", 0)
                    ))
                except Exception as e:
                    logger.debug("[BackupManager] 백업 정보 읽기 실패: %s", e)
            
            # 타임스탬프 역순 정렬
            backups.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error("[BackupManager] 백업 리스트 실패: %s", e)
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """백업 삭제.
        
        Args:
            backup_name: 백업 이름
        
        Returns:
            성공 여부
        """
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.warning("[BackupManager] 백업 없음: %s", backup_name)
                return False
            
            shutil.rmtree(backup_path)
            logger.info("[BackupManager] 백업 삭제 완료: %s", backup_name)
            return True
            
        except Exception as e:
            logger.error("[BackupManager] 백업 삭제 실패: %s", e)
            return False
    
    def _cleanup_old_backups(self):
        """오래된 백업 정리."""
        try:
            backups = self.list_backups()
            
            if len(backups) <= self.max_backups:
                return
            
            # 오래된 백업 삭제
            for backup in backups[self.max_backups:]:
                self.delete_backup(backup.name)
            
            logger.info("[BackupManager] 오래된 백업 정리 완료: %d개 삭제", 
                       len(backups) - self.max_backups)
            
        except Exception as e:
            logger.error("[BackupManager] 백업 정리 실패: %s", e)
    
    def start_auto_backup(self, interval_hours: int = 24):
        """자동 백업 시작.
        
        Args:
            interval_hours: 백업 간격 (시간)
        """
        if self._backup_running:
            logger.warning("[BackupManager] 자동 백업 이미 실행 중")
            return
        
        self._backup_running = True
        self._backup_thread = threading.Thread(
            target=self._auto_backup_loop,
            args=(interval_hours,),
            daemon=True
        )
        self._backup_thread.start()
        logger.info("[BackupManager] 자동 백업 시작: 간격=%d시간", interval_hours)
    
    def stop_auto_backup(self):
        """자동 백업 중지."""
        self._backup_running = False
        if self._backup_thread:
            self._backup_thread.join(timeout=5)
        logger.info("[BackupManager] 자동 백업 중지")
    
    def _auto_backup_loop(self, interval_hours: int):
        """자동 백업 루프.
        
        Args:
            interval_hours: 백업 간격 (시간)
        """
        interval_seconds = interval_hours * 3600
        
        while self._backup_running:
            try:
                self.create_backup()
            except Exception as e:
                logger.error("[BackupManager] 자동 백업 실패: %s", e)
            
            # 대기
            for _ in range(interval_seconds):
                if not self._backup_running:
                    break
                time.sleep(1)
    
    def repair_log_file(self, log_file: Path) -> bool:
        """손상된 로그 파일 복구.
        
        Args:
            log_file: 로그 파일 경로
        
        Returns:
            성공 여부
        """
        try:
            if not log_file.exists():
                logger.warning("[BackupManager] 로그 파일 없음: %s", log_file)
                return False
            
            # 백업 파일 생성
            backup_file = log_file.with_suffix(".jsonl.bak")
            shutil.copy2(log_file, backup_file)
            
            # 손상된 라인 제거 및 복구
            repaired_lines = []
            corrupted_count = 0
            
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        json.loads(line)
                        repaired_lines.append(line)
                    except Exception:
                        corrupted_count += 1
                        logger.debug("[BackupManager] 손상된 라인 제거: %s", line[:50])
            
            # 복구된 내용 쓰기
            with open(log_file, "w", encoding="utf-8") as f:
                for line in repaired_lines:
                    f.write(line + "\n")
            
            logger.info(
                "[BackupManager] 로그 파일 복구 완료: %s (손상된 라인: %d)",
                log_file, corrupted_count
            )
            
            return True
            
        except Exception as e:
            logger.error("[BackupManager] 로그 파일 복구 실패: %s", e)
            return False
