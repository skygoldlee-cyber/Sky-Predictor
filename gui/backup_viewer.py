"""백업 뷰어

백업 및 복구 기능을 제공하는 GUI 위젯.

Usage:
    from gui.backup_viewer import BackupViewer
    
    viewer = BackupViewer(parent=parent_widget)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BackupViewer:
    """백업 뷰어 위젯."""
    
    def __init__(self, parent=None, log_dir: Optional[Path] = None):
        """초기화.
        
        Args:
            parent: 부모 위젯
            log_dir: 로그 디렉토리 (기본: logs/trades)
        """
        try:
            from PySide6.QtWidgets import (
                QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                QLabel, QListWidget, QListWidgetItem, QGroupBox,
                QMessageBox, QProgressBar
            )
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont
        except ImportError:
            logger.error("[BackupViewer] PySide6 없음")
            self.widget = None
            return
        
        try:
            from prediction.backup_manager import BackupManager
            self.backup_manager = BackupManager(log_dir=log_dir)
        except Exception as e:
            logger.error("[BackupViewer] 백업 관리자 초기화 실패: %s", e)
            self.widget = None
            return
        
        # 위젯 생성
        self.widget = QWidget(None)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 제목
        title = QLabel("💾 백업 및 복구")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 백업 컨트롤 그룹
        control_group = QGroupBox("백업 제어")
        control_layout = QHBoxLayout(control_group)
        
        btn_create = QPushButton("📦 백업 생성")
        btn_create.clicked.connect(self._on_create_backup)
        control_layout.addWidget(btn_create)
        
        btn_refresh = QPushButton("↺ 새로고침")
        btn_refresh.clicked.connect(self._on_refresh)
        control_layout.addWidget(btn_refresh)
        
        control_layout.addStretch(1)
        layout.addWidget(control_group)
        
        # 백업 리스트 그룹
        list_group = QGroupBox("백업 목록")
        list_layout = QVBoxLayout(list_group)
        
        self.backup_list = QListWidget()
        list_layout.addWidget(self.backup_list)
        
        # 복구/삭제 버튼
        btn_row = QHBoxLayout()
        
        btn_restore = QPushButton("🔄 복구")
        btn_restore.clicked.connect(self._on_restore)
        btn_row.addWidget(btn_restore)
        
        btn_delete = QPushButton("🗑️ 삭제")
        btn_delete.clicked.connect(self._on_delete)
        btn_row.addWidget(btn_delete)
        
        btn_row.addStretch(1)
        list_layout.addLayout(btn_row)
        
        layout.addWidget(list_group)
        
        # 진행 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 상태 레이블
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # 초기 로드
        self._on_refresh()
    
    def _on_create_backup(self):
        """백업 생성 버튼 핸들러."""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 진행 중 표시
            self.status_label.setText("백업 생성 중...")
            
            backup_name = self.backup_manager.create_backup()
            
            self.progress_bar.setVisible(False)
            
            if backup_name:
                self.status_label.setText(f"백업 생성 완료: {backup_name}")
                self._on_refresh()
            else:
                self.status_label.setText("백업 생성 실패")
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"백업 생성 실패: {e}")
            logger.error("[BackupViewer] 백업 생성 실패: %s", e)
    
    def _on_restore(self):
        """복구 버튼 핸들러."""
        try:
            current_item = self.backup_list.currentItem()
            if not current_item:
                QMessageBox.warning(None, "경고", "백업을 선택하세요.")
                return
            
            backup_name = current_item.data(Qt.ItemDataRole.UserRole)
            
            # 확인 다이얼로그
            reply = QMessageBox.question(
                None,
                "복구 확인",
                f"'{backup_name}' 백업을 복구하시겠습니까?\n현재 로그는 자동으로 백업됩니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)
                self.status_label.setText("복구 중...")
                
                success = self.backup_manager.restore_backup(backup_name)
                
                self.progress_bar.setVisible(False)
                
                if success:
                    self.status_label.setText(f"복구 완료: {backup_name}")
                    QMessageBox.information(None, "성공", f"백업 복구 완료: {backup_name}")
                else:
                    self.status_label.setText("복구 실패")
                    QMessageBox.critical(None, "실패", "백업 복구 실패")
                    
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"복구 실패: {e}")
            logger.error("[BackupViewer] 복구 실패: %s", e)
    
    def _on_delete(self):
        """삭제 버튼 핸들러."""
        try:
            current_item = self.backup_list.currentItem()
            if not current_item:
                QMessageBox.warning(None, "경고", "백업을 선택하세요.")
                return
            
            backup_name = current_item.data(Qt.ItemDataRole.UserRole)
            
            # 확인 다이얼로그
            reply = QMessageBox.question(
                None,
                "삭제 확인",
                f"'{backup_name}' 백업을 삭제하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                success = self.backup_manager.delete_backup(backup_name)
                
                if success:
                    self.status_label.setText(f"삭제 완료: {backup_name}")
                    self._on_refresh()
                else:
                    self.status_label.setText("삭제 실패")
                    
        except Exception as e:
            self.status_label.setText(f"삭제 실패: {e}")
            logger.error("[BackupViewer] 삭제 실패: %s", e)
    
    def _on_refresh(self):
        """새로고침 버튼 핸들러."""
        try:
            self.backup_list.clear()
            
            backups = self.backup_manager.list_backups()
            
            for backup in backups:
                item = QListWidgetItem(
                    f"{backup.name} | {backup.timestamp} | "
                    f"파일: {backup.file_count} | 크기: {backup.size:,} bytes"
                )
                item.setData(Qt.ItemDataRole.UserRole, backup.name)
                self.backup_list.addItem(item)
            
            self.status_label.setText(f"백업 {len(backups)}개 로드 완료")
            
        except Exception as e:
            self.status_label.setText(f"로드 실패: {e}")
            logger.error("[BackupViewer] 로드 실패: %s", e)


def attach_backup_viewer(
    parent: Any,
    log_dir: Optional[Path] = None,
) -> Optional[BackupViewer]:
    """
    부모 위젯에 백업 뷰어를 삽입한다.
    
    Parameters
    ----------
    parent  : QWidget/QVBoxLayout
        부모 위젯 또는 레이아웃
    log_dir : Path
        로그 디렉토리 (기본: logs/trades)
    
    Returns
    -------
    BackupViewer | None
    
    ─────── 사용 예 ───────────────────────────────────────
    
    from gui.backup_viewer import attach_backup_viewer
    
    # QVBoxLayout에 추가
    backup_viewer = attach_backup_viewer(right_root)
    
    # QWidget에 추가
    backup_viewer = attach_backup_viewer(container)
    parent_layout.addWidget(backup_viewer.widget)
    ──────────────────────────────────────────────────────────
    """
    try:
        from PySide6.QtWidgets import QVBoxLayout
    except ImportError:
        logger.error("[attach_backup_viewer] PySide6 없음")
        return None
    
    viewer = BackupViewer(parent=None, log_dir=log_dir)
    
    if viewer.widget is None:
        return None
    
    # 부모가 QVBoxLayout이면 위젯 추가
    if hasattr(parent, 'addWidget'):
        parent.addWidget(viewer.widget)
    # 부모가 QWidget이면 레이아웃에 추가
    elif hasattr(parent, 'layout'):
        parent.layout().addWidget(viewer.widget)
    
    return viewer
