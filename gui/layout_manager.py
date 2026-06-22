"""레이아웃 관리자

사용자 정의 가능한 레이아웃 기능을 제공합니다.

Usage:
    from gui.layout_manager import LayoutManager
    
    manager = LayoutManager(tab_widget)
    manager.save_layout("default")
    manager.load_layout("default")

레이아웃 설정 파일은 layout/ 디렉토리에 저장됩니다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LayoutConfig:
    """레이아웃 설정."""
    name: str
    tab_order: List[str]  # 탭 순서
    visible_tabs: List[str]  # 표시할 탭
    tab_width: int  # 탭 너비
    tab_height: int  # 탭 높이


class LayoutManager:
    """레이아웃 관리자."""
    
    def __init__(self, tab_widget, config_dir: Optional[Path] = None):
        """초기화.
        
        Args:
            tab_widget: QTabWidget 인스턴스
            config_dir: 설정 디렉토리 (기본: layout)
        """
        self.tab_widget = tab_widget
        self.config_dir = config_dir or Path("layout")
        
        # 절대 경로로 변환
        if not self.config_dir.is_absolute():
            self.config_dir = Path.cwd() / self.config_dir
        
        # 폴더 생성 (이미 존재하면 무시)
        if not self.config_dir.exists():
            try:
                self.config_dir.mkdir(parents=True)
                logger.info("[LayoutManager] 폴더 생성 완료: %s", self.config_dir)
            except Exception as e:
                logger.error("[LayoutManager] 폴더 생성 실패: %s", e)
        else:
            logger.info("[LayoutManager] 폴더 이미 존재: %s", self.config_dir)
        
        self.current_layout: Optional[LayoutConfig] = None
        
        logger.info("[LayoutManager] 초기화 완료: tab_widget=%s, config_dir=%s", 
                   type(tab_widget).__name__, self.config_dir)
    
    def get_current_layout(self) -> LayoutConfig:
        """현재 레이아웃 가져오기.
        
        Returns:
            현재 레이아웃 설정
        """
        try:
            tab_count = self.tab_widget.count()
            tab_order = []
            visible_tabs = []
            
            for i in range(tab_count):
                tab_text = self.tab_widget.tabText(i)
                tab_order.append(tab_text)
                if not self.tab_widget.isTabVisible(i):
                    visible_tabs.append(tab_text)
            
            # 모든 탭이 표시되면 visible_tabs 비움
            if len(visible_tabs) == tab_count:
                visible_tabs = []
            
            geometry = self.tab_widget.geometry()
            
            return LayoutConfig(
                name="current",
                tab_order=tab_order,
                visible_tabs=visible_tabs,
                tab_width=geometry.width(),
                tab_height=geometry.height()
            )
        except Exception as e:
            logger.error("[LayoutManager] 레이아웃 가져오기 실패: %s", e)
            return LayoutConfig(
                name="current",
                tab_order=[],
                visible_tabs=[],
                tab_width=800,
                tab_height=600
            )
    
    def save_layout(self, name: str) -> bool:
        """레이아웃 저장.
        
        Args:
            name: 레이아웃 이름
        
        Returns:
            성공 여부
        """
        try:
            layout = self.get_current_layout()
            layout.name = name
            
            # 폴더 존재 확인 및 생성
            if not self.config_dir.exists():
                try:
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error("[LayoutManager] 폴더 생성 실패: %s", e)
                    return False
            
            config_file = self.config_dir / f"{name}.json"
            
            # 절대 경로로 변환
            config_file = config_file.resolve()
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(layout), f, indent=2, ensure_ascii=False)
            
            logger.info("[LayoutManager] 레이아웃 저장 완료: %s (경로: %s)", name, config_file)
            return True
        except Exception as e:
            logger.error("[LayoutManager] 레이아웃 저장 실패: %s", e)
            return False
    
    def load_layout(self, name: str) -> bool:
        """레이아웃 로드.
        
        Args:
            name: 레이아웃 이름
        
        Returns:
            성공 여부
        """
        try:
            config_file = self.config_dir / f"{name}.json"
            
            if not config_file.exists():
                logger.warning("[LayoutManager] 레이아웃 파일 없음: %s", name)
                return False
            
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            layout = LayoutConfig(**data)
            self.current_layout = layout
            
            # 탭 순서 적용
            self._apply_tab_order(layout.tab_order)
            
            # 탭 표시/숨기기 적용
            self._apply_tab_visibility(layout.visible_tabs)
            
            # 크기 적용
            self.tab_widget.resize(layout.tab_width, layout.tab_height)
            
            logger.info("[LayoutManager] 레이아웃 로드 완료: %s", name)
            return True
        except Exception as e:
            logger.error("[LayoutManager] 레이아웃 로드 실패: %s", e)
            return False
    
    def _apply_tab_order(self, tab_order: List[str]):
        """탭 순서 적용.
        
        Args:
            tab_order: 탭 순서 리스트
        """
        try:
            # 현재 탭 매핑
            tab_mapping = {}
            for i in range(self.tab_widget.count()):
                tab_text = self.tab_widget.tabText(i)
                widget = self.tab_widget.widget(i)
                tab_mapping[tab_text] = widget
            
            # 모든 탭 제거
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
            
            # 순서대로 다시 추가
            for tab_text in tab_order:
                if tab_text in tab_mapping:
                    widget = tab_mapping[tab_text]
                    self.tab_widget.addTab(widget, tab_text)
        except Exception as e:
            logger.error("[LayoutManager] 탭 순서 적용 실패: %s", e)
    
    def _apply_tab_visibility(self, visible_tabs: List[str]):
        """탭 표시/숨기기 적용.
        
        Args:
            visible_tabs: 표시할 탭 리스트 (비어있으면 모두 표시)
        """
        try:
            for i in range(self.tab_widget.count()):
                tab_text = self.tab_widget.tabText(i)
                if visible_tabs and tab_text not in visible_tabs:
                    self.tab_widget.setTabVisible(i, False)
                else:
                    self.tab_widget.setTabVisible(i, True)
        except Exception as e:
            logger.error("[LayoutManager] 탭 표시 적용 실패: %s", e)
    
    def list_layouts(self) -> List[str]:
        """저장된 레이아웃 리스트.
        
        Returns:
            레이아웃 이름 리스트
        """
        try:
            layouts = []
            for file in self.config_dir.glob("*.json"):
                layouts.append(file.stem)
            return layouts
        except Exception as e:
            logger.error("[LayoutManager] 레이아웃 리스트 실패: %s", e)
            return []
    
    def delete_layout(self, name: str) -> bool:
        """레이아웃 삭제.
        
        Args:
            name: 레이아웃 이름
        
        Returns:
            성공 여부
        """
        try:
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
                logger.info("[LayoutManager] 레이아웃 삭제 완료: %s", name)
                return True
            return False
        except Exception as e:
            logger.error("[LayoutManager] 레이아웃 삭제 실패: %s", e)
            return False
    
    def move_tab(self, from_index: int, to_index: int):
        """탭 이동.
        
        Args:
            from_index: 이동할 탭 인덱스
            to_index: 이동할 위치 인덱스
        """
        try:
            widget = self.tab_widget.widget(from_index)
            text = self.tab_widget.tabText(from_index)
            icon = self.tab_widget.tabIcon(from_index)
            tooltip = self.tab_widget.tabToolTip(from_index)
            
            self.tab_widget.removeTab(from_index)
            self.tab_widget.insertTab(to_index, widget, text)
            
            if not icon.isNull():
                self.tab_widget.setTabIcon(to_index, icon)
            if tooltip:
                self.tab_widget.setTabToolTip(to_index, tooltip)
            
            logger.debug("[LayoutManager] 탭 이동: %d -> %d", from_index, to_index)
        except Exception as e:
            logger.error("[LayoutManager] 탭 이동 실패: %s", e)
    
    def toggle_tab_visibility(self, index: int):
        """탭 표시/숨기기 토글.
        
        Args:
            index: 탭 인덱스
        """
        try:
            current_visible = self.tab_widget.isTabVisible(index)
            self.tab_widget.setTabVisible(index, not current_visible)
            logger.debug("[LayoutManager] 탭 표시 토글: %d", index)
        except Exception as e:
            logger.error("[LayoutManager] 탭 표시 토글 실패: %s", e)
