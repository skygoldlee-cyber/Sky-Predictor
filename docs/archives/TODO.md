# TODO Issues

## GUI Controller Refactoring (gui/controller.py)

### Description
GuiController 클래스의 구조를 개선하여 코드 유지보수성을 높이는 리팩토링 작업입니다.

### Tasks
- [x] task/replay_*/effective_pred_defaults → self 멤버 (4단계 완료)
- [x] 창 기하·설정 저장 → gui_controller_window.py (5단계 완료)
- [ ] 중첩 def들을 GuiController 메서드로 단계적 승격
- [ ] _build_ui(), _setup_timers(), _bind_events() 로 init 분리

### Priority
Medium

### Notes
이 리팩토링은 GuiController 클래스의 책임을 더 명확하게 분리하고, 코드 가독성을 높이는 것을 목표로 합니다. 일부 작업은 이미 완료되었으나, 남은 작업은 점진적으로 진행할 필요가 있습니다.
