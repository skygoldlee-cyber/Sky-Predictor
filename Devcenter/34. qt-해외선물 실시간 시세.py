import sys, asyncio, json
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QTextBrowser,  # Use QTextBrowser instead of QListWidget
)
from qasync import QEventLoop, asyncSlot
import ebest
from app_keys import (
    appkey,
    appsecretkey,
)  # app_keys.py 파일에 appkey, appsecretkey 변수를 정의하고 사용하세요


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("해외선물 실시간 시세")
        self.resize(600, 500)

        # UI 구성
        self.btn_login = QPushButton("로그인")
        self.combo_items = QComboBox()
        self.btn_item_info = QPushButton("현재가 (o3105)")
        self.btn_item_info.setEnabled(False)
        self.btn_realtime_add = QPushButton("실시간 시세 시작")
        self.btn_realtime_add.setEnabled(False)
        self.btn_realtime_remove = QPushButton("실시간 시세 중지")
        self.btn_realtime_remove.setEnabled(False)
        self.btn_clear = QPushButton("지우기")
        self.text_result = QTextBrowser()  # Use QTextBrowser instead of QListWidget

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.btn_login)
        layout.addWidget(self.combo_items)
        layout.addWidget(self.btn_item_info)
        layout.addWidget(self.btn_realtime_add)
        layout.addWidget(self.btn_realtime_remove)
        layout.addWidget(self.btn_clear)
        layout.addWidget(self.text_result)  # Add QTextBrowser here

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # API 객체 생성
        self.api = ebest.OpenApi()
        self.api.on_message.connect(lambda api, msg: self.print(msg))
        self.api.on_realtime.connect(
            lambda api, trcode, key, realtimedata: self.print(
                f"{trcode}, {key}, {realtimedata}"
            )
        )

        # 버튼 이벤트 연결
        self.btn_clear.clicked.connect(self.text_result.clear)
        self.btn_login.clicked.connect(self.func_login)
        self.btn_item_info.clicked.connect(self.func_item_info)
        self.btn_realtime_add.clicked.connect(self.func_realtime_add)
        self.btn_realtime_remove.clicked.connect(self.func_realtime_remove)

    def print(self, data):
        """텍스트브라우저에 출력"""
        text = json.dumps(data, ensure_ascii=False, indent=4)  # Pretty print the data
        self.text_result.append(text)  # Use append() to add text to QTextBrowser

    @asyncSlot()
    async def func_login(self):
        """로그인"""
        api = self.api
        if not await api.login(appkey, appsecretkey):
            return self.print(f"로그인 실패: {api.last_message}")
        self.print(
            "로그인 성공: 접속서버: " + ("모의투자" if api.is_simulation else "실투자")
        )

        # 해외선물 마스터 종목 조회
        request = {"o3101InBlock": {"gubun": "0"}}
        response = await api.request("o3101", request)
        if not response:
            return self.print(f"요청실패: {api.last_message}")

        # 조회결과 ComboBox에 추가
        items = response.body["o3101OutBlock"]
        self.combo_items.clear()
        for item in items:
            self.combo_items.addItem(f"{item['Symbol']}, {item['SymbolNm']}")

        # 버튼 활성화
        self.btn_login.setEnabled(False)
        self.btn_item_info.setEnabled(True)
        self.btn_realtime_add.setEnabled(True)
        self.btn_realtime_remove.setEnabled(True)

    @asyncSlot()
    async def func_item_info(self):
        """선물 종목 정보 조회"""
        api = self.api
        symbol = self.combo_items.currentText().split(",")[0]
        request = {"o3105InBlock": {"symbol": symbol}}
        response = await self.api.request("o3105", request)
        if not response:
            return self.print(f"요청실패: {api.last_message}")
        info = response.body["o3105OutBlock"]
        self.print(info)

    @asyncSlot()
    async def func_realtime_add(self):
        """실시간 시세 요청"""
        symbol = self.combo_items.currentText().split(",")[0]
        sym8 = f"{symbol:8}"
        await self.api.add_realtime("OVC", sym8)
        self.print(f"{symbol} 실시간 시세 요청 시작")

    @asyncSlot()
    async def func_realtime_remove(self):
        """실시간 시세 중지"""
        symbol = self.combo_items.currentText().split(",")[0]
        sym8 = f"{symbol:8}"
        await self.api.remove_realtime("OVC", sym8)
        self.print(f"{symbol} 실시간 시세 요청 중지")


def main():
    """메인 함수"""
    loop = QEventLoop(QApplication(sys.argv))
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
