import sys
import asyncio
import json
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import configparser
from qasync import QEventLoop, asyncSlot
import ebest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

config = configparser.ConfigParser()
with open("config.ini", "r", encoding="utf-8") as f:
    config.read_file(f)

appkey = config.get("CREDENTIALS", "appkey")
appsecretkey = config.get("CREDENTIALS", "appsecretkey")


class AsyncRealTimeProcessor(QObject):
    kp200_signal = Signal(dict)
    kospi_signal = Signal(dict)
    kosdaq_signal = Signal(dict)
    call_signal = Signal(dict)
    put_signal = Signal(dict)
    unknown_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._task = None
        self._running = asyncio.Event()
        self._running.set()

    async def start(self):
        """Starts the background processing task."""
        self._task = asyncio.create_task(self._run())

    async def enqueue(self, data):
        """Adds new real-time data to the processing queue."""
        await self.queue.put(data)

    async def stop(self):
        """Stops the processing loop gracefully."""
        self._running.clear()
        await self.queue.put(None)  # Unblocks the queue
        if self._task:
            await self._task

    async def _run(self):
        """Main loop for processing real-time data."""
        while self._running.is_set():
            try:
                item = await self.queue.get()
                if item is None:
                    break
                api, trcode, key, data = item
                self.handle_trcode(trcode, key, data)
            except Exception as e:
                self.unknown_signal.emit(f"[Processor Error] {e}")

    def handle_trcode(self, trcode, key, data):
        """Parses and dispatches real-time data based on trcode/key."""
        try:
            if trcode == "FC0" and key.startswith("101W"):
                self.kp200_signal.emit({"trcode": trcode, "key": key, "data": data})
            elif trcode == "IJ_":
                if key.startswith("001"):
                    self.kospi_signal.emit({"trcode": trcode, "key": key, "data": data})
                elif key.startswith("301"):
                    self.kosdaq_signal.emit(
                        {"trcode": trcode, "key": key, "data": data}
                    )
                else:
                    raise ValueError(f"Unrecognized IJ_ key: {key}")
            elif trcode == "OC0":
                if key.startswith("201W"):
                    self.call_signal.emit({"trcode": trcode, "key": key, "data": data})
                elif key.startswith("301W"):
                    self.put_signal.emit({"trcode": trcode, "key": key, "data": data})
                else:
                    raise ValueError(f"Unrecognized OC0 key: {key}")
            else:
                raise ValueError(f"Unknown trcode: {trcode}, key: {key}")
        except Exception as e:
            self.unknown_signal.emit(f"[Parse Error] {e}")


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("국내 선물옵션 실시간 시세")
        self.resize(600, 500)

        self.market_data = {}  # key별 데이터 저장용 dict
        self.prev_day_data = {}  # 이전 일자 데이터 저장용

        self.today = datetime.today().strftime("%Y%m%d")
        self.target_day = config.get("SETTINGS", "TARGET_DAY", fallback=self.today)
        self.prev_target_day = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

        self.api = ebest.OpenApi()
        self.processor = AsyncRealTimeProcessor()

        self.api.on_realtime.connect(
            lambda api, trcode, key, realtimedata: asyncio.create_task(
                self.processor.enqueue((api, trcode, key, realtimedata))
            )
        )

        self.init_ui()

        # 비동기 처리기 구동 : 생성자 안에서 타이머로 호출
        QTimer.singleShot(0, self.start_async)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        top_layout = QVBoxLayout()
        self.btn_login = QPushButton("로그인")
        self.combo_items = QComboBox()
        self.btn_item_info = QPushButton("시장데이타 요청 (t8415, t8418)")
        self.btn_realtime_add = QPushButton("실시간 시세 시작")
        self.btn_realtime_remove = QPushButton("실시간 시세 중지")
        self.btn_clear = QPushButton("지우기")

        self.btn_item_info.setEnabled(False)
        self.btn_realtime_add.setEnabled(False)
        self.btn_realtime_remove.setEnabled(False)

        top_layout.addWidget(self.btn_login)
        top_layout.addWidget(self.combo_items)
        top_layout.addWidget(self.btn_item_info)
        top_layout.addWidget(self.btn_realtime_add)
        top_layout.addWidget(self.btn_realtime_remove)
        top_layout.addWidget(self.btn_clear)

        self.text_result = QTextBrowser()

        layout.addLayout(top_layout)
        layout.addWidget(self.text_result)
        central_widget.setLayout(layout)

        self.btn_clear.clicked.connect(self.text_result.clear)
        self.btn_login.clicked.connect(self.func_login)
        self.btn_item_info.clicked.connect(self.func_item_info)
        self.btn_realtime_add.clicked.connect(self.func_realtime_add)
        self.btn_realtime_remove.clicked.connect(self.func_realtime_remove)

        # 시그널 연결
        self.processor.kp200_signal.connect(self.handle_kp200)
        self.processor.kospi_signal.connect(self.handle_kospi)
        self.processor.kosdaq_signal.connect(self.handle_kosdaq)
        self.processor.call_signal.connect(self.handle_call)
        self.processor.put_signal.connect(self.handle_put)
        self.processor.unknown_signal.connect(lambda d: self.print(f"[UNKNOWN] {d}"))

    def start_async(self):
        asyncio.create_task(self.processor.start())

    def handle_kp200(self, data):
        self.handle_data("KP200", data)

    def handle_kospi(self, data):
        self.handle_data("KOSPI", data)

    def handle_kosdaq(self, data):
        self.handle_data("KOSDAQ", data)

    def handle_call(self, data):
        self.handle_data("CALL", data)

    def handle_put(self, data):
        self.handle_data("PUT", data)

    def handle_data(self, data_type: str, data: dict):
        key = data.get("key")
        if not key:
            return
        df = self.make_ohlcv(data)
        self._update_df(key, df)
        self.print(f"[{data_type.upper()}] {key} -> {data}")

    def _update_df(self, key: str, df: pd.DataFrame):
        if key in self.market_data and "df" in self.market_data[key]:
            old_df = self.market_data[key]["df"]

            # 시간 인덱스를 기준으로 병합
            combined_df = pd.concat([old_df, df])

            # 중복 시간 인덱스는 최신 데이터로 갱신 (keep='last')
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

            # 정렬 (선택사항)
            combined_df.sort_index(inplace=True)

            self.market_data[key]["df"] = combined_df
        else:
            self.market_data[key] = {
                "prev_close": None,
                "prev_high": None,
                "prev_low": None,
                "df": df,
            }

        print(
            f"key: {key}, prev_high: {self.market_data[key]['prev_high']}, prev_low: {self.market_data[key]['prev_low']}, prev_close: {self.market_data[key]['prev_close']}, df:\n{self.market_data[key]['df']}"
        )

    def make_market_dict(
        self, key: str, prev_high: float, prev_low: float, prev_close: float
    ):
        if key not in self.market_data:
            self.market_data[key] = {
                "prev_close": None,
                "prev_high": None,
                "prev_low": None,
                "df": pd.DataFrame(),
            }
        self.market_data[key]["prev_close"] = prev_close
        self.market_data[key]["prev_high"] = prev_high
        self.market_data[key]["prev_low"] = prev_low
        print(
            f"[{key}] 전일고가: {prev_high}, 전일저가: {prev_low}, 전일종가: {prev_close}\n"
        )

    async def fetch_and_process(self, upcode, date, query_type="t8415"):
        result = await self.fetch_market_async_data(
            api=self.api, query_type=query_type, upcode=upcode, date=date
        )
        if result:
            _high, _low, _close, *_ = result
            self.make_market_dict(upcode, _high, _low, _close)
        else:
            print(f"{upcode} 데이터 조회 실패")

    async def request_market_data(self):
        # 1. 전일 날짜 문자열
        date = self.today  # 예: '20250611'
        # date = self.prev_target_day

        # 주요지수는 개별 요청
        for code, query_type in [
            (self.kp200_key, "t8415"),
            ("001", "t8418"),
            ("301", "t8418"),
        ]:
            await self.fetch_and_process(code, date, query_type)
            await asyncio.sleep(1.0)

        # CALL + PUT 합쳐서 병렬 처리
        all_codes = [
            code[0]
            for code in self.filtered_call_code_list + self.filtered_put_code_list
        ]
        batch_size = 1  # 병렬 요청 수 제한

        for i in range(0, len(all_codes), batch_size):
            batch = all_codes[i : i + batch_size]
            tasks = [self.fetch_and_process(code, date) for code in batch]
            await asyncio.gather(*tasks)  # 병렬 처리
            await asyncio.sleep(1.0)  # 배치 간 sleep

    async def fetch_market_async_data(self, api, query_type, upcode, date, timeframe=1):
        if query_type not in ["t8415", "t8418"]:
            raise ValueError("query_type must be either 't8415' or 't8418'")

        if date == self.prev_target_day:
            timeframe = 60

        inputs = {
            f"{query_type}InBlock": {
                "shcode": upcode,
                "ncnt": timeframe,
                "qrycnt": 1,
                "nday": "",
                "sdate": date,
                "stime": "",
                "edate": date,
                "etime": "",
                "cts_date": "",
                "cts_time": "",
                "comp_yn": "N",
            },
        }

        response = await api.request(query_type, inputs)
        if not response:
            print(f"요청 실패: {api.last_message}")
            return None

        try:
            response_data = json.loads(
                json.dumps(response, default=lambda o: o.__dict__)
            )
        except TypeError:
            response_data = str(response)

        outblock = response_data["body"].get(f"{query_type}OutBlock", {})
        outblock1 = response_data["body"].get(f"{query_type}OutBlock1", [])
        df = pd.DataFrame(outblock1)

        # print(f"outblock: {outblock}")

        if df.empty:
            return None

        df["Datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"], format="%Y%m%d %H%M%S"
        )
        df["Datetime"] = df["Datetime"].dt.tz_localize("Asia/Seoul")
        df = df.drop(columns=["date", "time", "value"], errors="ignore")

        numeric_cols = ["open", "high", "low", "close", "jdiff_vol"]

        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle openyak safely
        if "openyak" in df.columns:
            df["openyak"] = df["openyak"].fillna(0).astype(int)
        else:
            df["openyak"] = 0

        if "openyakcha" in df.columns:
            df["openyakcha"] = df["openyakcha"].fillna(0).astype(int)
        else:
            df["openyakcha"] = 0

        # Rename
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "jdiff_vol": "Volume",
                "openyak": "OpenInterest",
                "openyakcha": "OIChange",
            }
        )

        # 선택 가능한 컬럼만 유지
        columns = [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "OpenInterest",
            "OIChange",
        ]
        df = df[[col for col in columns if col in df.columns]]

        df["Volume"] = df["Volume"].astype(int)
        df["RangePct"] = (
            (df["High"].cummax() - df["Low"].cummin()) / df["Low"].cummin()
        ).where(df["Low"].cummin() > 0, 0.0) * 100

        # Key가 없으면 초기화
        if upcode not in self.market_data:
            self.market_data[upcode] = {
                "prev_close": None,
                "prev_high": None,
                "prev_low": None,
                "df": pd.DataFrame(),
            }

        self.market_data[upcode]["df"] = df
        print(f"key: [{upcode}], df: {self.market_data[upcode]["df"]}")

        # Dispatch by date
        if date == self.today:
            return self._handle_today(df, outblock, upcode)
        else:
            if date == self.prev_target_day:
                return self._handle_prev_day(df, upcode)
            elif date == self.target_day:
                return self._handle_target_day(df, upcode)
            else:
                return None

    def _handle_today(self, df, outblock, upcode):
        print(f"_handle_today")
        required_keys = ["jihigh", "jilow", "jiclose", "disiga"]
        if not all(outblock.get(k) for k in required_keys):
            print(f"[ERROR] Pivot 계산 불가: {upcode} - 필수값 중 None이 존재합니다.")
            return (None,) * 8

        high = float(outblock["jihigh"])
        low = float(outblock["jilow"])
        close = float(outblock["jiclose"])
        open_ = float(outblock["disiga"])
        pp, r1, r2, s1 = self.my_pivot(high, low, close, open_)

        return high, low, close, pp, r1, r2, s1, df

    def _handle_prev_day(self, df, upcode):
        print(f"_handle_prev_day")
        high = df["High"].max()
        low = df["Low"].min()
        close = df["Close"].iloc[-1]

        self.prev_day_data[upcode] = {"high": high, "low": low, "close": close}
        return high, low, close, None, None, None, None, df

    def _handle_target_day(self, df, upcode):
        if (
            upcode.startswith(("001", "101W")) or upcode == "301"
        ):  # KOSPI, KP200, KOSDAQ
            high = df["High"].max()
            low = df["Low"].min()
            close = df["Close"].iloc[-1]
            open = df["Open"].iloc[0]
        elif upcode.startswith(("201W", "301W")):  # OPTION
            prev = self.prev_day_data.get(upcode)
            if not prev or None in (
                prev.get("high"),
                prev.get("low"),
                prev.get("close"),
            ):
                print(
                    f"[ERROR] Pivot 계산 불가: {upcode} - 필수값 중 None이 존재합니다."
                )
                return (None,) * 8
            high = prev["high"]
            low = prev["low"]
            close = prev["close"]
            open = df["Open"].iloc[0]
        else:
            return (None,) * 8

        pp, r1, r2, s1 = self.my_pivot(
            float(high), float(low), float(close), float(open)
        )
        return float(high), float(low), float(close), pp, r1, r2, s1, df

    @asyncSlot()
    async def func_login(self):
        api = self.api
        if not await api.login(appkey, appsecretkey):
            return self.print(f"로그인 실패: {api.last_message}")
        self.print(
            "로그인 성공: 접속서버: " + ("모의투자" if api.is_simulation else "실투자")
        )

        request = {"t8432InBlock": {"gubun": "0"}}
        response = await api.request("t8432", request)
        if not response:
            return self.print(f"요청실패: {api.last_message}")

        items = response.body["t8432OutBlock"]
        self.combo_items.clear()
        for item in items:
            self.combo_items.addItem(f"{item['shcode']}, {item['hname']}")

        self.kp200_key = items[0]["shcode"]
        print(f"kp200_key: {self.kp200_key}")

        request = {"t2301InBlock": {"yyyymm": "202507", "gubun": "G"}}
        response = await self.api.request("t2301", request)

        if not response:
            return self.print(f"요청실패: {self.api.last_message}")

        if not response:
            return print(f"요청 실패: {self.api.last_message}")

        self.call_open_list = [
            (item["optcode"], item["open"]) for item in response.body["t2301OutBlock1"]
        ]

        self.put_open_list = [
            (item["optcode"], item["open"]) for item in response.body["t2301OutBlock2"]
        ]

        self.filtered_call_code_list = [
            (key, price)
            for key, price in self.call_open_list
            if 0.1 <= float(price) <= 1.99
        ]

        self.filtered_put_code_list = [
            (key, price)
            for key, price in self.put_open_list
            if 0.1 <= float(price) <= 1.99
        ]

        print(f"filtered_call_code_list: {self.filtered_call_code_list}")
        print(f"filtered_put_code_list: {self.filtered_put_code_list}")

        # asyncio.create_task(self.request_market_data())
        await self.request_market_data()

        self.btn_login.setEnabled(False)
        self.btn_item_info.setEnabled(True)
        self.btn_realtime_add.setEnabled(True)
        self.btn_realtime_remove.setEnabled(True)

    def make_ohlcv(self, data: dict):
        d = data["data"]
        now = datetime.now()
        key = data.get("key")

        # 시간 문자열 추출
        time_str = (
            d.get("chetime")
            if key.startswith(("101W", "201W", "301W"))
            else d.get("time")
        )
        if not time_str:
            print(f"Unknown or missing time for key: {key}")
            return None

        # 시간 변환 + 1분 보정
        dt = datetime(
            now.year, now.month, now.day, int(time_str[:2]), int(time_str[2:4]), 0
        ) + timedelta(minutes=1)
        dt_ts = pd.Timestamp(dt)

        # 가격 및 누적 거래량
        price = float(d.get("price") or d.get("jisu"))
        cum_volume = int(d["volume"])

        openyak = int(d["openyak"]) if key.startswith(("101W", "201W", "301W")) else 0

        # 기존 데이터프레임 및 상태 불러오기
        df_container = self.market_data.get(key, {})
        df = df_container.get(
            "df",
            pd.DataFrame(
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "OpenInterest",
                    "OIChange",
                ]
            ),
        )
        last_cum_volume = df_container.get("last_cum_volume", cum_volume)

        # 타임존 동기화 (Datetime dtype 확인 포함)
        if not df.empty and pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
            dt_tz = df["Datetime"].dt.tz
        else:
            dt_tz = None

        if dt_tz:
            dt = (
                dt_ts.tz_localize(dt_tz)
                if dt_ts.tzinfo is None
                else dt_ts.tz_convert(dt_tz)
            )
        else:
            dt = dt_ts

        # 최근 캔들이 부족할 경우 NaN 캔들 채우기
        if len(df) < 6:
            missing = 6 - len(df)
            start_dt = dt - timedelta(minutes=missing)
            new_dates = [start_dt + timedelta(minutes=i) for i in range(missing)]
            nan_rows = pd.DataFrame(
                {
                    "Datetime": new_dates,
                    "Open": [np.nan] * missing,
                    "High": [np.nan] * missing,
                    "Low": [np.nan] * missing,
                    "Close": [np.nan] * missing,
                    "Volume": [np.nan] * missing,
                    "OpenInterest": [np.nan] * missing,
                    "OIChange": [np.nan] * missing,
                }
            )
            if not df.empty:
                for col in nan_rows.columns:
                    nan_rows[col] = nan_rows[col].astype(df[col].dtype)
            df = pd.concat([df, nan_rows], ignore_index=True)

        # 마지막 캔들과 동일 시간이라면 업데이트
        if not df.empty and pd.Timestamp(df.iloc[-1]["Datetime"]) == pd.Timestamp(dt):
            volume = max(cum_volume - last_cum_volume, 0)
            df.at[df.index[-1], "High"] = max(df.iloc[-1]["High"], price)
            df.at[df.index[-1], "Low"] = min(df.iloc[-1]["Low"], price)
            df.at[df.index[-1], "Close"] = price

            if pd.isna(df.iloc[-1]["Open"]):
                df.at[df.index[-1], "Open"] = price  # 해당 분의 첫 체결값

            df.at[df.index[-1], "Volume"] = volume

        else:
            # 새 캔들 생성
            volume = max(cum_volume - last_cum_volume, 0)
            new_row = pd.DataFrame(
                [
                    {
                        "Datetime": dt,
                        "Open": price,
                        "High": price,
                        "Low": price,
                        "Close": price,
                        "Volume": volume,
                        "OpenInterest": openyak,
                        # "OIChange": np.nan,  # 임시값, 아래에서 diff로 다시 계산됨
                    }
                ]
            )
            if not df.empty:
                for col in new_row.columns:
                    new_row[col] = new_row[col].astype(df[col].dtype)
            df = pd.concat([df, new_row], ignore_index=True)
            last_cum_volume = cum_volume

        # OIChange 계산: OpenInterest 차이
        df["OIChange"] = df["OpenInterest"].astype("Int64").diff()

        # 정렬 및 저장
        df.sort_values("Datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.market_data[key].update(
            {
                "last_cum_volume": last_cum_volume,
                "df": df.copy(),
            }
        )

        return df.copy()

    def make_ohlcv_new(self, data: dict):
        d = data["data"]
        now = datetime.now()
        key = data.get("key")

        # 시간 문자열 추출
        time_str = (
            d.get("chetime")
            if key.startswith(("101W", "201W", "301W"))
            else d.get("time")
        )
        if not time_str:
            print(f"Unknown or missing time for key: {key}")
            return None

        # 시간 변환 + 1분 보정
        dt = datetime(
            now.year, now.month, now.day, int(time_str[:2]), int(time_str[2:4]), 0
        ) + timedelta(minutes=1)
        dt_ts = pd.Timestamp(dt)

        # 가격 및 누적 거래량
        price = float(d.get("price") or d.get("jisu"))
        cum_volume = int(d["volume"])
        openyak = (
            int(d["openyak"])
            if d.get("openyak") and key.startswith(("101W", "201W", "301W"))
            else np.nan
        )

        # 데이터프레임 컨테이너 초기화
        if key not in self.market_data:
            self.market_data[key] = {}
        df_container = self.market_data[key]

        df = df_container.get(
            "df",
            pd.DataFrame(
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "OpenInterest",
                    "OIChange",
                ]
            ),
        )
        last_cum_volume = df_container.get("last_cum_volume", cum_volume)

        # 타임존 동기화
        dt_tz = (
            df["Datetime"].dt.tz
            if not df.empty and pd.api.types.is_datetime64_any_dtype(df["Datetime"])
            else None
        )
        dt = (
            dt_ts.tz_localize(dt_tz)
            if dt_tz and dt_ts.tzinfo is None
            else (dt_ts.tz_convert(dt_tz) if dt_tz else dt_ts)
        )

        # 초기 결측 캔들 채우기
        if len(df) < 6:
            missing = 6 - len(df)
            start_dt = dt - timedelta(minutes=missing)
            new_dates = [start_dt + timedelta(minutes=i) for i in range(missing)]
            nan_rows = pd.DataFrame(
                {
                    "Datetime": new_dates,
                    "Open": [np.nan] * missing,
                    "High": [np.nan] * missing,
                    "Low": [np.nan] * missing,
                    "Close": [np.nan] * missing,
                    "Volume": [np.nan] * missing,
                    "OpenInterest": [np.nan] * missing,
                    "OIChange": [np.nan] * missing,
                }
            )
            if not df.empty:
                for col in nan_rows.columns:
                    nan_rows[col] = nan_rows[col].astype(df[col].dtype)
            df = pd.concat([df, nan_rows], ignore_index=True)

        # 캔들 업데이트 or 새로 생성
        if not df.empty and df["Datetime"].iloc[-1] == dt:
            volume = max(cum_volume - last_cum_volume, 0)
            df.at[df.index[-1], "High"] = max(df.iloc[-1]["High"], price)
            df.at[df.index[-1], "Low"] = min(df.iloc[-1]["Low"], price)
            df.at[df.index[-1], "Close"] = price
            if pd.isna(df.iloc[-1]["Open"]):
                df.at[df.index[-1], "Open"] = price
            df.at[df.index[-1], "Volume"] = volume
            df.at[df.index[-1], "OpenInterest"] = openyak
        else:
            volume = max(cum_volume - last_cum_volume, 0)
            new_row = pd.DataFrame(
                [
                    {
                        "Datetime": dt,
                        "Open": price,
                        "High": price,
                        "Low": price,
                        "Close": price,
                        "Volume": volume,
                        "OpenInterest": openyak,
                    }
                ]
            )
            if not df.empty:
                for col in new_row.columns:
                    new_row[col] = new_row[col].astype(df[col].dtype)
            df = pd.concat([df, new_row], ignore_index=True)
            last_cum_volume = cum_volume

        # OIChange 계산
        df["OIChange"] = df["OpenInterest"].astype("Int64").diff()

        # 정렬 및 인덱스 정리
        df.sort_values("Datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 저장
        self.market_data[key].update(
            {
                "last_cum_volume": last_cum_volume,
                "df": df.copy(),
            }
        )

        return df.copy()

    def print(self, data):
        text = json.dumps(data, ensure_ascii=False, indent=4)
        self.text_result.append(text)

    def my_pivot(self, prev_high, prev_low, prev_close, today_open):
        base_pp = (prev_high + prev_low + prev_close) / 3
        gap = today_open - prev_close
        PP = base_pp + gap

        range_ = (prev_high - prev_low) / 2
        R1 = PP + (range_ * 1)
        R2 = PP + (range_ * 2)
        S1 = PP - range_

        return round(PP, 2), round(R1, 2), round(R2, 2), round(S1, 2)

    @asyncSlot()
    async def func_item_info(self):
        # asyncio.create_task(self.request_market_data())
        pass

    @asyncSlot()
    async def func_realtime_add(self):
        tasks = []

        symbol = self.combo_items.currentText().split(",")[0]
        tasks.append(self.api.add_realtime("FC0", symbol))
        self.print(f"{symbol} 실시간 시세를 요청합니다.")

        tasks.append(self.api.add_realtime("IJ_", "001"))
        self.print("KOSPI 실시간 시세를 요청합니다.")

        tasks.append(self.api.add_realtime("IJ_", "301"))
        self.print("KOSDAQ 실시간 시세를 요청합니다.")

        for call_code in self.filtered_call_code_list:
            tasks.append(self.api.add_realtime("OC0", call_code[0]))
            self.print(f"Call Key: {call_code[0]} 실시간 시세를 요청합니다.")

        for put_code in self.filtered_put_code_list:
            tasks.append(self.api.add_realtime("OC0", put_code[0]))
            self.print(f"Put Key: {put_code[0]} 실시간 시세를 요청합니다.")

        await asyncio.gather(*tasks)

    @asyncSlot()
    async def func_realtime_remove(self):
        symbol = self.combo_items.currentText().split(",")[0]
        await self.api.remove_realtime("FC0", symbol)
        self.print(f"{symbol} 실시간시세 요청 중지")

    def closeEvent(self, event):
        # Schedule processor shutdown but don't block the UI
        asyncio.create_task(self.processor.stop())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MyApp()
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
