# eBest OpenAPI — 선물/옵션 TR 스키마 정리

> 선물·옵션 현재가 조회(REST), 실시간 체결(WebSocket), 실시간 호가(WebSocket), 차트 데이터, 옵션 일람 관련 TR 스키마를 정리한 문서입니다.

---

## 목차

1. [t8467 — 지수선물 마스터조회 API용](#t8467--지수선물-마스터조회-api용)
2. [t8433 — 지수옵션 마스터조회 API용](#t8433--지수옵션-마스터조회-api용)
3. [t2111 — 선물/옵션 현재가 조회](#t2111--선물옵션-현재가-조회)
4. [t2301 — 옵션전광판](#t2301--옵션-일람-조회)
5. [t8465 — 선물/옵션 차트 (N분)](#t8465--선물옵션-차트-n분)
6. [t8418 — 업종 차트 (N분)](#t8418--업종-차트-n분)
7. [t8466 — 선물/옵션 차트 (일주월)](#t8466--선물옵션-차트-일주월)
8. [IJ / IJ_ — 지수 종합주가 (REST/Push)](#ij--지수-종합주가-rest)
9. [JIF — 실시간 장운영 정보 (WebSocket)](#jif--실시간-장운영-정보-websocket)
10. [FC9 / OC0 — 실시간 체결 (WebSocket)](#fc9--oc0--실시간-체결-websocket)
11. [FH9 / OH0 — 실시간 호가 (WebSocket)](#fh9--oh0--실시간-호가-websocket)
12. [실시간 Tick 표준화(`tick_norm`)](#실시간-tick-표준화tick_norm)

---

## ■ 변경대상 선물옵션 TR 리스트

| 기존TR | 신규TR | TR명 |
|--------|--------|------|
| t2101 | t2111 | 선물/옵션 현재가(시세) 조회 |
| t2105 | t2112 | 선물/옵션 현재가 호가조회 |
| t2201 | t2212 | 선물/옵션 시간대별 체결조회 |
| t2203 | t2214 | 선물/옵션 기간별 주가 |
| t2209 | t2216 | 선물/옵션 틱분별 체결조회 차트 |
| t2405 | t2407 | 선물/옵션 호가잔량 비율 차트 |
| t2421 | t2424 | 선물/옵션 미결제약정 추이 |
| t8414 | t8464 | 선물옵션차트(틱/n틱) |
| t8415 | t8465 | 선물/옵션챠트(N분) |
| t8416 | t8466 | 선물/옵션챠트(일주월) |
| t8432 | t8467 | 지수선물마스터조회API용 |
| FC0 | FC9 | KOSPI200선물체결 |
| FH0 | FH9 | KOSPI200선물호가 |
| FX0 | FX9 | KOSPI200선물가격제한폭확대 |
| YFC | YF9 | 지수선물예상체결 |

---

## t8467 — 지수선물 마스터조회 API용

**방식:** REST (Request / Response)

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `gubun` | 구분 | str | 1 | |

### OutBlock (응답)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `hname` | 종목명 | str | 20 | |
| `shcode` | 단축코드 | str | 8 | |
| `expcode` | 확장코드 | str | 12 | |
| `uplmtprice` | 상한가 | number | 6.2 | |
| `dnlmtprice` | 하한가 | number | 6.2 | |
| `jnilclose` | 전일종가 | number | 6.2 | |
| `jnilhigh` | 전일고가 | number | 6.2 | |
| `jnillow` | 전일저가 | number | 6.2 | |
| `recprice` | 기준가 | number | 6.2 | |

---

## t8433 — 지수옵션 마스터조회 API용

**방식:** REST (Request / Response)

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `dummy` | Dummy | str | 1 | |

### OutBlock (응답)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `hname` | 종목명 | str | 20 | |
| `shcode` | 단축코드 | str | 8 | |
| `expcode` | 확장코드 | str | 12 | |
| `hprice` | 상한가 | number | 6.2 | |
| `lprice` | 하한가 | number | 6.2 | |
| `jnilclose` | 전일종가 | number | 6.2 | |
| `jnilhigh` | 전일고가 | number | 6.2 | |
| `jnillow` | 전일저가 | number | 6.2 | |
| `recprice` | 기준가 | number | 6.2 | |

---

## t2111 — 선물/옵션 현재가 조회

**방식:** REST (Request / Response)  
**설명:** 선물·옵션 종목의 현재가, 호가, 그리스(Greeks) 등 상세 정보를 조회합니다.

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `focode` | 단축코드 | str | 8 | |

### OutBlock (응답)

#### 가격 정보

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `hname` | 한글명 | str | 20 | |
| `price` | 현재가 | number | 6.2 | |
| `sign` | 전일대비구분 | str | 1 | 1:상한 2:상승 3:보합 4:하한 5:하락 |
| `change` | 전일대비 | number | 6.2 | |
| `diff` | 등락율 | number | 6.2 | |
| `jnilclose` | 전일종가 | number | 6.2 | |
| `open` | 시가 | number | 6.2 | |
| `high` | 고가 | number | 6.2 | |
| `low` | 저가 | number | 6.2 | |
| `volume` | 거래량 | number | 12 | |
| `value` | 거래대금 | number | 12 | |

#### 미결제 / 호가

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `mgjv` | 미결제량 | number | 8 | |
| `mgjvdiff` | 미결제증감 | number | 8 | |

#### 가격 제한

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `uplmtprice` | 상한가 | number | 6.2 | |
| `dnlmtprice` | 하한가 | number | 6.2 | |
| `cbhprice` | CB상한가 | number | 6.2 | |
| `cblprice` | CB하한가 | number | 6.2 | |
| `dy_gubun` | 실시간가격제한여부 | str | 1 | 0:대상아님 1:적용중 2:미적용 3:일시해제 |
| `dy_uplmtprice` | 실시간상한가 | number | 6.2 | |
| `dy_dnlmtprice` | 실시간하한가 | number | 6.2 | |
| `updnstep_gubun` | 가격제한폭확대 | str | 1 | 0:미확대 1:확대 2:대상아님 |
| `upstep` | 상한적용단계 | str | 2 | |
| `dnstep` | 하한적용단계 | str | 2 | |
| `uplmtprice_3rd` | 3단계상한가 | number | 6.2 | |
| `dnlmtprice_3rd` | 3단계하한가 | number | 6.2 | |

#### 52주 / 상장 최고·최저

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `high52w` | 52주최고가 | number | 6.2 |
| `low52w` | 52주최저가 | number | 6.2 |
| `listhprice` | 상장최고가 | number | 6.2 |
| `listlprice` | 상장최저가 | number | 6.2 |

#### 이론가 / 베이시스

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `recprice` | 기준가 | number | 6.2 |
| `theoryprice` | 이론가 | number | 6.2 |
| `theorypriceg` | 이론가(근월물) | number | 6.2 |
| `glyl` | 괴리율 | number | 6.3 |
| `sbasis` | 시장BASIS | number | 6.2 |
| `ibasis` | 이론BASIS | number | 6.2 |
| `basis` | 베이시스 | number | 6.2 |
| `histimpv` | 역사적변동성 | number | 6.2 |
| `impv` | 내재변동성 | number | 6.2 |

#### 지수 정보

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `pricejisu` | 종합지수 | number | 6.2 | |
| `jisusign` | 종합지수전일대비구분 | str | 1 | 1:상한 2:상승 3:보합 4:하한 5:하락 |
| `jisuchange` | 종합지수전일대비 | number | 6.2 | |
| `jisudiff` | 종합지수등락율 | number | 6.2 | |
| `kospijisu` | KOSPI200지수 | number | 6.2 | |
| `kospisign` | KOSPI200전일대비구분 | str | 1 | 1:상한 2:상승 3:보합 4:하한 5:하락 |
| `kospichange` | KOSPI200전일대비 | number | 6.2 | |
| `kospidiff` | KOSPI200등락율 | number | 6.2 | |

#### 근월물

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `gmprice` | 근월물현재가 | number | 6.2 |
| `gmsign` | 근월물전일대비구분 | str | 1 |
| `gmchange` | 근월물전일대비 | number | 6.2 |
| `gmdiff` | 근월물등락율 | number | 6.2 |
| `gmfutcode` | 근월물종목코드 | str | 8 |

#### 그리스 (Greeks)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `delt` | 델타 | number | 6.4 |
| `gama` | 감마 | number | 6.4 |
| `ceta` | 세타 | number | 6.4 |
| `vega` | 베가 | number | 6.4 |
| `rhox` | 로우 | number | 6.4 |
| `greeks_time` | 거래소민감도수신시간 | str | 6 |
| `greeks_confirm` | 거래소민감도확정여부 | str | 8 |

#### 만기 / 기타

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `lastmonth` | 만기일 | str | 8 | |
| `jandatecnt` | 잔여일 | number | 8 | |
| `bjandatecnt` | 잔여일(영업일) | number | 8 | |
| `actprice` | 행사가 | number | 6.2 | |
| `danhochk` | 단일가호가여부 | str | 1 | |
| `alloc_gubun` | 배분구분 | str | 1 | 1:배분개시 2:배분해제 0:미발생 |
| `yeprice` | 예상체결가 | number | 6.2 | |
| `jnilysign` | 예상체결가전일대비구분 | str | 1 | |
| `jnilychange` | 예상체결가전일대비 | number | 6.2 | |
| `jnilydrate` | 예상체결가등락율 | number | 6.2 | |
| `expct_ccls_q` | 예상체결수량 | number | 9 | |
| `focode` | 종목코드 | str | 8 | |

---

## t2301 — 옵션전광판

**방식:** REST (Request / Response)  
**설명:** 특정 만기월의 콜·풋 옵션 전 행사가를 한 번에 조회합니다.

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `yyyymm` | 월물 | str | 6 | 예) 미니·정규: `200604` / 위클리: `W1    ` |
| `gubun` | 미니구분 | str | 1 | M:미니 G:정규 W:위클리 |

### OutBlock (응답 — 요약)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `histimpv` | 역사적변동성 | number | 4 | |
| `jandatecnt` | 옵션잔존일 | number | 4 | |
| `cimpv` | 콜옵션대표IV | number | 6.3 | |
| `pimpv` | 풋옵션대표IV | number | 6.3 | |
| `gmprice` | 근월물현재가 | number | 6.2 | |
| `gmsign` | 근월물전일대비구분 | str | 1 | 1:상한 2:상승 3:보합 4:하한 5:하락 |
| `gmchange` | 근월물전일대비 | number | 6.2 | |
| `gmdiff` | 근월물등락율 | number | 6.2 | |
| `gmvolume` | 근월물거래량 | number | 12 | |
| `gmshcode` | 근월물선물코드 | str | 8 | |

### OutBlock1 (응답 — 콜옵션 배열)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `actprice` | 행사가 | number | 6.2 | |
| `optcode` | 콜옵션코드 | str | 8 | |
| `price` | 현재가 | number | 6.2 | |
| `sign` | 전일대비구분 | str | 1 | 1:상한 2:상승 3:보합 4:하한 5:하락 |
| `change` | 전일대비 | number | 6.2 | |
| `diff` | 등락율 | number | 6.2 | |
| `volume` | 거래량 | number | 12 | |
| `iv` | IV | number | 6.2 | |
| `mgjv` | 미결제약정 | number | 12 | |
| `mgjvupdn` | 미결제약정증감 | number | 12 | |
| `offerho1` | 매도호가 | number | 6.2 | |
| `bidho1` | 매수호가 | number | 6.2 | |
| `cvolume` | 체결량 | number | 12 | |
| `delt` | 델타 | number | 6.4 | |
| `gama` | 감마 | number | 6.4 | |
| `vega` | 베가 | number | 6.4 | |
| `ceta` | 세타 | number | 6.4 | |
| `rhox` | 로우 | number | 6.4 | |
| `theoryprice` | 이론가 | number | 6.2 | |
| `impv` | 내재가치 | number | 6.2 | |
| `timevl` | 시간가치 | number | 6.2 | |
| `jvolume` | 잔고수량 | number | 12 | |
| `parpl` | 평가손익 | number | 12 | |
| `jngo` | 청산가능수량 | number | 6 | |
| `offerrem1` | 매도잔량 | number | 12 | |
| `bidrem1` | 매수잔량 | number | 12 | |
| `open` | 시가 | number | 6.2 | |
| `high` | 고가 | number | 6.2 | |
| `low` | 저가 | number | 6.2 | |
| `atmgubun` | ATM구분 | str | 1 | 0:선물 1:ATM 2:ITM 3:OTM |
| `jisuconv` | 지수환산 | number | 6.2 | |
| `value` | 거래대금 | number | 12 | |

### OutBlock2 (응답 — 풋옵션 배열)

OutBlock1과 동일한 필드 구조 (`optcode`는 풋옵션코드).

---

## t8465 — 선물/옵션 차트 (N분)

**방식:** REST (Request / Response)
**설명:** 선물·옵션 종목의 N분봉 OHLCV 데이터를 조회합니다. (구 t8415 폐지, 신규 t8465 사용)

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `shcode` | 단축코드 | str | 8 | |
| `ncnt` | 단위(n분) | number | 4 | 1:1분 2:2분 … n:n분 |
| `qrycnt` | 요청건수 | number | 4 | 최대 500건 |
| `nday` | 조회영업일수 사용여부 | str | 1 | 0:미사용 1이상:사용 |
| `sdate` | 시작일자 | str | 8 | 기본값: Space (edate 기준으로 qrycnt만큼 조회) |
| `stime` | 시작시간 | str | 6 | |
| `edate` | 종료일자 | str | 8 | 처음조회기준일(LE). `99999999` 또는 `당일` 입력 가능 |
| `etime` | 종료시간 | str | 6 | |
| `cts_date` | 연속일자 | str | 8 | 처음 조회 시 Space, 연속 조회 시 이전 OutBlock의 `cts_date` 값 |
| `cts_time` | 연속시간 | str | 6 | 분봉은 시각까지 필요 |
| `comp_yn` | 압축여부 | str | 1 | Y:압축 N:비압축 |

### OutBlock (응답 — 헤더)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `shcode` | 단축코드 | str | 8 |
| `jisiga` | 전일시가 | number | 6.2 |
| `jihigh` | 전일고가 | number | 6.2 |
| `jilow` | 전일저가 | number | 6.2 |
| `jiclose` | 전일종가 | number | 6.2 |
| `jivolume` | 전일거래량 | number | 12 |
| `disiga` | 당일시가 | number | 6.2 |
| `dihigh` | 당일고가 | number | 6.2 |
| `dilow` | 당일저가 | number | 6.2 |
| `diclose` | 당일종가 | number | 6.2 |
| `highend` | 상한가 | number | 6.2 |
| `lowend` | 하한가 | number | 6.2 |
| `cts_date` | 연속일자 | str | 8 |
| `cts_time` | 연속시간 | str | 6 |
| `s_time` | 장시작시간 (HHMMSS) | str | 6 |
| `e_time` | 장종료시간 (HHMMSS) | str | 6 |
| `dshmin` | 동시호가처리시간 (MM:분) | str | 2 |
| `rec_count` | 레코드카운트 | number | 7 |

### OutBlock1 (응답 — 봉 배열)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `date` | 날짜 | str | 8 |
| `time` | 시간 | str | 6 |
| `open` | 시가 | number | 6.2 |
| `high` | 고가 | number | 6.2 |
| `low` | 저가 | number | 6.2 |
| `close` | 종가 | number | 6.2 |
| `jdiff_vol` | 누적거래량 | number | 12 |
| `value` | 거래대금 | number | 12 |
| `openyak` | 미결제약정 | number | 12 |

---

## t8466 — 선물/옵션 차트 (일주월)

**방식:** REST (Request / Response)
**설명:** 선물·옵션 종목의 일봉/주봉/월봉 OHLCV 데이터를 조회합니다. (구 t8416 폐지, 신규 t8466 사용)

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `shcode` | 단축코드 | str | 8 | |
| `ncnt` | 주기구분 | str | 1 | 1:일봉 2:주봉 3:월봉 |
| `qrycnt` | 요청건수 | number | 4 | 최대 500건 |
| `nday` | 조회영업일수 사용여부 | str | 1 | 0:미사용 1이상:사용 |
| `sdate` | 시작일자 | str | 8 | 기본값: Space (edate 기준으로 qrycnt만큼 조회) |
| `edate` | 종료일자 | str | 8 | 처음조회기준일(LE). `99999999` 또는 `당일` 입력 가능 |
| `cts_date` | 연속일자 | str | 8 | 처음 조회 시 Space, 연속 조회 시 이전 OutBlock의 `cts_date` 값 |
| `comp_yn` | 압축여부 | str | 1 | Y:압축 N:비압축 |

### OutBlock (응답 — 헤더)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `shcode` | 단축코드 | str | 8 |
| `jisiga` | 전일시가 | number | 6.2 |
| `jihigh` | 전일고가 | number | 6.2 |
| `jilow` | 전일저가 | number | 6.2 |
| `jiclose` | 전일종가 | number | 6.2 |
| `jivolume` | 전일거래량 | number | 12 |
| `disiga` | 당일시가 | number | 6.2 |
| `dihigh` | 당일고가 | number | 6.2 |
| `dilow` | 당일저가 | number | 6.2 |
| `diclose` | 당일종가 | number | 6.2 |
| `highend` | 상한가 | number | 6.2 |
| `lowend` | 하한가 | number | 6.2 |
| `cts_date` | 연속일자 | str | 8 |
| `s_time` | 장시작시간 (HHMMSS) | str | 6 |
| `e_time` | 장종료시간 (HHMMSS) | str | 6 |
| `dshmin` | 동시호가처리시간 (MM:분) | str | 2 |
| `rec_count` | 레코드카운트 | number | 7 |

### OutBlock1 (응답 — 봉 배열)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `date` | 날짜 | str | 8 |
| `open` | 시가 | number | 6.2 |
| `high` | 고가 | number | 6.2 |
| `low` | 저가 | number | 6.2 |
| `close` | 종가 | number | 6.2 |
| `jdiff_vol` | 누적거래량 | number | 12 |
| `value` | 거래대금 | number | 12 |
| `openyak` | 미결제약정 | number | 12 |

---

## t8418 — 업종 차트 (N분)

**방식:** REST (Request / Response)
**설명:** 업종 종목의 N분봉 OHLCV 데이터를 조회합니다.

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `shcode` | 단축코드 | str | 8 | |
| `ncnt` | 단위(n분) | number | 4 | 1:1분 2:2분 … n:n분 |
| `qrycnt` | 요청건수 | number | 4 | 최대 2000건(압축), 500건(비압축) |
| `nday` | 조회영업일수 사용여부 | str | 1 | 0:미사용 1이상:사용 |
| `sdate` | 시작일자 | str | 8 | 기본값: Space (edate 기준으로 qrycnt만큼 조회) |
| `stime` | 시작시간 | str | 6 | 현재 미사용 |
| `edate` | 종료일자 | str | 8 | 처음조회기준일(LE). `99999999` 또는 `당일` 입력 가능 |
| `etime` | 종료시간 | str | 6 | 현재 미사용 |
| `cts_date` | 연속일자 | str | 8 | 처음 조회 시 Space, 연속 조회 시 이전 OutBlock의 `cts_date` 값 |
| `cts_time` | 연속시간 | str | 6 | 분봉은 시각까지 필요 |
| `comp_yn` | 압축여부 | str | 1 | Y:압축 N:비압축 |

### OutBlock (응답 — 헤더)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `shcode` | 단축코드 | str | 8 |
| `jisiga` | 전일시가 | number | 6.2 |
| `jihigh` | 전일고가 | number | 6.2 |
| `jilow` | 전일저가 | number | 6.2 |
| `jiclose` | 전일종가 | number | 6.2 |
| `jivolume` | 전일거래량 | number | 12 |
| `disiga` | 당일시가 | number | 6.2 |
| `dihigh` | 당일고가 | number | 6.2 |
| `dilow` | 당일저가 | number | 6.2 |
| `diclose` | 당일종가 | number | 6.2 |
| `disvalue` | 당일거래대금 | number | 12 |
| `cts_date` | 연속일자 | str | 8 |
| `cts_time` | 연속시간 | str | 6 |
| `s_time` | 장시작시간 (HHMMSS) | str | 6 |
| `e_time` | 장종료시간 (HHMMSS) | str | 6 |
| `dshmin` | 동시호가처리시간 (MM:분) | str | 2 |
| `rec_count` | 레코드카운트 | number | 7 |

### OutBlock1 (응답 — 봉 배열)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `date` | 날짜 | str | 8 |
| `time` | 시간 | str | 6 |
| `open` | 시가 | number | 6.2 |
| `high` | 고가 | number | 6.2 |
| `low` | 저가 | number | 6.2 |
| `close` | 종가 | number | 6.2 |
| `jdiff_vol` | 누적거래량 | number | 12 |
| `value` | 거래대금 | number | 12 |

---

## IJ — 지수 종합주가 (REST)

**방식:** REST (Request / Response)

### InBlock (요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `tr_cd` | 거래 CD | str | 3 | |
| `tr_key` | 단축코드 | str | 8 | |

### OutBlock (응답)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `time` | 시간 | str | - | |
| `jisu` | 지수 | number | - | |
| `sign` | 전일대비구분 | str | - | |
| `change` | 전일비 | number | - | |
| `drate` | 등락율 | number | - | |
| `cvolume` | 체결량 | number | - | |
| `volume` | 거래량 | number | - | |
| `value` | 거래대금 | number | - | |
| `upjo` | 상한종목수 | number | - | |
| `highjo` | 상승종목수 | number | - | |
| `unchgjo` | 보합종목수 | number | - | |
| `lowjo` | 하락종목수 | number | - | |
| `downjo` | 하한종목수 | number | - | |
| `upjrate` | 상승종목비율 | number | - | |
| `openjisu` | 시가지수 | number | - | |
| `opentime` | 시가시간 | str | - | |
| `highjisu` | 고가지수 | number | - | |
| `hightime` | 고가시간 | str | - | |
| `lowjisu` | 저가지수 | number | - | |
| `lowtime` | 저가시간 | str | - | |
| `frgsvolume` | 외인순매수수량 | number | - | |
| `orgsvolume` | 기관순매수수량 | number | - | |
| `frgsvalue` | 외인순매수금액 | number | - | |
| `orgsvalue` | 기관순매수금액 | number | - | |
| `upcode` | 업종코드 | str | - | |

---

## IJ_ — 지수 종합주가 (WebSocket)

**방식:** WebSocket (Subscribe / Push)

### InBlock (구독 요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `tr_cd` | 거래 CD | str | 3 | |
| `tr_key` | 단축코드 | str | 8 | KOSPI: `001`, KOSDAQ: `301`, KOSPI200: `101` |

### OutBlock (실시간 Push)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `time` | 시간 | str | - | |
| `jisu` | 지수 | number | - | |
| `sign` | 전일대비구분 | str | - | |
| `change` | 전일비 | number | - | |
| `drate` | 등락율 | number | - | |
| `cvolume` | 체결량 | number | - | |
| `volume` | 거래량 | number | - | |
| `value` | 거래대금 | number | - | |
| `upjo` | 상한종목수 | number | - | |
| `highjo` | 상승종목수 | number | - | |
| `unchgjo` | 보합종목수 | number | - | |
| `lowjo` | 하락종목수 | number | - | |
| `downjo` | 하한종목수 | number | - | |
| `upjrate` | 상승종목비율 | number | - | |
| `openjisu` | 시가지수 | number | - | |
| `opentime` | 시가시간 | str | - | |
| `highjisu` | 고가지수 | number | - | |
| `hightime` | 고가시간 | str | - | |
| `lowjisu` | 저가지수 | number | - | |
| `lowtime` | 저가시간 | str | - | |
| `frgsvolume` | 외인순매수수량 | number | - | |
| `orgsvolume` | 기관순매수수량 | number | - | |
| `frgsvalue` | 외인순매수금액 | number | - | |
| `orgsvalue` | 기관순매수금액 | number | - | |
| `upcode` | 업종코드 | str | - | |

---

## JIF — 실시간 장운영 정보 (WebSocket)

**방식:** WebSocket (Subscribe / Push)

### InBlock (구독 요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `tr_cd` | 거래 CD | str | 3 | LS증권 거래코드 |
| `tr_key` | 단축코드 | str | 8 | 단축코드 6자리 또는 8자리 (단건, 연속)<br>(계좌등록/해제일 경우 필수값 아님) |

### OutBlock (실시간 Push)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `jangubun` | 장구분 | str | 1 | 코드표 참조 |
| `jstatus` | 장상태 | str | 2 | 코드표 참조 |

#### 코드표 — `jangubun` (장구분)

| 코드 | 의미 |
|------|------|
| `1` | 코스피 |
| `2` | 코스닥 |
| `5` | 선물/옵션 |
| `6` | NXT전용 |
| `8` | KRX야간파생 |
| `9` | 미국주식 |
| `A` | 중국주식오전 |
| `B` | 중국주식오후 |
| `C` | 홍콩주식오전 |
| `D` | 홍콩주식오후 |
| `E` | 일본주식오전 |
| `F` | 일본주식오후 |

#### 코드표 — `jstatus` (장상태)

| 코드 | 의미 | 적용 |
|------|------|------|
| `11` | 장전동시호가개시 | 공통 |
| `21` | 장시작 | 공통 |
| `22` | 장개시10초전 | 공통 |
| `23` | 장개시1분전 | 공통 |
| `24` | 장개시5분전 | 공통 |
| `25` | 장개시10분전 | 공통 |
| `31` | 장후동시호가개시 | 공통 |
| `41` | 장마감 | 공통 |
| `42` | 장마감10초전 | 공통 |
| `43` | 장마감1분전 | 공통 |
| `44` | 장마감5분전 | 공통 |
| `51` | 시간외종가매매개시 | 공통 |
| `52` | 시간외종가매매종료,시간외단일가매매개시 | 공통 |
| `53` | 사용안함 | 공통 |
| `54` | 시간외단일가매매종료 | 공통 |
| `55` | 프리마켓 개시 | 공통 |
| `A2` | 프리마켓 장개시,10초전 | 공통 |
| `A3` | 프리마켓 장개시,1분전 | 공통 |
| `A4` | 프리마켓 장개시,5분전 | 공통 |
| `A5` | 프리마켓 장개시,10분전 | 공통 |
| `56` | 에프터마켓 개시 | 공통 |
| `B2` | 에프터마켓 장개시,10초전 | 공통 |
| `B3` | 에프터마켓 장개시,1분전 | 공통 |
| `B4` | 에프터마켓 장개시,5분전 | 공통 |
| `B5` | 에프터마켓 장개시,10분전 | 공통 |
| `57` | 프리마켓 마감 | 공통 |
| `C2` | 프리마켓 장마감,10초전 | 공통 |
| `C3` | 프리마켓 장마감,1분전 | 공통 |
| `C4` | 프리마켓 장마감,5분전 | 공통 |
| `58` | 에프터마켓 마감 | 공통 |
| `D2` | 에프터마켓 장마감,10초전 | 공통 |
| `D3` | 에프터마켓 장마감,1분전 | 공통 |
| `D4` | 에프터마켓 장마감,5분전 | 공통 |
| `61` | 서킷브레이크1단계발동 | KOSPI/KOSDAQ(jangubun=1,2) 또는 선물/옵션(jangubun=5) |
| `62` | 서킷브레이크1단계해제,호가접수개시 | KOSPI/KOSDAQ(jangubun=1,2) 또는 선물/옵션(jangubun=5) |
| `63` | 서킷브레이크1단계,동시호가종료 | KOSPI/KOSDAQ(jangubun=1,2) 또는 선물/옵션(jangubun=5) |
| `64` | 사이드카 매도발동 | KOSPI/KOSDAQ(jangubun=1,2) |
| `65` | 사이드카 매도해제 | KOSPI/KOSDAQ(jangubun=1,2) |
| `66` | 사이드카 매수발동 | KOSPI/KOSDAQ(jangubun=1,2) |
| `67` | 사이드카 매수해제 | KOSPI/KOSDAQ(jangubun=1,2) |
| `68` | 서킷브레이크2단계발동 | KOSPI/KOSDAQ(jangubun=1,2) |
| `69` | 서킷브레이크3단계발동,당일 장종료 | KOSPI/KOSDAQ(jangubun=1,2) |
| `70` | 서킷브레이크2단계해제,호가접수개시 | KOSPI/KOSDAQ(jangubun=1,2) |
| `71` | 서킷브레이크2단계,동시호가종료 | KOSPI/KOSDAQ(jangubun=1,2) |
| `70` | 2단계상한가,5분 후 확대 예정 | 선물/옵션(jangubun=5) |
| `71` | 2단계하한가,5분 후 확대 예정 | 선물/옵션(jangubun=5) |
| `72` | 3단계상한가,5분 후 확대 예정 | 선물/옵션(jangubun=5) |
| `73` | 3단계하한가,5분 후 확대 예정 | 선물/옵션(jangubun=5) |
| `74` | 2단계상한가,확대 적용 | 선물/옵션(jangubun=5) |
| `75` | 2단계하한가,확대 적용 | 선물/옵션(jangubun=5) |
| `76` | 3단계상한가,확대 적용 | 선물/옵션(jangubun=5) |
| `77` | 3단계하한가,확대 적용 | 선물/옵션(jangubun=5) |

---

## FC9 / OC0 — 실시간 체결 (WebSocket)

**방식:** WebSocket (실시간 스트리밍)
**FC9:** 선물 실시간 체결
**OC0:** 옵션 실시간 체결

### InBlock (구독 요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `tr_cd` | 거래 CD | str | 3 | LS증권 거래코드 |
| `tr_key` | 단축코드 | str | 8 | 6자리 또는 8자리 |

### OutBlock (FC9 — 선물 체결)

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `chetime` | 체결시간 | str | 6 |
| `sign` | 전일대비구분 | str | 1 |
| `change` | 전일대비 | str | 6.2 |
| `drate` | 등락율 | str | 6.2 |
| `price` | 현재가 | str | 6.2 |
| `open` | 시가 | str | 6.2 |
| `high` | 고가 | str | 6.2 |
| `low` | 저가 | str | 6.2 |
| `cgubun` | 체결구분 | str | 1 |
| `cvolume` | 체결량 | str | 6 |
| `volume` | 누적거래량 | str | 12 |
| `value` | 누적거래대금 | str | 12 |
| `mdvolume` | 매도누적체결량 | str | 12 |
| `mdchecnt` | 매도누적체결건수 | str | 8 |
| `msvolume` | 매수누적체결량 | str | 12 |
| `mschecnt` | 매수누적체결건수 | str | 8 |
| `cpower` | 체결강도 | str | 9.2 |
| `offerho1` | 매도호가1 | str | 6.2 |
| `bidho1` | 매수호가1 | str | 6.2 |
| `openyak` | 미결제약정수량 | str | 8 |
| `k200jisu` | KOSPI200지수 | str | 6.2 |
| `theoryprice` | 이론가 | str | 6.2 |
| `kasis` | 괴리율 | str | 6.2 |
| `sbasis` | 시장BASIS | str | 6.2 |
| `ibasis` | 이론BASIS | str | 6.2 |
| `openyakcha` | 미결제약정증감 | str | 8 |
| `jgubun` | 장운영정보 | str | 2 |
| `jnilvolume` | 전일동시간대거래량 | str | 12 |
| `futcode` | 단축코드 | str | 8 |

### OutBlock (OC0 — 옵션 체결)

FC9와 동일한 구조에 아래 필드가 추가되며, `futcode` 대신 `optcode`로 변경됩니다.

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `eqva` | KOSPI등가 | str | 7.2 |
| `impv` | 내재변동성 | str | 6.2 |
| `timevalue` | 시간가치 | str | 6.2 |
| `optcode` | 단축코드 | str | 8 |

---

## FH9 / OH0 — 실시간 호가 (WebSocket)

**방식:** WebSocket (실시간 스트리밍)
**FH9:** 선물 실시간 5단계 호가
**OH0:** 옵션 실시간 5단계 호가

### InBlock (구독 요청)

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `tr_cd` | 거래 CD | str | 3 | LS증권 거래코드 |
| `tr_key` | 단축코드 | str | 8 | 6자리 또는 8자리 |

### OutBlock (FH9 / OH0 공통 구조)

> 각 호가 단계(1~5)마다 `offerhoN`, `bidhoN`, `offerremN`, `bidremN`, `offercntN`, `bidcntN` 필드가 반복됩니다.
> FH9의 호가수량 크기는 6자리, OH0는 7자리입니다.

#### 호가 시간

| 필드 | 한글명 | 타입 | 크기 |
|------|--------|------|------|
| `hotime` | 호가시간 | str | 6 |

#### 매도·매수 호가 (1~5단계)

| 필드 패턴 | 설명 | 타입 | 크기 (FH9 / OH0) |
|-----------|------|------|-----------------|
| `offerhoN` | 매도호가N | str | 6.2 |
| `bidhoN` | 매수호가N | str | 6.2 |
| `offerremN` | 매도호가수량N | str | 6 / 7 |
| `bidremN` | 매수호가수량N | str | 6 / 7 |
| `offercntN` | 매도호가건수N | str | 5 |
| `bidcntN` | 매수호가건수N | str | 5 |

#### 호가 합계

| 필드 | 한글명 | 타입 | 크기 (FH9 / OH0) |
|------|--------|------|-----------------|
| `totofferrem` | 매도호가총수량 | str | 6 / 7 |
| `totbidrem` | 매수호가총수량 | str | 6 / 7 |
| `totoffercnt` | 매도호가총건수 | str | 5 |
| `totbidcnt` | 매수호가총건수 | str | 5 |

#### 기타

| 필드 | 한글명 | 타입 | 크기 | 비고 |
|------|--------|------|------|------|
| `futcode` / `optcode` | 단축코드 | str | 8 | FH9: futcode / OH0: optcode |
| `danhochk` | 단일가호가여부 | str | 1 | |
| `alloc_gubun` | 배분적용구분 | str | 1 | |

---

## TR 요약 비교표

| TR | 구분 | 방식 | 호가 단계 | 주요 특징 |
|----|------|------|-----------|-----------|
| `t2111` | 선물·옵션 현재가 | REST | 없음 (단순 현재가) | Greeks, 지수, 가격제한 등 상세 |
| `t2301` | 옵션 일람 | REST | 매도·매수 1단계 | 전 행사가 콜·풋 일괄 조회 |
| `t8465` | 선물·옵션 차트(N분) | REST | 없음 | OHLCV N분봉, 연속조회 지원 |
| `t8418` | 업종 차트(N분) | REST | 없음 | OHLCV N분봉, 연속조회 지원, 압축 지원 |
| `t8466` | 선물·옵션 차트(일주월) | REST | 없음 | OHLCV 일/주/월봉, 연속조회 지원 |
| `FC9` | 선물 실시간 체결 | WebSocket | 매도·매수 1단계 | 실시간 체결 스트리밍 |
| `OC0` | 옵션 실시간 체결 | WebSocket | 매도·매수 1단계 | 실시간 체결 + 내재변동성 |
| `FH9` | 선물 실시간 호가 | WebSocket | **5단계** | 단계별 수량·건수 포함 |
| `OH0` | 옵션 실시간 호가 | WebSocket | **5단계** | 단계별 수량·건수 포함 |

> **다단계 호가(5단계)**는 `FH9`(선물), `OH0`(옵션) WebSocket TR을 통해 수신합니다.

---

## 실시간 Tick 표준화(`tick_norm`)

본 저장소는 eBest 실시간 payload의 스키마 차이/환경별 누락 키를 흡수하기 위해,
원본 `tick`을 그대로 보존하면서 **표준화된 딕셔너리(`tick_norm`)** 를 함께 생성한다.

- 원본 payload: `tick` (eBest에서 받은 raw dict)
- 표준 payload: `tick_norm` (본 저장소의 공통 schema)

### 생성 위치

- `ebest_live.py`의 실시간 콜백에서 `tick_normalizer.normalize_realtime_tick()`를 호출해 `tick_norm`을 만든다.
- `PredictionPipeline.add_realtime_tick({...})`에 `{tick, tick_norm}` 형태로 함께 전달한다.
- `tick_processor.py`는 `tick_norm`이 존재하면 이를 우선 사용하고, 없으면 raw `tick`으로 fallback 한다.

### 목적

- 스키마 필드의 의미 혼동 방지
  - 예: `cvolume`(단건 체결량) vs `volume`(누적거래량)
- 여러 모듈에서 중복되는 파싱/alias 처리 최소화
- 하위호환 유지 (raw `tick`은 항상 유지)

### 표준 스키마(요약)

#### 공통

| 필드 | 타입 | 설명 |
|------|------|------|
| `trcode` | str | TR 코드 (`FC9`, `OC0`, `FH9`, `OH0`) |
| `symbol` | str | 구독 심볼 |
| `chetime` | str | 체결시간(HHMMSS), 제공 시 |

#### FC9 / OC0 (체결)

| 필드 | 타입 | 설명 |
|------|------|------|
| `price` | float | 현재가 |
| `open` | float | 시가 |
| `high` | float | 고가 |
| `low` | float | 저가 |
| `cvolume` | int | 단건 체결량 |
| `volume` | int | 누적거래량 |
| `value` | float | 누적거래대금 |
| `bid1` | float | 매수 1호가 |
| `ask1` | float | 매도 1호가 |
| `openyak` | int | 미결제약정수량 |
| `k200jisu` | float | KOSPI200지수 |
| `theoryprice` | float | 이론가 |

OC0 추가:

| 필드 | 타입 | 설명 |
|------|------|------|
| `optcode` | str | 단축코드 |
| `impv` | float | 내재변동성 |
| `timevalue` | float | 시간가치 |
| `eqva` | float | KOSPI 등가 |

FC9 추가:

| 필드 | 타입 | 설명 |
|------|------|------|
| `futcode` | str | 단축코드 |

#### FH9 / OH0 (호가 5단계)

| 필드 | 타입 | 설명 |
|------|------|------|
| `hotime` | str | 호가시간(HHMMSS) |
| `offerhos` | list[float] | 매도호가 1~5 (길이 5) |
| `bidhos` | list[float] | 매수호가 1~5 (길이 5) |
| `offerrems` | list[float] | 매도수량 1~5 (길이 5) |
| `bidrems` | list[float] | 매수수량 1~5 (길이 5) |
| `offercnts` | list[float] | 매도건수 1~5 (길이 5) |
| `bidcnts` | list[float] | 매수건수 1~5 (길이 5) |
| `totofferrem` | float | 매도 총수량 |
| `totbidrem` | float | 매수 총수량 |
| `totoffercnt` | float | 매도 총건수 |
| `totbidcnt` | float | 매수 총건수 |
| `danhochk` | str | 단일가호가여부 |
| `alloc_gubun` | str | 배분적용구분 |

FH9 추가: `futcode` / OH0 추가: `optcode`
