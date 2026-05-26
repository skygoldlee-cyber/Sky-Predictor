# Market Open Subscription Flow (Pre-Open -> Open)

This document describes how this project handles **pre-open** and **market-open** flows for:

- Getting KP200 previous close (pre-open reference)
- Calculating ATM and building option call/put lists
- Registering realtime subscriptions (JIF/FC0/IJ_/OC0)
- Recalculating ATM using KP200 **open price** after market open
- Subscribing missing OC0 symbols that were not subscribed pre-open

---

## Key files / entry points

- `ebest_live.py::run_ebest_live_mode`
  - Live-mode loop entry point.
  - Calls `_initialize_api(...)` then runs the periodic prediction loop.

- `ebest_live.py::_initialize_api`
  - Pre-open realtime registration.
  - Starts a background task `_post_open_init()` which gates post-open snapshots/subscriptions.

- `ebest_callbacks.py::_make_realtime_callback`
  - Receives realtime ticks (`JIF`, `FC0`, `FH0`, `OC0`, `OH0`, `IJ_`).
  - Sets `state.market_opened=True` when JIF indicates open.

- `ebest_options.py::_filter_option_symbols_by_atm`
  - ATM-based symbol selection.

- `ebest_options.py::filter_option_symbols_dynamic_otm_by_open`
  - Post-open symbol selection using open-map (OTM filtered by `otm_open_min`, and capped by `max_otm_calls/max_otm_puts`).

---

## Pre-open: KP200 previous close source (`t8432`)

### Goal

Before market open, the project uses a stable reference price to calculate ATM.
In this repository, previous close is obtained during initialization and stored on the live state.

### How it is populated

- `_initialize_api(...)` attempts to call:
  - `_ebest_fetch_front_month_and_all_option_symbols(...)`
    - best-effort pack containing `kp200_symbol`, `kp200_prev_close`, and option symbol lists
- If `kp200_symbol` is missing, it falls back to:
  - `_ebest_fetch_kp200_symbol(...)`

The prev close is persisted as:

- `state.kp200_prev_close`
- and also attached to the predictor (best-effort): `predictor.kp200_prev_close`

---

## Pre-open: building base call/put lists (ATM-based)

### Goal

Before market open, compute ATM using a reference price and build an initial OC0 universe.
This repo selects symbols directly from the option symbol lists using an ATM strike grid.

### Builder

- `ebest_options.py::_filter_option_symbols_by_atm(...)`

Inputs:

- `calls` / `puts` option symbol lists (fetched from eBest)
- `underlying_price`
  - Pre-open: `state.kp200_prev_close` is used as the reference price when possible

Outputs:

- `sel_calls`, `sel_puts`
- `atm_strike`

---

## Pre-open: realtime registration

### Entry

- `ebest_live.py::_initialize_api(...)`

### What gets registered

- Always:
  - `FC0` (KP200 futures trades)
  - `FH0` (KP200 futures quote / orderbook snapshot)
  - `JIF` (market operation)
  - `IJ_` (spot index)

- If `include_options=True`:
  - `OC0` (options trades) for the initial pre-open symbol set
  - `OH0` (options quotes) for ATM±N universe (pre-open)

### Important behavior

- Pre-open OC0 selection intentionally avoids relying on t2301 open-map.
  - OTM selection can be enabled pre-open using the same caps as post-open:
    - `max_otm_calls`, `max_otm_puts`

- Pre-open OH0 selection is intentionally limited for performance.
  - It subscribes only within `ATM±preopen_oh0_window`.

### Subscription tracking

During pre-open OC0 registration, subscribed OC0 symbols are recorded into:

- `state.subscribed_oc0`

This is later used to compute the "missing" list after open.

---

## Market open trigger (JIF)

### Where it is handled

- `ebest_callbacks.py::_make_realtime_callback(...)`

When `JIF` tick indicates open (`jangubun=="5" and jstatus=="21"`), it sets:

- `state.market_opened = True`

If JIF is not received, there is a fallback time policy in `_post_open_init()`:

- emits `[GATE_FALLBACK] ...`
- sets `state.market_opened=True` during market hours (KST) after a wait threshold

---

## After open: recalc ATM using KP200 open and subscribe missing OC0

### Orchestration point

- `ebest_live.py::_initialize_api` spawns `asyncio.create_task(_post_open_init())`

Sequence:

1. Wait until `state.market_opened=True` (JIF open or time-based fallback).
2. Fetch post-open snapshots:
   - `t8415` price (best-effort)
   - `t2101` futures snapshot (used for `open` price when available)
   - `t2301` snapshot and `t2301 open_map`
3. Recalculate the desired OC0 universe using `filter_option_symbols_dynamic_otm_by_open(...)`.
4. Subscribe missing only:
   - `missing = desired - state.subscribed_oc0`

### Implementation

- `ebest_live.py::_post_open_init()`

Behavior:

- Determine `underlying_open`:
  - Prefer `t2101` snapshot `open`
  - Fallback to `predictor.tick_processor.get_current_price()`

- Determine `desired` symbols:
  - `filter_option_symbols_dynamic_otm_by_open(...)` with:
    - `otm_open_min`, `max_otm_calls`, `max_otm_puts` (from `config.json: options_subscription`)

- Compute missing and subscribe:
  - `missing = desired - state.subscribed_oc0`
  - `await _ebest_register_realtime(api, trcode="OC0", symbol=sym)`

Logs:

```text
[OPTIONS_CFG] opt_itm=...->... wait_sec=...->... otm_open_min=... max_otm_calls=... max_otm_puts=... preopen_oh0_window=...
[eBest] subscribe OC0 (pre-open) calls=... puts=... ATM=... prev_close=... (otm_caps call=... put=...)
[eBest] subscribe OH0 (pre-open, ATM±N) symbols=... ATM=... prev_close=...
[eBest] pre-open subscription breakdown: OC0 calls=... puts=... OH0 symbols=... total=...
[OPEN_FLOW] include_options=... option_month_info=... opt_itm=... otm_open_min=... max_otm_calls=... max_otm_puts=...
[GATE_FALLBACK] ... (only when JIF open is not received)
[OPEN_FLOW] t2301 open_map unavailable ... (when open_map is missing)
[OPEN_FLOW] open_map sizes: call=... put=...
[eBest] subscribe OC0 (post-open) calls=... puts=... open=... ATM=... (otm_open_min=... otm_caps call=... put=...)
[eBest] post-open subscription breakdown: OC0 calls=... puts=... desired=... already=... missing=...
[OPEN][OC0] desired=... missing=... open=... ATM=...
[OPEN][OC0] added_missing=N
```

---

## Notes / gotchas

- Post-open OC0 subscription is best-effort and depends on external data availability.
  - If `t2301 open_map` is unavailable, the flow still runs but may not apply OTM filtering.

- The open-based resubscribe step is intended to run once.
  - Guarded by: `state.open_oc0_subscribed`
