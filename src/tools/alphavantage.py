"""Alpha Vantage data adapter.

Translates the six top-level data calls the hedge fund needs into Alpha Vantage's
REST API, returning the same pydantic shapes (`Price`, `FinancialMetrics`,
`LineItem`, `InsiderTrade`, `CompanyNews`) the rest of the codebase expects.

Premium tier (1200 req/min) is assumed. Heavy in-memory caching of the three
quarterly statements keeps real call volume down to ~1 OVERVIEW + 1
TIME_SERIES_DAILY_ADJUSTED + 3 statement calls per ticker per process.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date as _date
from typing import Any, Iterable

import requests


# Concurrency cap on AV requests. Caps how many calls can be in-flight at once
# so 19 parallel agents don't trip AV's per-second fair-use ceiling. Premium
# accounts handle ~6 concurrent fine; tune via ALPHAVANTAGE_CONCURRENCY=N.
_AV_SEMAPHORE = threading.Semaphore(int(os.environ.get("ALPHAVANTAGE_CONCURRENCY", "6")))

# Per-(function, symbol) locks so concurrent agents asking for the SAME data
# don't all fire — they coalesce, the first call hits AV, the rest read cache.
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: dict[tuple, threading.Lock] = {}


def _coalesce_lock(key: tuple) -> threading.Lock:
    with _INFLIGHT_LOCK:
        lock = _INFLIGHT.get(key)
        if lock is None:
            lock = threading.Lock()
            _INFLIGHT[key] = lock
        return lock

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)

_BASE = "https://www.alphavantage.co/query"
_TIMEOUT = 30


# Tier toggle. Defaults to PREMIUM whenever an ALPHA_VANTAGE_API_KEY is set —
# Premium real-time entitles every endpoint we use. Override to free with
# DATA_PROVIDER_TIER=free if you ever swap in a free-tier key.
def _is_premium() -> bool:
    tier = (os.environ.get("DATA_PROVIDER_TIER") or "").strip().lower()
    if tier in ("premium", "pro", "paid", "realtime"):
        return True
    if tier in ("free", "basic"):
        return False
    return bool(os.environ.get("ALPHA_VANTAGE_API_KEY"))


# Endpoints that AV gates behind a paid plan. We hit these only on premium and
# silently return empty results on free tier — agents already handle missing
# data gracefully (default to neutral).
_PREMIUM_ONLY = {
    "INSIDER_TRANSACTIONS",
    "NEWS_SENTIMENT",
    "TIME_SERIES_DAILY_ADJUSTED",
    "TIME_SERIES_INTRADAY",  # without entitlement
}

# Endpoints proven dead on the current account (e.g. responded with a "premium"
# notice this session). Subsequent calls short-circuit instead of burning quota.
_DEAD_ENDPOINTS: set[str] = set()


# ─── small per-process cache of the heavy ticker-level payloads ──────────────────
# AV reports change at most once a quarter; we don't need to re-fetch them inside
# a single hedge-fund run.
_OVERVIEW: dict[str, dict] = {}
_INCOME: dict[str, dict] = {}
_BALANCE: dict[str, dict] = {}
_CASHFLOW: dict[str, dict] = {}
_DAILY: dict[str, dict] = {}


def _key(api_key: str | None) -> str:
    """Resolve the AV API key.

    NOTE: the `api_key` parameter on this module's public functions exists for
    interface compatibility with the original Financial Datasets adapter, where
    callers pass FINANCIAL_DATASETS_API_KEY. That key is **not** an AV key and
    must never be sent to alphavantage.co — doing so makes AV identify the
    request as unknown/free-tier (25/day rate limit). Always prefer the
    ALPHA_VANTAGE_API_KEY env var; only fall back to the parameter if it
    *looks* like an AV key (hex-ish, no special chars typical of FD keys).
    """
    env_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if env_key:
        return env_key
    if api_key and len(api_key) <= 32 and api_key.isalnum():
        return api_key
    raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set.")


def _get(params: dict) -> dict:
    """GET an AV endpoint and return the JSON body.

    Three response shapes handled:
      • Real data → return as-is.
      • Data + advisory Note/Information → return data, log quietly.
      • Note/Information only (throttled OR premium-gated) → empty + cache the
        endpoint as 'dead' so we stop pounding it for the rest of the session.
    """
    func = params.get("function") or ""
    if func in _DEAD_ENDPOINTS:
        return {}

    # Block known premium-only calls when we're on free tier — keeps quota intact
    # and avoids noisy upsell warnings.
    if not _is_premium() and func in _PREMIUM_ONLY:
        logger.debug("AV: skipping premium-only %s on free tier", func)
        return {}

    for attempt in range(3):
        with _AV_SEMAPHORE:
            try:
                r = requests.get(_BASE, params=params, timeout=_TIMEOUT)
            except requests.RequestException as e:
                logger.warning("AV request error: %s (attempt %d)", e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
                continue
        if r.status_code != 200:
            logger.warning("AV non-200 %s for %s", r.status_code, func)
            return {}
        try:
            data = r.json()
        except Exception:
            logger.warning("AV non-JSON body for %s", func)
            return {}

        if isinstance(data, dict) and ("Note" in data or "Information" in data):
            msg = (data.get("Note") or data.get("Information") or "").strip()
            ignored_keys = {"Note", "Information", "Meta Data"}
            data_keys = [k for k in data.keys() if k not in ignored_keys]

            if data_keys:
                # Notice rode along with real data — use it.
                logger.debug("AV notice (data still present): %s", msg[:120])
                return data

            # Notice-only response. On premium accounts these are transient
            # fair-use throttles under burst load — back off and retry quietly.
            # On free accounts a "premium" message is a hard gate; cache it.
            low = msg.lower()
            premium_gated = (
                "premium endpoint" in low
                or "premium feature" in low
                or "premium plan" in low
            )
            if premium_gated and not _is_premium():
                logger.warning("AV %s requires premium; suppressing further calls.", func)
                _DEAD_ENDPOINTS.add(func)
                return {}

            logger.debug("AV transient notice on %s (attempt %d): %s", func, attempt + 1, msg[:160])
            # NEWS_SENTIMENT throttles aggressively under burst load and the
            # cost per failed retry is high (1.5–4.5s). Cap at 1 retry for it
            # so we fail fast and let the negative-cache kick in.
            if func == "NEWS_SENTIMENT" and attempt >= 1:
                logger.warning("AV %s throttled — short-circuiting after 2 attempts.", func)
                return {}
            time.sleep(0.8 + attempt * 1.0)
            continue

        return data

    logger.warning("AV %s gave only notices after 3 attempts; returning empty.", func)
    return {}


def _f(x: Any) -> float | None:
    if x is None or x == "" or x == "None" or x == "-":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0, 0.0):
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None


# ─── Overview / market cap ───────────────────────────────────────────────────────
def _get_overview(ticker: str, api_key: str | None) -> dict:
    cached = _OVERVIEW.get(ticker)
    if cached:
        return cached
    with _coalesce_lock(("OVERVIEW", ticker)):
        cached = _OVERVIEW.get(ticker)  # double-check after acquiring lock
        if cached:
            return cached
        data = _get({"function": "OVERVIEW", "symbol": ticker, "apikey": _key(api_key)})
        if data:
            _OVERVIEW[ticker] = data
        return data or {}


# ─── Statement helpers ───────────────────────────────────────────────────────────
def _statement(function: str, ticker: str, api_key: str | None, store: dict) -> dict:
    cached = store.get(ticker)
    if cached:
        return cached
    with _coalesce_lock((function, ticker)):
        cached = store.get(ticker)
        if cached:
            return cached
        data = _get({"function": function, "symbol": ticker, "apikey": _key(api_key)})
        if data:
            store[ticker] = data
        return data or {}


def _income(ticker: str, api_key: str | None) -> dict:
    return _statement("INCOME_STATEMENT", ticker, api_key, _INCOME)


def _balance(ticker: str, api_key: str | None) -> dict:
    return _statement("BALANCE_SHEET", ticker, api_key, _BALANCE)


def _cashflow(ticker: str, api_key: str | None) -> dict:
    return _statement("CASH_FLOW", ticker, api_key, _CASHFLOW)


def _periods(statement: dict, period: str) -> list[dict]:
    """Pick annual or quarterly reports out of a statement payload."""
    key = "annualReports" if period == "annual" else "quarterlyReports"
    return statement.get(key, []) or []


def _filter_by_end_date(reports: list[dict], end_date: str | None) -> list[dict]:
    """AV returns reports newest-first. Drop anything past `end_date`."""
    if not end_date:
        return reports
    return [r for r in reports if r.get("fiscalDateEnding", "") <= end_date]


# ─── PUBLIC: get_prices ──────────────────────────────────────────────────────────
def get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
    cache_key = f"{ticker}"
    series = _DAILY.get(cache_key)
    if not series:
        with _coalesce_lock(("DAILY", ticker)):
            series = _DAILY.get(cache_key)
            if not series:
                outputsize = "full" if _is_premium() else "compact"
                if _is_premium():
                    data = _get({
                        "function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": ticker,
                        "outputsize": outputsize,
                        "apikey": _key(api_key),
                    })
                    series = data.get("Time Series (Daily)") or {}
                else:
                    series = {}
                if not series:
                    data = _get({
                        "function": "TIME_SERIES_DAILY",
                        "symbol": ticker,
                        "outputsize": outputsize,
                        "apikey": _key(api_key),
                    })
                    series = data.get("Time Series (Daily)") or {}
                if series:
                    _DAILY[cache_key] = series

    out: list[Price] = []
    for day, bars in series.items():
        if day < start_date or day > end_date:
            continue
        # Adjusted endpoint uses keys "1. open" .. "6. volume"; non-adjusted ".5. volume"
        try:
            out.append(Price(
                open=float(bars["1. open"]),
                high=float(bars["2. high"]),
                low=float(bars["3. low"]),
                close=float(bars.get("5. adjusted close", bars["4. close"])),
                volume=int(float(bars.get("6. volume", bars.get("5. volume", 0)))),
                time=day,
            ))
        except (KeyError, ValueError):
            continue
    out.sort(key=lambda p: p.time)
    return out


# ─── PUBLIC: get_market_cap ─────────────────────────────────────────────────────
def get_market_cap(ticker: str, end_date: str | None = None, api_key: str | None = None) -> float | None:
    ov = _get_overview(ticker, api_key)
    return _f(ov.get("MarketCapitalization"))


# ─── PUBLIC: get_financial_metrics ──────────────────────────────────────────────
def _price_on_or_before(ticker: str, day: str, api_key: str | None) -> float | None:
    """Closest cached close on or before `day`. Triggers a daily-series fetch if needed."""
    series = _DAILY.get(ticker)
    if series is None:
        # Reuse the same fetch path get_prices uses.
        get_prices(ticker, "1990-01-01", day, api_key=api_key)
        series = _DAILY.get(ticker) or {}
    if not series:
        return None
    candidates = [d for d in series.keys() if d <= day]
    if not candidates:
        return None
    bars = series[max(candidates)]
    try:
        return float(bars.get("5. adjusted close", bars.get("4. close")))
    except (TypeError, ValueError):
        return None


def _ttm_sum(reports: list[dict], idx: int, field: str) -> float | None:
    """Sum the last 4 quarterly values starting at index `idx` (newest at 0)."""
    if idx + 4 > len(reports):
        return None
    total = 0.0
    for j in range(idx, idx + 4):
        v = _f(reports[j].get(field))
        if v is None:
            return None
        total += v
    return total


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> list[FinancialMetrics]:
    """Synthesize FinancialMetrics rows from AV's OVERVIEW + statements.

    `period`:
      - 'ttm' / 'quarterly' → quarterly statements with TTM-rolled income items
      - 'annual'            → annual statements
    """
    ov = _get_overview(ticker, api_key)
    full_inc = _periods(_income(ticker, api_key), period)
    full_bs = _periods(_balance(ticker, api_key), period)
    full_cf = _periods(_cashflow(ticker, api_key), period)

    inc_q = _filter_by_end_date(full_inc, end_date)[:limit]
    bs_q = _filter_by_end_date(full_bs, end_date)[:limit]
    cf_q = _filter_by_end_date(full_cf, end_date)[:limit]

    is_ttm = period in ("ttm", "quarterly")

    # Snapshot ratios from OVERVIEW (latest period only).
    snap_pe = _f(ov.get("PERatio"))
    snap_pb = _f(ov.get("PriceToBookRatio"))
    snap_ps = _f(ov.get("PriceToSalesRatioTTM"))
    snap_ev_ebitda = _f(ov.get("EVToEBITDA"))
    snap_ev_rev = _f(ov.get("EVToRevenue"))
    snap_peg = _f(ov.get("PEGRatio"))
    market_cap = _f(ov.get("MarketCapitalization"))
    payout = _f(ov.get("PayoutRatio"))
    rev_growth_yoy = _f(ov.get("QuarterlyRevenueGrowthYOY"))
    eps_growth_yoy = _f(ov.get("QuarterlyEarningsGrowthYOY"))
    snap_eps = _f(ov.get("EPS"))

    out: list[FinancialMetrics] = []
    n = max(len(inc_q), len(bs_q), len(cf_q))
    for i in range(n):
        inc = inc_q[i] if i < len(inc_q) else {}
        bs = bs_q[i] if i < len(bs_q) else {}
        cf = cf_q[i] if i < len(cf_q) else {}
        report = inc.get("fiscalDateEnding") or bs.get("fiscalDateEnding") or cf.get("fiscalDateEnding") or end_date

        # ── TTM rollups (quarterly only) — preferred over single-quarter values ──
        if is_ttm:
            revenue = _ttm_sum(full_inc, i, "totalRevenue")
            gross_profit = _ttm_sum(full_inc, i, "grossProfit")
            op_income = _ttm_sum(full_inc, i, "operatingIncome")
            net_income = _ttm_sum(full_inc, i, "netIncome")
            interest_expense = _ttm_sum(full_inc, i, "interestExpense")
        else:
            revenue = _f(inc.get("totalRevenue"))
            gross_profit = _f(inc.get("grossProfit"))
            op_income = _f(inc.get("operatingIncome"))
            net_income = _f(inc.get("netIncome"))
            interest_expense = _f(inc.get("interestExpense"))

        total_assets = _f(bs.get("totalAssets"))
        total_curr_assets = _f(bs.get("totalCurrentAssets"))
        total_curr_liab = _f(bs.get("totalCurrentLiabilities"))
        total_liab = _f(bs.get("totalLiabilities"))
        equity = _f(bs.get("totalShareholderEquity"))
        cash = _f(bs.get("cashAndCashEquivalentsAtCarryingValue"))
        inventory = _f(bs.get("inventory"))
        receivables = _f(bs.get("currentNetReceivables"))
        long_term_debt = _f(bs.get("longTermDebt")) or 0.0
        short_term_debt = _f(bs.get("shortTermDebt")) or 0.0
        current_debt = _f(bs.get("currentDebt")) or 0.0
        total_debt = (long_term_debt or 0) + (short_term_debt or 0) + (current_debt or 0)
        if total_debt == 0:
            total_debt = _f(bs.get("shortLongTermDebtTotal"))
        # Three-source share count, in order of preference:
        #   1. AV SHARES_OUTSTANDING endpoint at-or-before the report date — gives
        #      both basic and diluted directly with quarterly history.
        #   2. Derived: net_income_ttm / DilutedEPSTTM (from OVERVIEW).
        #   3. Basic common-stock shares outstanding from the balance sheet.
        report_dt = inc.get("fiscalDateEnding") or bs.get("fiscalDateEnding") or end_date
        shares_from_endpoint = shares_at_or_before(ticker, report_dt, api_key=api_key)
        basic_shares = _f(bs.get("commonStockSharesOutstanding")) or _f(ov.get("SharesOutstanding"))
        diluted_eps_ttm = _f(ov.get("DilutedEPSTTM"))
        diluted_shares: float | None = None
        ni_ttm = net_income if is_ttm else _ttm_sum(full_inc, i, "netIncome")
        if diluted_eps_ttm and ni_ttm and diluted_eps_ttm != 0:
            diluted_shares = ni_ttm / diluted_eps_ttm
        shares = shares_from_endpoint or diluted_shares or basic_shares

        if is_ttm:
            op_cf = _ttm_sum(full_cf, i, "operatingCashflow")
            capex = _ttm_sum(full_cf, i, "capitalExpenditures")
            da_ttm = _ttm_sum(full_cf, i, "depreciationDepletionAndAmortization")
            if da_ttm is None:
                da_ttm = _ttm_sum(full_inc, i, "depreciationAndAmortization")
            income_before_tax = _ttm_sum(full_inc, i, "incomeBeforeTax")
            income_tax_expense = _ttm_sum(full_inc, i, "incomeTaxExpense")
        else:
            op_cf = _f(cf.get("operatingCashflow"))
            capex = _f(cf.get("capitalExpenditures"))
            da_ttm = _f(cf.get("depreciationDepletionAndAmortization")) or _f(inc.get("depreciationAndAmortization"))
            income_before_tax = _f(inc.get("incomeBeforeTax"))
            income_tax_expense = _f(inc.get("incomeTaxExpense"))

        # Effective tax rate (bounded 0-40%; fall back to US statutory 21% if unknown)
        tax_rate: float
        if income_before_tax and income_tax_expense is not None and income_before_tax > 0:
            raw = income_tax_expense / income_before_tax
            tax_rate = max(0.0, min(raw, 0.40))
        else:
            tax_rate = 0.21

        # Change in working capital — prefer YoY comparison against the same-quarter-last-year
        # balance sheet (index i+4 if available). Falls back to zero when unavailable, which
        # is a reasonable assumption for mature businesses.
        delta_wc: float | None = None
        if total_curr_assets is not None and total_curr_liab is not None and i + 4 < len(full_bs):
            prior_bs = full_bs[i + 4]
            prior_ca = _f(prior_bs.get("totalCurrentAssets"))
            prior_cl = _f(prior_bs.get("totalCurrentLiabilities"))
            if prior_ca is not None and prior_cl is not None:
                delta_wc = (total_curr_assets - total_curr_liab) - (prior_ca - prior_cl)

        # Two FCF notions: legacy FCFE-ish (OCF − Capex), and the proper unlevered FCFF
        # (Damodaran textbook). Agents that look at `free_cash_flow` on FinancialMetrics
        # get the unlevered FCFF — that's what the DCF should discount.
        fcf = (op_cf - capex) if (op_cf is not None and capex is not None) else None
        fcff: float | None = None
        if op_income is not None and da_ttm is not None and capex is not None:
            nopat = op_income * (1.0 - tax_rate)
            fcff = nopat + da_ttm - capex - (delta_wc or 0.0)

        gross_margin = _safe_div(gross_profit, revenue)
        operating_margin = _safe_div(op_income, revenue)
        net_margin = _safe_div(net_income, revenue)
        roe = _safe_div(net_income, equity)
        roa = _safe_div(net_income, total_assets)
        debt_to_equity = _safe_div(total_debt, equity)
        debt_to_assets = _safe_div(total_debt, total_assets)
        current_ratio = _safe_div(total_curr_assets, total_curr_liab)
        # Quick ratio = (current assets - inventory) / current liabilities
        quick_ratio = _safe_div(
            (total_curr_assets - inventory) if (total_curr_assets is not None and inventory is not None) else None,
            total_curr_liab,
        )
        cash_ratio = _safe_div(cash, total_curr_liab)
        ocf_ratio = _safe_div(op_cf, total_curr_liab)
        interest_coverage = _safe_div(op_income, interest_expense) if (interest_expense and interest_expense > 0) else None

        # ROIC ≈ NOPAT / invested_capital. Approximate with op_income * (1 - tax_rate)
        # but tax_rate per period is messy; use net_income / (equity + debt) as proxy.
        roic = _safe_div(net_income, (equity or 0) + total_debt) if equity else None

        # Per-share figures use TTM shares snapshot when historical isn't available.
        eps = _safe_div(net_income, shares)
        bvps = _safe_div(equity, shares)
        fcf_per_share = _safe_div(fcf, shares)

        # Per-period valuation: price on the report date × shares → market cap snapshot,
        # then derive PE/PB/PS/FCF-yield/EV ratios from the rolled TTM income items.
        period_price = _price_on_or_before(ticker, report, api_key)
        period_mcap = (period_price * shares) if (period_price is not None and shares) else None
        period_ev = (period_mcap + (total_debt or 0) - (cash or 0)) if period_mcap is not None else None
        period_ebitda = None
        if op_income is not None:
            # crude TTM EBITDA = TTM operating income + TTM D&A (from cashflow if available)
            if is_ttm:
                da = _ttm_sum(full_cf, i, "depreciationDepletionAndAmortization")
            else:
                da = _f(cf.get("depreciationDepletionAndAmortization"))
            period_ebitda = (op_income + da) if (op_income is not None and da is not None) else op_income

        pe = _safe_div(period_price, eps) if eps and eps > 0 else None
        pb = _safe_div(period_price, bvps) if bvps and bvps > 0 else None
        ps = _safe_div(period_mcap, revenue) if revenue and revenue > 0 else None
        fcf_yield = _safe_div(fcf, period_mcap) if period_mcap else None
        ev_rev = _safe_div(period_ev, revenue) if (period_ev is not None and revenue) else None
        ev_ebitda = _safe_div(period_ev, period_ebitda) if (period_ev is not None and period_ebitda) else None
        peg = snap_peg if i == 0 else None  # AV doesn't expose historical PEG
        # Latest row falls back to OVERVIEW snapshots if computed values failed.
        if i == 0:
            pe = pe or snap_pe
            pb = pb or snap_pb
            ps = ps or snap_ps
            ev_ebitda = ev_ebitda or snap_ev_ebitda
            ev_rev = ev_rev or snap_ev_rev

        beta_snap = _f(ov.get("Beta")) if i == 0 else None

        # Per-period dividend yield: trailing 12m dividend per share at the report
        # date / period_price. Growth metric attached only on the latest row.
        ttm_dps = trailing_12m_dividend(ticker, report, api_key=api_key) if report else 0.0
        period_div_yield = (ttm_dps / period_price) if (period_price and ttm_dps) else None
        div_growth_5y = dividend_growth_cagr(ticker, report, years=5, api_key=api_key) if i == 0 else None

        out.append(FinancialMetrics(
            ticker=ticker,
            report_period=report,
            period=period,
            currency=inc.get("reportedCurrency") or bs.get("reportedCurrency") or "USD",
            market_cap=period_mcap or (market_cap if i == 0 else None),
            enterprise_value=period_ev,
            price_to_earnings_ratio=pe,
            price_to_book_ratio=pb,
            price_to_sales_ratio=ps,
            enterprise_value_to_ebitda_ratio=ev_ebitda,
            enterprise_value_to_revenue_ratio=ev_rev,
            free_cash_flow_yield=fcf_yield,
            peg_ratio=peg,
            # Raw totals for agents that pull them off FinancialMetrics directly.
            # `free_cash_flow` is the UNLEVERED FCFF Damodaran's DCF expects:
            #   NOPAT + D&A − Capex − ΔWC     where NOPAT = EBIT × (1 − tax_rate)
            # Falls back to OCF − Capex if we couldn't compute the FCFF form.
            free_cash_flow=fcff if fcff is not None else fcf,
            outstanding_shares=shares,  # diluted when derivable, else basic
            revenue=revenue,
            net_income=net_income,
            total_debt=total_debt if total_debt else None,
            ebit=op_income,
            interest_expense=interest_expense,
            ebitda=period_ebitda,
            beta=beta_snap,
            dividend_yield=period_div_yield,
            trailing_dividends_per_share=ttm_dps if ttm_dps else None,
            dividend_growth_5y=div_growth_5y,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=roic,
            asset_turnover=_safe_div(revenue, total_assets),
            inventory_turnover=_safe_div(revenue, inventory),
            receivables_turnover=_safe_div(revenue, receivables),
            days_sales_outstanding=_safe_div(receivables * 365 if receivables else None, revenue),
            operating_cycle=None,
            working_capital_turnover=_safe_div(
                revenue,
                (total_curr_assets - total_curr_liab) if (total_curr_assets is not None and total_curr_liab is not None) else None,
            ),
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio,
            operating_cash_flow_ratio=ocf_ratio,
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage,
            revenue_growth=rev_growth_yoy if i == 0 else None,
            earnings_growth=eps_growth_yoy if i == 0 else None,
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=payout if i == 0 else None,
            earnings_per_share=eps if eps is not None else (snap_eps if i == 0 else None),
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcf_per_share,
        ))
    return out


# ─── PUBLIC: search_line_items ──────────────────────────────────────────────────
# Mapping of hedge-fund line-item key → callable that pulls/derives it from
# (income_report, balance_report, cashflow_report, overview_dict).
def _li_revenue(inc, bs, cf, ov): return _f(inc.get("totalRevenue"))
def _li_gross_profit(inc, bs, cf, ov): return _f(inc.get("grossProfit"))
def _li_operating_expense(inc, bs, cf, ov): return _f(inc.get("operatingExpenses"))
def _li_operating_income(inc, bs, cf, ov): return _f(inc.get("operatingIncome"))
def _li_net_income(inc, bs, cf, ov): return _f(inc.get("netIncome"))
def _li_ebit(inc, bs, cf, ov):
    v = _f(inc.get("ebit"))
    if v is not None:
        return v
    op = _f(inc.get("operatingIncome"))
    return op  # Best-effort proxy
def _li_ebitda(inc, bs, cf, ov):
    v = _f(inc.get("ebitda"))
    if v is not None:
        return v
    ebit = _li_ebit(inc, bs, cf, ov)
    da = _f(inc.get("depreciationAndAmortization")) or _f(cf.get("depreciationDepletionAndAmortization"))
    if ebit is not None and da is not None:
        return ebit + da
    return None
def _li_interest_expense(inc, bs, cf, ov): return _f(inc.get("interestExpense"))
def _li_dep_amort(inc, bs, cf, ov):
    return _f(inc.get("depreciationAndAmortization")) or _f(cf.get("depreciationDepletionAndAmortization"))
def _li_rd(inc, bs, cf, ov): return _f(inc.get("researchAndDevelopment"))
def _li_eps(inc, bs, cf, ov):
    ni = _f(inc.get("netIncome"))
    sh = _f(bs.get("commonStockSharesOutstanding")) or _f(ov.get("SharesOutstanding"))
    return _safe_div(ni, sh)
def _li_outstanding_shares(inc, bs, cf, ov):
    # Prefer diluted weighted-average shares derived from TTM net income / EPS TTM
    # snapshot. Falls back to basic common-stock shares.
    ni = _f(inc.get("netIncome"))
    deps = _f(ov.get("DilutedEPSTTM"))
    # For single-quarter line items, scaling quarterly NI by TTM diluted EPS is wrong,
    # so we only apply this derivation when NI looks TTM-sized (>1.5x quarterly EPS × shares).
    basic = _f(bs.get("commonStockSharesOutstanding")) or _f(ov.get("SharesOutstanding"))
    if deps and deps != 0 and ni and basic:
        derived = ni / deps
        if derived > basic * 1.01:  # diluted is always ≥ basic in practice
            return derived
    return basic
def _li_total_assets(inc, bs, cf, ov): return _f(bs.get("totalAssets"))
def _li_current_assets(inc, bs, cf, ov): return _f(bs.get("totalCurrentAssets"))
def _li_total_liabilities(inc, bs, cf, ov): return _f(bs.get("totalLiabilities"))
def _li_current_liabilities(inc, bs, cf, ov): return _f(bs.get("totalCurrentLiabilities"))
def _li_equity(inc, bs, cf, ov): return _f(bs.get("totalShareholderEquity"))
def _li_cash(inc, bs, cf, ov): return _f(bs.get("cashAndCashEquivalentsAtCarryingValue"))
def _li_inventory(inc, bs, cf, ov): return _f(bs.get("inventory"))
def _li_intangible(inc, bs, cf, ov): return _f(bs.get("intangibleAssets")) or _f(bs.get("intangibleAssetsExcludingGoodwill"))
def _li_goodwill_intangible(inc, bs, cf, ov):
    g = _f(bs.get("goodwill")) or 0.0
    i = _f(bs.get("intangibleAssets")) or _f(bs.get("intangibleAssetsExcludingGoodwill")) or 0.0
    total = (g or 0) + (i or 0)
    return total if total else None
def _li_total_debt(inc, bs, cf, ov):
    parts = [
        _f(bs.get("longTermDebt")),
        _f(bs.get("shortTermDebt")),
        _f(bs.get("currentDebt")),
    ]
    s = sum(p for p in parts if p)
    if s:
        return s
    return _f(bs.get("shortLongTermDebtTotal"))
def _li_working_capital(inc, bs, cf, ov):
    a = _f(bs.get("totalCurrentAssets"))
    l = _f(bs.get("totalCurrentLiabilities"))
    if a is None or l is None:
        return None
    return a - l
def _li_capex(inc, bs, cf, ov):
    v = _f(cf.get("capitalExpenditures"))
    return -abs(v) if v is not None else None  # negative-as-outflow convention
def _li_fcf(inc, bs, cf, ov):
    """Unlevered FCFF for THIS period (not TTM — single-quarter line-item surface).

        NOPAT + D&A − Capex     where NOPAT = EBIT × (1 − effective_tax_rate)

    Falls back to the legacy OCF − Capex form when any input is missing.
    """
    ebit = _f(inc.get("ebit")) or _f(inc.get("operatingIncome"))
    da = _f(inc.get("depreciationAndAmortization")) or _f(cf.get("depreciationDepletionAndAmortization"))
    cx = _f(cf.get("capitalExpenditures"))
    ibt = _f(inc.get("incomeBeforeTax"))
    tax = _f(inc.get("incomeTaxExpense"))
    if ebit is not None and da is not None and cx is not None and ibt and tax is not None and ibt > 0:
        rate = max(0.0, min(tax / ibt, 0.40))
        return ebit * (1.0 - rate) + da - cx
    op = _f(cf.get("operatingCashflow"))
    if op is not None and cx is not None:
        return op - cx
    return None
def _li_dividends(inc, bs, cf, ov):
    v = _f(cf.get("dividendPayout")) or _f(cf.get("dividendPayoutCommonStock"))
    return -abs(v) if v is not None else None
def _li_equity_issuance(inc, bs, cf, ov):
    issued = _f(cf.get("proceedsFromIssuanceOfCommonStock")) or 0.0
    repurchased = _f(cf.get("paymentsForRepurchaseOfCommonStock")) or 0.0
    return (issued or 0) - (repurchased or 0)
def _li_book_value_per_share(inc, bs, cf, ov):
    eq = _f(bs.get("totalShareholderEquity"))
    sh = _f(bs.get("commonStockSharesOutstanding")) or _f(ov.get("SharesOutstanding"))
    return _safe_div(eq, sh)
def _li_debt_to_equity(inc, bs, cf, ov):
    return _safe_div(_li_total_debt(inc, bs, cf, ov), _f(bs.get("totalShareholderEquity")))
def _li_gross_margin(inc, bs, cf, ov):
    return _safe_div(_f(inc.get("grossProfit")), _f(inc.get("totalRevenue")))
def _li_operating_margin(inc, bs, cf, ov):
    return _safe_div(_f(inc.get("operatingIncome")), _f(inc.get("totalRevenue")))
def _li_roic(inc, bs, cf, ov):
    ni = _f(inc.get("netIncome"))
    eq = _f(bs.get("totalShareholderEquity")) or 0
    debt = _li_total_debt(inc, bs, cf, ov) or 0
    denom = (eq or 0) + (debt or 0)
    return _safe_div(ni, denom) if denom else None


_LINE_ITEM_RESOLVERS = {
    "revenue": _li_revenue,
    "gross_profit": _li_gross_profit,
    "operating_expense": _li_operating_expense,
    "operating_income": _li_operating_income,
    "net_income": _li_net_income,
    "ebit": _li_ebit,
    "ebitda": _li_ebitda,
    "interest_expense": _li_interest_expense,
    "depreciation_and_amortization": _li_dep_amort,
    "research_and_development": _li_rd,
    "earnings_per_share": _li_eps,
    "outstanding_shares": _li_outstanding_shares,
    "total_assets": _li_total_assets,
    "current_assets": _li_current_assets,
    "total_liabilities": _li_total_liabilities,
    "current_liabilities": _li_current_liabilities,
    "shareholders_equity": _li_equity,
    "cash_and_equivalents": _li_cash,
    "inventory": _li_inventory,
    "intangible_assets": _li_intangible,
    "goodwill_and_intangible_assets": _li_goodwill_intangible,
    "total_debt": _li_total_debt,
    "working_capital": _li_working_capital,
    "capital_expenditure": _li_capex,
    "free_cash_flow": _li_fcf,
    "dividends_and_other_cash_distributions": _li_dividends,
    "issuance_or_purchase_of_equity_shares": _li_equity_issuance,
    "book_value_per_share": _li_book_value_per_share,
    "debt_to_equity": _li_debt_to_equity,
    "gross_margin": _li_gross_margin,
    "operating_margin": _li_operating_margin,
    "return_on_invested_capital": _li_roic,
}


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> list[LineItem]:
    inc_reports = _filter_by_end_date(_periods(_income(ticker, api_key), period), end_date)[:limit]
    bs_reports = _filter_by_end_date(_periods(_balance(ticker, api_key), period), end_date)[:limit]
    cf_reports = _filter_by_end_date(_periods(_cashflow(ticker, api_key), period), end_date)[:limit]
    ov = _get_overview(ticker, api_key)

    n = max(len(inc_reports), len(bs_reports), len(cf_reports))
    out: list[LineItem] = []
    for i in range(n):
        inc = inc_reports[i] if i < len(inc_reports) else {}
        bs = bs_reports[i] if i < len(bs_reports) else {}
        cf = cf_reports[i] if i < len(cf_reports) else {}
        report_period = inc.get("fiscalDateEnding") or bs.get("fiscalDateEnding") or cf.get("fiscalDateEnding") or end_date
        currency = inc.get("reportedCurrency") or bs.get("reportedCurrency") or cf.get("reportedCurrency") or "USD"

        payload: dict[str, Any] = {
            "ticker": ticker,
            "report_period": report_period,
            "period": period,
            "currency": currency,
        }
        for name in line_items:
            resolver = _LINE_ITEM_RESOLVERS.get(name)
            payload[name] = resolver(inc, bs, cf, ov) if resolver else None
        out.append(LineItem(**payload))
    return out


# ─── PUBLIC: get_insider_trades ─────────────────────────────────────────────────
_INSIDER_CACHE: dict[str, list] = {}


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> list[InsiderTrade]:
    # Insider data doesn't change inside a single hedge-fund run; one fetch per ticker.
    raw = _INSIDER_CACHE.get(ticker)
    if raw is None:
        with _coalesce_lock(("INSIDER_TRANSACTIONS", ticker)):
            raw = _INSIDER_CACHE.get(ticker)
            if raw is None:
                data = _get({
                    "function": "INSIDER_TRANSACTIONS",
                    "symbol": ticker,
                    "apikey": _key(api_key),
                })
                raw = data.get("data") or []
                if raw:
                    _INSIDER_CACHE[ticker] = raw
    rows = raw or []
    out: list[InsiderTrade] = []
    for r in rows:
        filing = r.get("transaction_date") or r.get("filing_date") or ""
        if end_date and filing and filing > end_date:
            continue
        if start_date and filing and filing < start_date:
            continue
        try:
            shares = _f(r.get("shares"))
            price = _f(r.get("share_price"))
            value = (shares * price) if (shares is not None and price is not None) else None
            # AV uses A=Acquisition, D=Disposal (sale).
            acq = (r.get("acquisition_or_disposal") or "").upper().startswith("A")
            signed_shares = shares if acq else (-shares if shares is not None else None)
            out.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=r.get("executive"),
                title=r.get("executive_title"),
                is_board_director=None,
                transaction_date=filing,
                transaction_shares=signed_shares,
                transaction_price_per_share=price,
                transaction_value=value if acq or value is None else -value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=r.get("security_type"),
                filing_date=filing or "1970-01-01",
            ))
        except Exception as e:
            logger.debug("Skipped insider row: %s", e)
        if len(out) >= limit:
            break
    return out


# ─── PUBLIC: get_company_news ───────────────────────────────────────────────────
def _bucket_sentiment(score: float | None) -> str | None:
    if score is None:
        return None
    if score <= -0.15:
        return "bearish"
    if score >= 0.15:
        return "bullish"
    return "neutral"


def _av_time(t: str) -> str:
    # AV news uses "20240115T093000" → "2024-01-15"
    if not t or len(t) < 8:
        return t or ""
    return f"{t[0:4]}-{t[4:6]}-{t[6:8]}"


_NEWS_CACHE: dict[str, list] = {}  # keyed by ticker only — fetch once, filter in-memory
_NEWS_NEGATIVE_CACHE: set[str] = set()  # tickers we already failed on this run


def reset_news_negative_cache() -> None:
    """Clear the news negative-cache so a new run can re-attempt tickers that
    were transiently throttled in a previous run. Called by run_hedge_fund at
    the start of each invocation; transient throttles last seconds, not the
    whole process lifetime.
    """
    _NEWS_NEGATIVE_CACHE.clear()
_EARNINGS_CACHE: dict[str, dict] = {}
_DIVIDENDS_CACHE: dict[str, list] = {}
_SPLITS_CACHE: dict[str, list] = {}
_SHARES_CACHE: dict[str, list] = {}
_EARNINGS_ESTIMATES_CACHE: dict[str, list] = {}
_ETF_PROFILE_CACHE: dict[str, dict] = {}


def get_earnings_history(ticker: str, api_key: str | None = None) -> dict:
    """AV's EARNINGS endpoint: quarterly reportedEPS / estimatedEPS / surprise."""
    cached = _EARNINGS_CACHE.get(ticker)
    if cached:
        return cached
    with _coalesce_lock(("EARNINGS", ticker)):
        cached = _EARNINGS_CACHE.get(ticker)
        if cached:
            return cached
        data = _get({"function": "EARNINGS", "symbol": ticker, "apikey": _key(api_key)})
        if data:
            _EARNINGS_CACHE[ticker] = data
        return data or {}


def get_dividend_history(ticker: str, api_key: str | None = None) -> list[dict]:
    """AV's DIVIDENDS endpoint: full per-share dividend event history.

    Returns a list ordered newest-first with keys
    ex_dividend_date / declaration_date / record_date / payment_date / amount.
    """
    cached = _DIVIDENDS_CACHE.get(ticker)
    if cached is not None:
        return cached
    with _coalesce_lock(("DIVIDENDS", ticker)):
        cached = _DIVIDENDS_CACHE.get(ticker)
        if cached is not None:
            return cached
        data = _get({"function": "DIVIDENDS", "symbol": ticker, "apikey": _key(api_key)})
        rows = data.get("data") or [] if isinstance(data, dict) else []
        # Sort newest-first by ex_dividend_date for consistent downstream use.
        rows.sort(key=lambda d: d.get("ex_dividend_date") or "", reverse=True)
        if rows:
            _DIVIDENDS_CACHE[ticker] = rows
        return rows


def get_splits_history(ticker: str, api_key: str | None = None) -> list[dict]:
    """AV's SPLITS endpoint, sorted newest-first by effective_date."""
    cached = _SPLITS_CACHE.get(ticker)
    if cached is not None:
        return cached
    with _coalesce_lock(("SPLITS", ticker)):
        cached = _SPLITS_CACHE.get(ticker)
        if cached is not None:
            return cached
        data = _get({"function": "SPLITS", "symbol": ticker, "apikey": _key(api_key)})
        rows = data.get("data") or [] if isinstance(data, dict) else []
        rows.sort(key=lambda d: d.get("effective_date") or "", reverse=True)
        if rows:
            _SPLITS_CACHE[ticker] = rows
        return rows


def split_factor_after(ticker: str, ex_date: str, api_key: str | None = None) -> float:
    """Multiply all split factors with effective_date AFTER `ex_date`.

    Example: AAPL paid $0.73 in 2018-08 (pre 4:1 split on 2020-08-31). To compare
    that to 2026 amounts, we divide by the split factor (4.0) so the value is
    expressed in current shares: $0.73 / 4 = $0.18 effective.

    Returns 1.0 when no splits intervene.
    """
    splits = get_splits_history(ticker, api_key=api_key)
    if not splits:
        return 1.0
    factor = 1.0
    for s in splits:
        eff = s.get("effective_date") or ""
        if not eff or eff <= ex_date:
            continue
        try:
            factor *= float(s.get("split_factor") or 1.0)
        except (TypeError, ValueError):
            continue
    return factor


def trailing_12m_dividend(ticker: str, end_date: str, api_key: str | None = None) -> float:
    """Split-adjusted sum of per-share dividends with ex-dividend date in (end_date − 1y, end_date]."""
    rows = get_dividend_history(ticker, api_key=api_key)
    if not rows:
        return 0.0
    end = end_date
    one_year_ago = f"{int(end[:4]) - 1}{end[4:]}"
    total = 0.0
    for r in rows:
        ex = r.get("ex_dividend_date") or ""
        if not ex or ex > end or ex <= one_year_ago:
            continue
        amt = _f(r.get("amount"))
        if amt is None:
            continue
        # AV reports amounts as paid (un-split-adjusted). Divide by the cumulative
        # split factor of any splits AFTER this ex date so historical amounts are
        # expressed in current-share terms.
        total += amt / split_factor_after(ticker, ex, api_key=api_key)
    return total


def dividend_growth_cagr(ticker: str, end_date: str, years: int = 5, api_key: str | None = None) -> float | None:
    """N-year CAGR of split-adjusted trailing-12m per-share dividends."""
    if years <= 0:
        return None
    end = end_date
    start = f"{int(end[:4]) - years}{end[4:]}"
    latest = trailing_12m_dividend(ticker, end_date=end, api_key=api_key)
    earliest = trailing_12m_dividend(ticker, end_date=start, api_key=api_key)
    if not latest or not earliest or latest <= 0 or earliest <= 0:
        return None
    return (latest / earliest) ** (1.0 / years) - 1.0


def get_shares_outstanding(ticker: str, api_key: str | None = None) -> list[dict]:
    """AV's SHARES_OUTSTANDING endpoint — quarterly basic + diluted history."""
    cached = _SHARES_CACHE.get(ticker)
    if cached is not None:
        return cached
    with _coalesce_lock(("SHARES_OUTSTANDING", ticker)):
        cached = _SHARES_CACHE.get(ticker)
        if cached is not None:
            return cached
        data = _get({"function": "SHARES_OUTSTANDING", "symbol": ticker, "apikey": _key(api_key)})
        rows = data.get("data") or [] if isinstance(data, dict) else []
        rows.sort(key=lambda d: d.get("date") or "", reverse=True)
        if rows:
            _SHARES_CACHE[ticker] = rows
        return rows


def shares_at_or_before(ticker: str, date: str, *, prefer_diluted: bool = True, api_key: str | None = None) -> float | None:
    """Pick the most recent SHARES_OUTSTANDING row at or before `date`.

    Returns diluted when available + > basic, else basic.
    """
    rows = get_shares_outstanding(ticker, api_key=api_key)
    if not rows:
        return None
    for r in rows:
        if (r.get("date") or "") <= date:
            diluted = _f(r.get("shares_outstanding_diluted"))
            basic = _f(r.get("shares_outstanding_basic"))
            if prefer_diluted and diluted is not None and (basic is None or diluted >= basic):
                return diluted
            if basic is not None:
                return basic
            return diluted
    return None


def get_earnings_estimates(ticker: str, api_key: str | None = None) -> list[dict]:
    """AV's EARNINGS_ESTIMATES endpoint.

    Returned list is what AV ships under data["estimates"]. Each row has
    `eps_estimate_average` plus *_7_days_ago, *_30_days_ago, *_60_days_ago,
    *_90_days_ago — the data needed to compute true analyst revisions.
    """
    cached = _EARNINGS_ESTIMATES_CACHE.get(ticker)
    if cached is not None:
        return cached
    with _coalesce_lock(("EARNINGS_ESTIMATES", ticker)):
        cached = _EARNINGS_ESTIMATES_CACHE.get(ticker)
        if cached is not None:
            return cached
        data = _get({"function": "EARNINGS_ESTIMATES", "symbol": ticker, "apikey": _key(api_key)})
        rows = data.get("estimates") or [] if isinstance(data, dict) else []
        if rows:
            _EARNINGS_ESTIMATES_CACHE[ticker] = rows
        return rows


def get_etf_profile(ticker: str, api_key: str | None = None) -> dict:
    """AV's ETF_PROFILE endpoint. Returns {} for non-ETF tickers."""
    cached = _ETF_PROFILE_CACHE.get(ticker)
    if cached is not None:
        return cached
    with _coalesce_lock(("ETF_PROFILE", ticker)):
        cached = _ETF_PROFILE_CACHE.get(ticker)
        if cached is not None:
            return cached
        data = _get({"function": "ETF_PROFILE", "symbol": ticker, "apikey": _key(api_key)})
        # AV returns {} or an error blob for non-ETFs; only cache when we got real fields.
        if isinstance(data, dict) and data.get("net_assets") is not None:
            _ETF_PROFILE_CACHE[ticker] = data
            return data
        return {}


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> list[CompanyNews]:
    """Fetch news ONCE per ticker per process and filter the date window in-memory.

    Older code keyed cache on (ticker, start_date, end_date) so 7 agents asking
    different date windows each fired their own request. Under AV news-throttle
    that meant ~9s × 7 = 60+ seconds of wasted backoff PER RUN.

    Negative cache prevents re-attempting throttled tickers within the same run.
    """
    feed: list | None = _NEWS_CACHE.get(ticker)
    if feed is None and ticker not in _NEWS_NEGATIVE_CACHE:
        with _coalesce_lock(("NEWS_SENTIMENT", ticker)):
            feed = _NEWS_CACHE.get(ticker)
            if feed is None and ticker not in _NEWS_NEGATIVE_CACHE:
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": min(limit, 1000),
                    "apikey": _key(api_key),
                }
                # Use only end_date — we want the broadest cached feed; per-agent
                # callers filter the window themselves below.
                if end_date:
                    params["time_to"] = end_date.replace("-", "") + "T2359"
                data = _get(params)
                feed = data.get("feed") or []
                if feed:
                    _NEWS_CACHE[ticker] = feed
                else:
                    _NEWS_NEGATIVE_CACHE.add(ticker)
    if feed is None:
        feed = []

    # Per-agent date filter applied here (in-memory)
    def _in_window(time_published: str) -> bool:
        if not time_published:
            return True
        d = _av_time(time_published)
        if start_date and d < start_date:
            return False
        if end_date and d > end_date:
            return False
        return True
    feed = [item for item in feed if _in_window(item.get("time_published", ""))]

    out: list[CompanyNews] = []
    for item in feed:
        try:
            pub = _av_time(item.get("time_published", ""))
            # Ticker-specific sentiment if present (preferred), else overall.
            tsent = None
            tlabel = None
            relevance = None
            for ts in item.get("ticker_sentiment", []) or []:
                if ts.get("ticker") == ticker:
                    tsent = _f(ts.get("ticker_sentiment_score"))
                    tlabel = ts.get("ticker_sentiment_label")
                    relevance = _f(ts.get("relevance_score"))
                    break
            if tsent is None:
                tsent = _f(item.get("overall_sentiment_score"))
            if tlabel is None:
                tlabel = item.get("overall_sentiment_label")

            out.append(CompanyNews(
                ticker=ticker,
                title=item.get("title") or "",
                author=(item.get("authors") or [None])[0],
                source=item.get("source") or "Alpha Vantage",
                date=pub,
                url=item.get("url") or "",
                sentiment=_bucket_sentiment(tsent),
                sentiment_label=tlabel,
                sentiment_score=tsent,
                relevance=relevance,
                summary=item.get("summary"),
                topics=item.get("topics"),
            ))
        except Exception as e:
            logger.debug("Skipped news row: %s", e)
        if len(out) >= limit:
            break
    return out
