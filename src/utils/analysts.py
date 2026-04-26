"""Constants and utilities related to analysts configuration."""

from src.agents import portfolio_manager
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.nassim_taleb import nassim_taleb_agent
from src.agents.news_sentiment import news_sentiment_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.deterministic import (
    piotroski_agent,
    greenblatt_agent,
    momentum_agent,
    low_volatility_agent,
    acquirers_multiple_agent,
    multi_factor_agent,
    earnings_revision_agent,
    mohanram_agent,
    dividend_aristocrat_agent,
    etf_profile_agent,
    dashan_huang_agent,
    cross_stock_agent,
)
from src.agents.personas import (
    kenneth_fisher_agent,
    motley_fool_agent,
)

# Define analyst configuration - single source of truth.
#
# Each entry's `kind` field declares whether the agent is a heuristic LLM
# persona ("heuristic") or a pure-math factor agent ("deterministic"). The
# field is used both for workflow construction and for the dashboard's
# split-view rendering. Existing personas default to "heuristic" via the
# fallback in get_agents_list / the frontend.
ANALYST_CONFIG = {
    "aswath_damodaran": {
        "display_name": "Aswath Damodaran",
        "description": "The Dean of Valuation",
        "investing_style": "Focuses on intrinsic value and financial metrics to assess investment opportunities through rigorous valuation analysis.",
        "agent_func": aswath_damodaran_agent,
        "type": "analyst",
        "order": 0,
    },
    "ben_graham": {
        "display_name": "Ben Graham",
        "description": "The Father of Value Investing",
        "investing_style": "Emphasizes a margin of safety and invests in undervalued companies with strong fundamentals through systematic value analysis.",
        "agent_func": ben_graham_agent,
        "type": "analyst",
        "order": 1,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "description": "The Activist Investor",
        "investing_style": "Seeks to influence management and unlock value through strategic activism and contrarian investment positions.",
        "agent_func": bill_ackman_agent,
        "type": "analyst",
        "order": 2,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "description": "The Queen of Growth Investing",
        "investing_style": "Focuses on disruptive innovation and growth, investing in companies that are leading technological advancements and market disruption.",
        "agent_func": cathie_wood_agent,
        "type": "analyst",
        "order": 3,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "description": "The Rational Thinker",
        "investing_style": "Advocates for value investing with a focus on quality businesses and long-term growth through rational decision-making.",
        "agent_func": charlie_munger_agent,
        "type": "analyst",
        "order": 4,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "description": "The Big Short Contrarian",
        "investing_style": "Makes contrarian bets, often shorting overvalued markets and investing in undervalued assets through deep fundamental analysis.",
        "agent_func": michael_burry_agent,
        "type": "analyst",
        "order": 5,
    },
    "mohnish_pabrai": {
        "display_name": "Mohnish Pabrai",
        "description": "The Dhandho Investor",
        "investing_style": "Focuses on value investing and long-term growth through fundamental analysis and a margin of safety.",
        "agent_func": mohnish_pabrai_agent,
        "type": "analyst",
        "order": 6,
    },
    "nassim_taleb": {
        "display_name": "Nassim Taleb",
        "description": "The Black Swan Risk Analyst",
        "investing_style": "Focuses on tail risk, antifragility, and asymmetric payoffs. Uses barbell strategy, avoids fragile companies via negativa, and seeks convex positions with limited downside and unlimited upside.",
        "agent_func": nassim_taleb_agent,
        "type": "analyst",
        "order": 7,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "description": "The 10-Bagger Investor",
        "investing_style": "Invests in companies with understandable business models and strong growth potential using the 'buy what you know' strategy.",
        "agent_func": peter_lynch_agent,
        "type": "analyst",
        "order": 8,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "description": "The Scuttlebutt Investor",
        "investing_style": "Emphasizes investing in companies with strong management and innovative products, focusing on long-term growth through scuttlebutt research.",
        "agent_func": phil_fisher_agent,
        "type": "analyst",
        "order": 9,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "Rakesh Jhunjhunwala",
        "description": "The Big Bull Of India",
        "investing_style": "Leverages macroeconomic insights to invest in high-growth sectors, particularly within emerging markets and domestic opportunities.",
        "agent_func": rakesh_jhunjhunwala_agent,
        "type": "analyst",
        "order": 10,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "description": "The Macro Investor",
        "investing_style": "Focuses on macroeconomic trends, making large bets on currencies, commodities, and interest rates through top-down analysis.",
        "agent_func": stanley_druckenmiller_agent,
        "type": "analyst",
        "order": 11,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "description": "The Oracle of Omaha",
        "investing_style": "Seeks companies with strong fundamentals and competitive advantages through value investing and long-term ownership.",
        "agent_func": warren_buffett_agent,
        "type": "analyst",
        "order": 12,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "description": "Chart Pattern Specialist",
        "investing_style": "Focuses on chart patterns and market trends to make investment decisions, often using technical indicators and price action analysis.",
        "agent_func": technical_analyst_agent,
        "type": "analyst",
        "order": 13,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "description": "Financial Statement Specialist",
        "investing_style": "Delves into financial statements and economic indicators to assess the intrinsic value of companies through fundamental analysis.",
        "agent_func": fundamentals_analyst_agent,
        "type": "analyst",
        "order": 14,
    },
    "growth_analyst": {
        "display_name": "Growth Analyst",
        "description": "Growth Specialist",
        "investing_style": "Analyzes growth trends and valuation to identify growth opportunities through growth analysis.",
        "agent_func": growth_analyst_agent,
        "type": "analyst",
        "order": 15,
    },
    "news_sentiment_analyst": {
        "display_name": "News Sentiment Analyst",
        "description": "News Sentiment Specialist",
        "investing_style": "Analyzes news sentiment to predict market movements and identify opportunities through news analysis.",
        "agent_func": news_sentiment_agent,
        "type": "analyst",
        "order": 16,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "description": "Market Sentiment Specialist",
        "investing_style": "Gauges market sentiment and investor behavior to predict market movements and identify opportunities through behavioral analysis.",
        "agent_func": sentiment_analyst_agent,
        "type": "analyst",
        "order": 17,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "description": "Company Valuation Specialist",
        "investing_style": "Specializes in determining the fair value of companies, using various valuation models and financial metrics for investment decisions.",
        "agent_func": valuation_analyst_agent,
        "type": "analyst",
        "order": 18,
    },
    # ─── Deterministic factor agents ────────────────────────────────────────────
    # Pure math, no LLM call. Each translates published rules from finance
    # research into a fixed signal/confidence based on thresholds.
    "piotroski": {
        "display_name": "Piotroski F-Score",
        "description": "Quality filter (9-point financial-strength score)",
        "investing_style": "Pure-math quality screen on profitability, leverage, and operating efficiency. Bullish at F ≥ 7, bearish at F ≤ 3.",
        "agent_func": piotroski_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 100,
    },
    "greenblatt": {
        "display_name": "Greenblatt (Magic Formula)",
        "description": "Earnings yield × ROIC composite",
        "investing_style": "Two-factor deep-value/quality from 'The Little Book That Beats the Market': EBIT/EV and EBIT/Invested-Capital.",
        "agent_func": greenblatt_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 101,
    },
    "acquirers_multiple": {
        "display_name": "Carlisle (Acquirer's Multiple)",
        "description": "Single-factor EV/EBIT deep value",
        "investing_style": "Tobias Carlisle's deep-value test. Sub-8 EV/EBIT bullish, > 15 bearish.",
        "agent_func": acquirers_multiple_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 102,
    },
    "momentum": {
        "display_name": "Momentum (12-1)",
        "description": "Price strength factor",
        "investing_style": "Standard academic 12-month return excluding the most recent month. Positive trend → bullish.",
        "agent_func": momentum_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 103,
    },
    "low_volatility": {
        "display_name": "Low Volatility",
        "description": "Defensive low-vol factor (van Vliet)",
        "investing_style": "Annualized realized volatility of trailing 60d, blended with a Sharpe-like reading. Low vol + non-negative return → bullish.",
        "agent_func": low_volatility_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 104,
    },
    "multi_factor": {
        "display_name": "Multi-Factor Composite",
        "description": "Weighted value/quality/momentum/low-vol",
        "investing_style": "Composite that ranks the stock on four classical factors and weights them 30/30/20/20.",
        "agent_func": multi_factor_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 105,
    },
    "earnings_revision": {
        "display_name": "Earnings Revision",
        "description": "Surprise history + estimate trend",
        "investing_style": "Composite of last-4-quarter beats/misses, average surprise %, and 8-quarter slope of analyst estimate-EPS.",
        "agent_func": earnings_revision_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 106,
    },
    "mohanram": {
        "display_name": "Mohanram G-Score",
        "description": "Growth-stock 7-point quality screen",
        "investing_style": "Mohanram (2005) G-Score. Profitability, accruals quality, earnings/sales stability, R&D and CapEx intensity.",
        "agent_func": mohanram_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 107,
    },
    "dividend_aristocrat": {
        "display_name": "Dividend Aristocrat",
        "description": "TTM yield + 5-yr CAGR + streak",
        "investing_style": "Real split-adjusted per-share dividend history from AV: bullish on multi-year non-decreasing payers with positive 5-yr CAGR; bearish on cuts or non-payers.",
        "agent_func": dividend_aristocrat_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 108,
    },
    "etf_profile": {
        "display_name": "ETF Profile",
        "description": "ETF size, cost, sector concentration",
        "investing_style": "ETF-only structural analysis: net assets, expense ratio, turnover, sector & holding concentration, leveraged flag. Returns low-confidence neutral on non-ETF tickers.",
        "agent_func": etf_profile_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 109,
    },
    "dashan_huang": {
        "display_name": "Dashan Huang (Twin Momentum)",
        "description": "Fundamental momentum + 12-1 price momentum",
        "investing_style": "Huang-Zhang-Zhong (2018): combines a 7-variable fundamental composite (earnings, ROE, ROA, AOP/E, COP/A, GP/A, net payout) with classical 12-1 price momentum. Both positive → strong bullish; both negative → strong bearish.",
        "agent_func": dashan_huang_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 110,
    },
    "cross_stock": {
        "display_name": "Cross-Stock Concentration",
        "description": "Penalize sector/industry overlap within the run",
        "investing_style": "Huang et al. (2026, adapted): builds a sector/industry/market-cap candidate graph across every ticker in the request and counts each ticker's neighbors. 0-1 neighbors → diversifying (bullish); 3+ → concentration risk (bearish). Stops the PM from doubling down on a single cluster (e.g. NVDA+AMD+INTC).",
        "agent_func": cross_stock_agent,
        "type": "analyst",
        "kind": "deterministic",
        "order": 111,
    },
    # ─── Heuristic LLM personas (Validea-style) ─────────────────────────────────
    # Personas kept here are the ones whose qualitative judgment cannot be
    # reduced to a single deterministic factor. Mechanical Validea screens
    # (Greenblatt, O'Shaughnessy, Zweig, Neff, Dreman, Wesley Gray, Meb Faber)
    # were removed because they duplicate existing deterministic agents
    # (greenblatt, multi_factor, dividend_aristocrat, acquirers_multiple, etc.).
    "kenneth_fisher": {
        "display_name": "Kenneth Fisher",
        "description": "Super Stocks / P/S focus",
        "investing_style": "Ken Fisher's Super Stocks framework: low P/S, sales growth, FCF positive, modest debt.",
        "agent_func": kenneth_fisher_agent,
        "type": "analyst",
        "kind": "heuristic",
        "order": 200,
    },
    "motley_fool": {
        "display_name": "Motley Fool",
        "description": "Foolish Small Cap growth",
        "investing_style": "Sub-$3B small-caps with 25%+ revenue/earnings growth and healthy margins.",
        "agent_func": motley_fool_agent,
        "type": "analyst",
        "kind": "heuristic",
        "order": 206,
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}


def get_agents_list():
    """Get the list of agents for API responses."""
    return [
        {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "investing_style": config["investing_style"],
            "order": config["order"],
            "kind": config.get("kind", "heuristic"),
        }
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]
