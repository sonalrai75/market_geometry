WINDOWS = (63, 126, 252)
HORIZON = 5
RIDGE = 1.0e-3

XCOLS = [
    "VIX",
    "Treasury10Y",
    "Curve10Y2Y",
    "CreditSpread_BAA10Y",
    "WTI",
    "BroadUSD",
    "RSP_SPY_Ratio",
    "SPY_Volume_Z20",
]

YCOLS = [
    "SPY_Fwd5D_Return",
    "QQQ_Fwd5D_Return",
    "SPY_RVol20",
    "SPY_Drawdown63",
]

FRIENDLY = {
    "VIX": "VIX",
    "Treasury10Y": "10Y Treasury",
    "Curve10Y2Y": "10Y–2Y Curve",
    "CreditSpread_BAA10Y": "Baa Credit Spread",
    "WTI": "WTI Oil",
    "BroadUSD": "Broad USD",
    "RSP_SPY_Ratio": "Breadth (RSP/SPY)",
    "SPY_Volume_Z20": "SPY Volume",
}
