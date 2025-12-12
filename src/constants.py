"""
Constants for the Portfolio Management Application
"""

STOCK_SYMBOLS_BY_COUNTRY = {
    "United States": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
        "V", "WMT", "JPM", "PG", "MA", "HD", "CVX", "ABBV", "PFE", "KO",
        "AVGO", "PEP", "COST", "TMO", "DHR", "MRK", "VZ", "ADBE", "WFC", "BAC",
        "NFLX", "CRM", "XOM", "LLY", "ABT", "ORCL", "ACN", "NVS", "CMCSA", "DIS",
        "CSCO", "TXN", "MDT", "PM", "QCOM", "HON", "RTX", "UPS", "LOW", "NKE",
        "INTC", "AMGN", "SPGI", "INTU", "CAT", "GS", "IBM", "SBUX", "AMD", "T"
    ],
    "United Kingdom": [
        "LLOY.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L", "VOD.L", "LSEG.L", "RIO.L", "HSBA.L", "GSK.L",
        "BARC.L", "NG.L", "DGE.L", "BT-A.L", "REL.L", "GLEN.L", "AAL.L", "NWG.L", "STAN.L", "PRU.L",
        "SSE.L", "CNA.L", "FLTR.L", "IAG.L", "RB.L", "CRDA.L", "INF.L", "LAND.L", "IMB.L", "III.L",
        "ADM.L", "ANTO.L", "AUTO.L", "AV.L", "BA.L", "BNZL.L", "BRBY.L", "CCL.L", "CPG.L", "CRDS.L",
        "EXPN.L", "FRAS.L", "HLMA.L", "IHG.L", "JET.L", "KGF.L", "LGEN.L", "MNG.L", "OCDO.L", "PSH.L",
        "RTO.L", "SGRO.L", "SMDS.L", "SPX.L", "TW.L", "UU.L", "WTB.L", "3IN.L", "ABDN.L"
    ],
    "Australia": [
        "CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "WOW.AX", "FMG.AX", "MQG.AX", "WES.AX",
        "TLS.AX", "RIO.AX", "TCL.AX", "GMG.AX", "STO.AX", "QBE.AX", "ASX.AX", "COL.AX", "JHX.AX", "REA.AX",
        "AMP.AX", "ALL.AX", "APT.AX", "ASP.AX", "AWC.AX", "BEN.AX", "BKL.AX", "BLD.AX", "BOQ.AX", "BPT.AX",
        "BRG.AX", "BSL.AX", "BWP.AX", "CAR.AX", "CCP.AX", "CHC.AX", "CPU.AX", "CTX.AX", "CWN.AX", "DMP.AX",
        "DXS.AX", "ELD.AX", "EVN.AX", "FLT.AX", "GOR.AX", "GPT.AX", "HVN.AX", "IAG.AX", "IEL.AX", "IGO.AX",
        "ILU.AX", "IPL.AX", "JBH.AX", "LLC.AX", "MGR.AX", "MIN.AX", "NEC.AX", "NHF.AX", "NST.AX", "ORA.AX"
    ],
    "Hong Kong": [
        "0700.HK", "0941.HK", "0388.HK", "0005.HK", "1299.HK", "2318.HK", "0939.HK", "3690.HK", "0883.HK", "1398.HK",
        "2388.HK", "0267.HK", "0175.HK", "0002.HK", "0011.HK", "0016.HK", "0027.HK", "1109.HK", "0006.HK", "0001.HK",
        "0012.HK", "0017.HK", "0019.HK", "0023.HK", "0066.HK", "0083.HK", "0101.HK", "0144.HK", "0151.HK", "0200.HK",
        "0291.HK", "0293.HK", "0322.HK", "0386.HK", "0390.HK", "0392.HK", "0688.HK", "0762.HK", "0823.HK", "0857.HK",
        "0868.HK", "0881.HK", "0914.HK", "0916.HK", "0960.HK", "0968.HK", "0992.HK", "1044.HK", "1072.HK", "1093.HK",
        "1113.HK", "1171.HK", "1177.HK", "1211.HK", "1288.HK", "1336.HK", "1378.HK", "1816.HK", "1880.HK", "1928.HK"
    ],
    "China": [
        "BABA", "JD", "BIDU", "NIO", "PDD", "BILI", "TME", "IQ", "NTES", "VIPS",
        "YMM", "LI", "XPEV", "EDU", "TAL", "WB", "DOYU", "KC", "TUYA", "DADA",
        "YSG", "TIGR", "FUTU", "RLX", "GOTU", "MOMO", "HUYA", "DOCU", "ZTO", "YTO",
        "STO", "BEST", "QFIN", "LKNCY", "ZLAB", "CAAS", "CBPO", "CANG", "CAN", "CARS",
        "CADC", "CXDC", "DQ", "EH", "FENG", "GSMG", "HEAR", "HCM", "HIMX", "HUIZ",
        "JOBS", "LAIX", "LX", "NAAS", "NIU", "QTT", "RERE", "SOHU", "TOUR", "WDH"
    ]
}

AVAILABLE_COUNTRIES = [
    "United States", "United Kingdom", "Australia", "Hong Kong", "China"
]