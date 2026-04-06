"""
XFeatures — External Feature Download & Cache for Crypto Assets.

Downloads daily external series (DVOL, Fear & Greed Index, News Sentiment)
and caches them as CSVs in this directory.  Both Fincast and Kronos pipelines
can read the same cached files.

Usage:
    from Data_MLA.XFeatures import load_xfeatures

    xf = load_xfeatures(
        date_start="2023-01-01",
        date_end="2026-01-01",
        force_refresh=False,
    )
    # xf is a DataFrame indexed by date (daily) with columns:
    #   dvol, fear_greed_idx, news_sentiment
"""