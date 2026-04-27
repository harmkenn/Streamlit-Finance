@st.cache_data(ttl=300)
def get_all_metrics(tickers):
    results = []

    if not tickers:
        return results

    hist = yf.download(
        tickers,
        period="1mo",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            # --- Price data
            fast = stock.fast_info or {}
            current = fast.get("last_price")
            opening = fast.get("open")

            # --- Fundamentals
            try:
                info = stock.get_info()
            except:
                info = {}

            pe = info.get("trailingPE")
            fpe = info.get("forwardPE")
            peg = info.get("pegRatio")
            ps = info.get("priceToSalesTrailing12Months")
            pb = info.get("priceToBook")
            sector = info.get("sector")
            industry = info.get("industry")

            # --- Historical gains
            try:
                if isinstance(hist.columns, pd.MultiIndex):
                    data = hist[ticker]
                else:
                    data = hist

                close = data["Close"].dropna()

                day_gain = week_gain = month_gain = None

                if len(close) > 1 and current:
                    yesterday = close.iloc[-2]
                    if yesterday != 0:
                        day_gain = (current - yesterday) / yesterday * 100

                if len(close) >= 7 and current:
                    week_price = close.iloc[-7]
                    if week_price != 0:
                        week_gain = (current / week_price - 1) * 100

                if len(close) >= 22 and current:
                    month_price = close.iloc[-22]
                    if month_price != 0:
                        month_gain = (current / month_price - 1) * 100

            except:
                day_gain = week_gain = month_gain = None

            # --- Scoring
            def score_metric(value, good, bad):
                if value is None:
                    return None
                if value <= good:
                    return 100
                if value >= bad:
                    return 0
                return 100 * (bad - value) / (bad - good)

            scores = []
            for val, good, bad in [
                (pe, 15, 30),
                (peg, 1, 2),
                (ps, 2, 5),
                (pb, 1.5, 4)
            ]:
                s = score_metric(val, good, bad)
                if s is not None:
                    scores.append(s)

            overall_score = round(sum(scores) / len(scores), 1) if scores else None

            if overall_score is None:
                label = "N/A"
            elif overall_score >= 70:
                label = "Undervalued"
            elif overall_score >= 40:
                label = "Fair"
            else:
                label = "Expensive"

            # --- Output
            results.append({
                "Ticker": ticker.upper(),
                "Price": safe_round(current),
                "PE": safe_round(pe),
                "Forward PE": safe_round(fpe),
                "PEG": safe_round(peg),
                "PS": safe_round(ps),
                "PB": safe_round(pb),
                "Sector": sector,
                "Industry": industry,
                DAY_COL: safe_round(day_gain),
                WEEK_COL: safe_round(week_gain),
                MONTH_COL: safe_round(month_gain),
                "Score": overall_score,
                "Valuation": label,
                "Updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            })

        except:
            continue

    return results
