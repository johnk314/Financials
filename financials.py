# Run in notebooks_postgres conda environment.
import os
import re
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

## Utils
def get_response(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                       AppleWebKit/605.1.15 (KHTML, like Gecko) \
                       Chrome/100.0.4896.127 Safari/605.1.15 Firefox/100.0"
    }

    session = requests.Session()
    response = session.get(url, headers=headers)

    return response


def to_float(value):
    if type(value) == str:
        result = float(re.search("[0-9\.\-]+", value).group())
        if "B" in value:
            result *= 1000
    else:
        result = value

    return result

## S&P 500 Companies
def get_sp500_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S&P_500_companies"
    data = pd.read_html(url)[0]

    return data

## Stock Screener
def stock_screener():
    url= "https://www.macrotrends.net/stocks/stock-screener"

    response = get_response(url).text
    data = json.loads(re.findall("(?<=\ \=\ )[\S\ ]+(?=\;)", response)[2])
    df = pd.DataFrame(data)

    return df

## Glossary
# ticker                      Ticker or trading symbol
# comp_name                   Company name
# comp_name_2                 Company name 2
# exchange                    Exchange traded
# country_code                Country code (abbreviation)
# zacks_x_ind_desc            Zacks expanded industry description
# zacks_x_sector_desc         Zacks expanded sector description
# zacks_m_ind_desc            Zacks medium industry description
# emp_cnt                     Number of employees
# tot_revenue_f0              Total revenues for most recent completed fiscal year
# net_income_f0               Net Income reported for most recent completed fiscal year
# iad                         Indicated annual dividend (IAD), based on the dividend paid last quarter
# diluted_eps_net_4q          Diluted earnings per share (EPS) for the most recent trailing 12 months
# eps_mean_est_qr0            Estimated earnings per share (EPS) for the last reported fiscal quarter (QR0)
# eps_act_qr0                 Actual earnings per share (EPS) for the last reported fiscal quarter (QR0)
# eps_amt_diff_surp_qr0       Surprise amount for the last reported quarter (QR0) in total dollars
# eps_pct_diff_surp_qr0       Surprise percent for the last reported fiscal quarter (QR0)
# eps_mean_est_fr1            Estimated earnings per share (EPS) for the current fiscal year (FR1)
# eps_mean_est_fr2            Estimated earnings per share (EPS) for the next fiscal year (FR2)
# eps_act_fr0                 Actual earnings per share (EPS) for the last reported fiscal year (FR0)
# free_float                  Shares outstanding available to common shareholdings on the open market, calculated as total shares outstanding times one minus the percentage held by insiders
# price_20ma                  
# price_30y                   
# beta                        Stock return relative to the S&P 500 over the past 60 months, including dividends
# market_val                  Market capitalization (current price times current shares outstanding, including all classes of common shares)
# zacks_oper_margin_q0        Zacks operating margin for the most recent trailing twelve month period, Net Income before Non-Recurring Items and Discontinued Operations divided by Net Sales
# price_1w                    
# pe_ratio_12m                Price to earnings (PE) ratio using current price over trailing 12 months of EPS estimate before non-recurring items
# pre_tax_margin_q0           Pre-tax margin for the most recent trailing twelve month period, Pre-Tax Income divided by Net Sales
# net_margin_q0               Net margin for the most recent trailing twelve month period, GAAP Net Income divided by Net Sales
# avg_vol_20d                 Average daily volume in actual shares traded for the last 20 trading days
# price_1m                    
# div_yield                   Indicated Annual Dividend (IAD) divided by Price Per Share
# held_by_insiders_pct        Percentage of shares outstanding held by insiders from the latest annual proxy report
# price_3m                    
# forward_pe_ratio            
# peg_ratio                   Price to earnings growth (PEG) ratio, F1 P/E Ratio divided by the Long Term Growth rate estimate
# held_by_institutions_pct    Percentage of shares outstanding held by institutions from the latest 13F reports filed
# price_6m                    
# last_close                  
# price_cash                  Current price over most current available annual cash flow per common share
# cons_recom_curr             Current consensus recommendation, arithmetic mean of all recommendations
# price_1y                    
# price_ytd                   
# eps_growth_qoq              
# price_per_sales             Current price over trailing 12 months of total revenues
# price_book                  Current price over most current available quarterly book value per common share
# price_3y                    
# dividend_payout_ratio       
# debt_to_comm_equ_q0         Most recent reported quarter"s Total Long-Term debt over most recent quarter"s Total Common Equity
# price_5y                    
# sales_growth_qoq            
# roe_q0                      Return on common equity for the most recent trailing twelve month period, TTM Net Income divided by Avg Total Common Equity
# roa_q0                      Return on assets for the most recent trailing twelve month period, TTM Net Income divided by Avg Total Assets
# price_10y                   
# curr_ratio_q0               Most recent quarter current ratio: Current Assets divided by Current Liabilities
# price_20y                   
# quick_ratio_q0              Most recent reported quarter quick ratio, (Current Assets - Inventory) divided by Current Liabilities
# fifty_two_week              
# inv_turnover_q0             Inventory turnover: TTM COGS divided by 4 Qtrs Avg Total Inventory
# price_50ma                  
# price_200ma                 
# eps_growth_5y               
# sales_growth_5y             
# eps_growth_this_yr          
# eps_growth_next_yr          
# name_link                   

## Baseclass for defining Ticker.
class SingleBase:
    def __init__(self, ticker):
        self._ticker = ticker.upper()
        self._base_stocks_url = f"https://www.macrotrends.net/stocks/charts/{self._ticker}"
        self._base_assets_url = "https://www.macrotrends.net/assets/php"
        self._years = 10


    def _get_url(self, item):
        response = get_response(self._base_stocks_url)
        short_name = response.url.split("/")[-2]

        url = f"{self._base_stocks_url}/{short_name}/{item}"

        return url


    def _get_info(self):
        try:
            info = stock_screener().set_index("ticker").loc[self._ticker]

            return info

        except KeyError:
            print("No data found.")


    def _is_condition(self):
        try:
            url = self._get_url("stock-price-history")
            condition = len(pd.read_html(url)[0]) > 5
        except:
            return False

        return condition


    def _get_price_history(self):
        url = f"{self._base_assets_url}/stock_price_history.php?t={self._ticker}"

        response = get_response(url).text

        data = json.loads(re.findall("(?<=\ \=\ )\S+(?=\;)", response)[0])
        df = pd.DataFrame(data, columns=["d", "o", "h", "l", "c", "v"])

        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index(ascending=False)

        df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

        return df


    def _get_charts(self, data):
        if data.empty:
            print("No data to display.")
        else:
            chart = px.area(x=data.index, y=data["Close"])

            axes_layout = {
                            "gridcolor": "rgb(240, 240, 240)",
                            "showspikes": True,
                            "spikecolor": "rgb(120, 120, 240)",
                            "spikemode": "across",
                            "spikesnap": "cursor",
                            "spikedash": "dash",
                            "spikethickness": 0.5
                            }

            chart.update_xaxes(
                axes_layout,
                title_text="Date",
                rangeselector={
                    "buttons": [
                        {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                        {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                        {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                        {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                        {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                        {"label": "MAX", "step": "all"}]})

            chart.update_yaxes(axes_layout, title_text="Price", tickprefix="$")

            chart.update_layout(
                plot_bgcolor="rgb(250, 250, 250)",
                hovermode="x",
                spikedistance=-1,
                hoverdistance=-1,
                showlegend=False,
                title={
                    "text": f"{self._ticker.upper()} Stock Price History",
                    "y": 0.97,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top"})

            chart.update_traces(hovertemplate="Date: %{x}<br>Price: %{y}")

            chart.show();


    def _get_market_cap(self):
        url = f"{self._base_assets_url}/market_cap.php?t={self._ticker}"

        response = get_response(url).text

        data = json.loads(re.findall("(?<=\ \=\ )\S+(?=\;)", response)[0])
        df = pd.DataFrame(data, columns=["d", "v1"])

        df.columns = ["Date", "Market Cap"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index(ascending=True)

        df = df.apply(lambda x: (pd.to_numeric(x, errors="coerce") * 1000))

        return df


    def _get_financials(self, statement, freq=None):
        url = self._get_url(statement)
        response = get_response(f"{url}?freq={freq}")

        if response.status_code == 200:
            lbls = re.findall("(?<=\'\>)\D+(?=\<\\\/a\>)", response.text)
            lbls = [lbl.replace("\\", "") for lbl in lbls]
            
            data = re.findall("(?<=div\>\"\,)[0-9\.\"\:\-\, ]*", response.text)
            data = [json.loads("{"+i+"}") for i in data]

            df = pd.DataFrame(data=data, index=lbls)
            df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
            df.replace({0: np.nan}, inplace=True)
            
            df = df.T
            df.sort_index(inplace=True, ascending=True) # re-order columns to increase from left to right.
            df = df.T

        else:
            df = pd.DataFrame()

        return df


    def _get_statistics(self, key):
        url = self._get_url(key)

        try:
            df = pd.read_html(url, header=1, index_col=0, parse_dates=True)[0]
            df = df.applymap(lambda x: to_float(x))
        except:
            return pd.DataFrame()

        return df


    def _get_items(self, item, freq=None):
        url = self._get_url(item)

        try:
            df = pd.read_html(url, index_col=0, parse_dates=True)[1] if freq == "Q" else pd.read_html(url, index_col=0)[0]

            df.index.name = "Date"
            df.columns = [df.columns[0].split("(")[0]]
            df.replace(r"[\$\,]", "", regex=True, inplace=True)

            df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
        except:
            return pd.DataFrame()

        return df


    def _get_eight_pillars(self):
        if self._is_condition():
            income_statement = self._get_financials(statement="income-statement")
            balance_sheet = self._get_financials(statement="balance-sheet")
            cash_flow = self._get_financials(statement="cash-flow-statement")

            info = self._get_info()

            # Pillar 1: Average 5-Year P/E Ratio < 22.5
            try:
                pe_ratio = self._get_statistics(key="pe-ratio")
                pe_ratio_5y_avg = pe_ratio.iloc[:20, 2].mean()
            except:
                pe_ratio_5y_avg = np.nan

            # Pillar 2: Average 5-Year ROIC > 9%
            sector = info["zacks_x_sector_desc"]

            roic = {}
            if sector == "Finance":
                roic_5y_avg = np.nan
            else:
                for i in range(5):
                    pre_tax_income = income_statement.loc["Pre-Tax Income"][i]

                    try:
                        income_taxes = income_statement.loc["Income Taxes"][i]
                        effective_tax_rate = income_taxes / pre_tax_income
                    except KeyError:
                        effective_tax_rate = 0

                    ebit = income_statement.loc["EBIT"][i]
                    nopat = ebit * (1.0 - effective_tax_rate)

                    try:
                        debt = balance_sheet.loc["Total Current Liabilities"][i:i+2] + balance_sheet.loc["Long Term Debt"][i:i+2]
                    except KeyError:
                        debt = balance_sheet.loc["Total Current Liabilities"][i:i+2]

                    equity = balance_sheet.loc["Share Holder Equity"][i:i+2]
                    non_operating_cash = cash_flow.loc["Cash Flow From Investing Activities"][i:i+2] + cash_flow.loc["Cash Flow From Financial Activities"][i:i+2]

                    invested_capital = debt + equity + non_operating_cash
                    invested_capital_avg = invested_capital.mean()

                    roic[i] = nopat / invested_capital_avg * 100

                roic_5y_avg = np.mean(list(roic.values()))

            # Pillar 3: 5-Year Revenue Growth (Mil)
            revenue = income_statement.loc["Revenue"]
            revenue_growth_5y = revenue[0] - revenue[4]

            # Pillar 4: 5-Year Net Income Growth (Mil)
            net_income = income_statement.loc["Net Income"]
            net_income_growth_5y = net_income[0] - net_income[4]

            # Pillar 5: 5-Year Shares Outstanding Change < 0
            shares_outstanding = income_statement.loc["Shares Outstanding"]
            shares_outstanding_change = (shares_outstanding[0] / shares_outstanding[4] - 1) * 100

            # Free Cash Flow
            try:
                free_cash_flow = cash_flow.loc["Cash Flow From Operating Activities"] - abs(cash_flow.loc["Net Change In Property, Plant, And Equipment"])
            except KeyError:
                free_cash_flow = cash_flow.loc["Cash Flow From Operating Activities"]

            # Pillar 6: 5-Year LTL to FCF < 5
            long_term_liabilities = balance_sheet.loc["Total Long Term Liabilities"][0]
            ltl_to_fcf = long_term_liabilities / free_cash_flow[0]

            # Pillar 7: 5-Year Free Cash Flow Growth (Mil)
            free_cash_flow_growth_5y = free_cash_flow[0] - free_cash_flow[4]

            # Pillar 8: Average 5-Year P/FCF Ratio < 22.5
            price_fcf = self._get_statistics(key="price-fcf")
            price_fcf_5y_avg = price_fcf.iloc[:20, 2].mean()

            pillar_1 = "✔️" if pe_ratio_5y_avg < 22.5 else "❌"
            pillar_2 = "✔️" if roic_5y_avg > 9 else "❌"
            pillar_3 = "✔️" if revenue_growth_5y > 0 else "❌"
            pillar_4 = "✔️" if net_income_growth_5y > 0 else "❌"
            pillar_5 = "✔️" if shares_outstanding_change < 0 else "❌"
            pillar_6 = "✔️" if ltl_to_fcf < 5 else "❌"
            pillar_7 = "✔️" if free_cash_flow_growth_5y > 0 else "❌"
            pillar_8 = "✔️" if price_fcf_5y_avg < 22.5 else "❌"

            df = pd.DataFrame(
                data=[[round(pe_ratio_5y_avg, 2), pillar_1],
                    [round(roic_5y_avg, 2), pillar_2],
                    [round(revenue_growth_5y, 2), pillar_3],
                    [round(net_income_growth_5y, 2), pillar_4],
                    [round(shares_outstanding_change, 2), pillar_5],
                    [round(ltl_to_fcf, 2), pillar_6],
                    [round(free_cash_flow_growth_5y, 2), pillar_7],
                    [round(price_fcf_5y_avg, 2), pillar_8]],
                index=["5-Year P/E Ratio < 22.5",
                    "5-Year ROIC > 9%",
                    "5-Year Revenue Growth (Mil)",
                    "5-Year Net Income Growth (Mil)",
                    "5-Year Shares Outstanding Growth (%)",
                    "5-Year LTL to FCF < 5",
                    "5-Year Free Cash Flow Growth (Mil)",
                    "5-Year Price to FCF < 22.5"],
                columns=["Value", "Mark"]
                )

            return df

        return pd.DataFrame(
                    data=[[np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan]],
                    index=["5-Year P/E Ratio < 22.5",
                           "5-Year ROIC > 9%",
                           "5-Year Revenue Growth (Mil)",
                           "5-Year Net Income Growth (Mil)",
                           "5-Year Shares Outstanding Growth (%)",
                           "5-Year LTL to FCF < 5",
                           "5-Year Free Cash Flow Growth (Mil)",
                           "5-Year Price to FCF < 22.5"],
                    columns=["Value", "Mark"]
                    )


    def _get_intrinsic_value(self, discount_rate):
        if self._is_condition():
            info = self._get_info()

            company_name = info["comp_name_2"]
            sector = info["zacks_x_sector_desc"]
            industry = info["zacks_x_ind_desc"]

            eps_est = float(info["eps_mean_est_fr2"])
            
            try:
                terminal_multiple = float(info["forward_pe_ratio"])
            except TypeError:
                terminal_multiple = float(info["price_cash"])

            last_close = float(info["last_close"])

            income_statement = self._get_financials(statement="income-statement", freq="Q")
            revenue = income_statement.loc["Revenue"]

            growth_est = round(revenue.sort_index().pct_change()[-20:].mean(), 4)

            eps_fwd = {}
            eps_pv = {}
            for i in range(1, self._years + 1):
                eps_fwd[i] = eps_fwd[i - 1] * (1.0 + growth_est) if i > 1 else eps_est
                eps_pv[i] = eps_fwd[i] / (1.0 + discount_rate) ** i

            terminal_value = eps_pv[self._years] * terminal_multiple
            intrinsic_value = round(float(sum(eps_pv.values()) + terminal_value), 2)

            df = pd.DataFrame(
                data=[company_name,
                      sector,
                      industry,
                      eps_est,
                      growth_est,
                      terminal_multiple,
                      discount_rate,
                      last_close,
                      intrinsic_value],
                index=["Company Name",
                       "Sector",
                       "Industry",
                       "EPS Estimate",
                       "Growth Estimate",
                       "Terminal Multiple",
                       "Discount Rate",
                       "Current Price",
                       "Intrinsic Value"],
                columns=[self._ticker]
                )

            return df

        return pd.DataFrame(
                    data=[np.nan,
                          np.nan,
                          np.nan,
                          np.nan,
                          np.nan,
                          np.nan,
                          np.nan,
                          np.nan,
                          np.nan],
                    index=["Company Name",
                           "Sector",
                           "Industry",
                           "EPS Estimate",
                           "Growth Estimate",
                           "Terminal Multiple",
                           "Discount Rate",
                           "Current Price",
                           "Intrinsic Value"],
                    columns=[self._ticker]
                    )
    
## Ticker inherits from SingleBase
class Ticker(SingleBase):
    def __repr__(self):
        return "macrotrends.Ticker object <%s>" % self._ticker

    @property
    def info(self):
        return self._get_info()

    @property
    def price_history(self):
        return self._get_price_history()

    @property
    def chart(self):
        return self._get_charts(self._get_price_history())

    @property
    def market_cap(self):
        return self._get_market_cap()

    @property
    def income_statement_annual(self):
        return self._get_financials(statement="income-statement")

    @property
    def balance_sheet_annual(self):
        return self._get_financials(statement="balance-sheet")

    @property
    def cash_flow_annual(self):
        return self._get_financials(statement="cash-flow-statement")

    @property
    def financial_ratios_annual(self):
        return self._get_financials(statement="financial-ratios")

    @property
    def income_statement_quarterly(self):
        return self._get_financials(statement="income-statement", freq="Q")

    @property
    def balance_sheet_quarterly(self):
        return self._get_financials(statement="balance-sheet", freq="Q")

    @property
    def cash_flow_quarterly(self):
        return self._get_financials(statement="cash-flow-statement", freq="Q")

    @property
    def financial_ratios_quarterly(self):
        return self._get_financials(statement="financial-ratios", freq="Q")

    @property
    def gross_margin(self):
        return self._get_statistics(key="gross-margin")

    @property
    def operating_margin(self):
        return self._get_statistics(key="operating-margin")

    @property
    def ebitda_margin(self):
        return self._get_statistics(key="ebitda-margin")

    @property
    def pre_tax_profit_margin(self):
        return self._get_statistics(key="pre-tax-profit-margin")

    @property
    def net_margin(self):
        return self._get_statistics(key="net-profit-margin")

    @property
    def price_earnings(self):
        return self._get_statistics(key="pe-ratio")

    @property
    def price_sales(self):
        return self._get_statistics(key="price-sales")

    @property
    def price_book(self):
        return self._get_statistics(key="price-book")

    @property
    def price_cashflow(self):
        return self._get_statistics(key="price-fcf")

    @property
    def current_ratio(self):
        return self._get_statistics(key="current-ratio")

    @property
    def quick_ratio(self):
        return self._get_statistics(key="quick-ratio")

    @property
    def debt_equity(self):
        return self._get_statistics(key="debt-equity-ratio")

    @property
    def roe(self):
        return self._get_statistics(key="roe")

    @property
    def roa(self):
        return self._get_statistics(key="roa")

    @property
    def roi(self):
        return self._get_statistics(key="roi")

    @property
    def roe_tangible(self):
        return self._get_statistics(key="return-on-tangible-equity")

    @property
    def eight_pillars(self):
        return self._get_eight_pillars()

    def intrinsic_value(self, discount_rate=0.125):
        return self._get_intrinsic_value(discount_rate)
    
## 
class MultiBase:
    def __init__(self, tickers):
        tickers = tickers if isinstance(tickers, (list, set, tuple)) else tickers

        self._tickers = [ticker.upper() for ticker in tickers]
        self._workers = os.cpu_count() + 4


    def _get_price_history(self, ticker):
        data = Ticker(ticker).price_history["Close"]
        data.rename(ticker, inplace=True)

        return data


    def _get_eight_pillars_values(self, ticker):
        data = Ticker(ticker).eight_pillars["Value"]
        data.rename(ticker, inplace=True)

        return data


    def _get_eight_pillars_marks(self, ticker):
        data = Ticker(ticker).eight_pillars["Mark"]
        data.rename(ticker, inplace=True)

        return data


    def _get_intrinsic_value(self, ticker):
        data = Ticker(ticker).intrinsic_value()

        return data


    def _multiprocessing(self, function):
        chunksize = round(len(self._tickers) / self._workers)
        with ProcessPoolExecutor(self._workers) as executor:
            df = pd.concat(list(executor.map(function, self._tickers, chunksize=chunksize)), axis=1)

        return df
    
class Tickers(MultiBase):
    def __repr__(self):
        return "macrotrends.Tickers object <%s>" % ", ".join(self._tickers)

    @property
    def price_history(self):
        return self._multiprocessing(self._get_price_history).sort_index(ascending=False)

    @property
    def eight_pillars_values(self):
        return self._multiprocessing(self._get_eight_pillars_values)

    @property
    def eight_pillars_marks(self):
        return self._multiprocessing(self._get_eight_pillars_marks)

    @property
    def intrinsic_value(self):
        return self._multiprocessing(self._get_intrinsic_value)
