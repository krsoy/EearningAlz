import os
import re
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


# =========================
# 你必须改成你自己的信息
# SEC 要求声明 User-Agent
# 格式建议：公司/个人名 + 邮箱
# =========================
USER_AGENT = "cluo25@aau.student.dk"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

DATA_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}


class SECFilingDownloader:
    def __init__(self, user_agent: str, sleep_time: float = 0.15):
        """
        sleep_time=0.15 大约每秒 6-7 个请求，低于 SEC 10 requests/second 限制。
        """
        self.user_agent = user_agent
        self.sleep_time = sleep_time

        self.headers_sec = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }

        self.headers_data = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }

    def _get_json(self, url: str, data_sec: bool = True):
        time.sleep(self.sleep_time)
        headers = self.headers_data if data_sec else self.headers_sec
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_text(self, url: str, data_sec: bool = False):
        time.sleep(self.sleep_time)
        headers = self.headers_data if data_sec else self.headers_sec
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    def get_company_tickers(self) -> pd.DataFrame:
        """
        获取 SEC 官方 ticker -> CIK 映射表
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get_json(url, data_sec=False)

        rows = []
        for _, item in data.items():
            rows.append({
                "ticker": item["ticker"].upper(),
                "cik": int(item["cik_str"]),
                "title": item["title"]
            })

        return pd.DataFrame(rows)

    def ticker_to_cik(self, ticker: str) -> str:
        """
        股票代码转 10 位 CIK
        例如 AAPL -> 0000320193
        """
        ticker = ticker.upper()
        df = self.get_company_tickers()

        matched = df[df["ticker"] == ticker]

        if matched.empty:
            raise ValueError(f"找不到 ticker: {ticker}")

        cik_int = int(matched.iloc[0]["cik"])
        return str(cik_int).zfill(10)

    def get_submissions(self, ticker: str = None, cik: str = None) -> dict:
        """
        获取公司 filing history
        """
        if cik is None:
            if ticker is None:
                raise ValueError("ticker 和 cik 至少提供一个")
            cik = self.ticker_to_cik(ticker)

        cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        return self._get_json(url, data_sec=True)

    def filings_to_dataframe(self, submissions: dict) -> pd.DataFrame:
        """
        把 submissions JSON 转成 DataFrame
        """
        recent = submissions["filings"]["recent"]

        df = pd.DataFrame({
            "accessionNumber": recent["accessionNumber"],
            "filingDate": recent["filingDate"],
            "reportDate": recent["reportDate"],
            "acceptanceDateTime": recent["acceptanceDateTime"],
            "act": recent["act"],
            "form": recent["form"],
            "fileNumber": recent["fileNumber"],
            "filmNumber": recent["filmNumber"],
            "items": recent["items"],
            "size": recent["size"],
            "isXBRL": recent["isXBRL"],
            "isInlineXBRL": recent["isInlineXBRL"],
            "primaryDocument": recent["primaryDocument"],
            "primaryDocDescription": recent["primaryDocDescription"],
        })

        return df

    def filter_filings(
        self,
        df: pd.DataFrame,
        forms=("10-Q", "10-K"),
        start_date=None,
        end_date=None,
        limit=None
    ) -> pd.DataFrame:
        """
        过滤 10-Q / 10-K / 8-K 等
        """
        out = df[df["form"].isin(forms)].copy()

        out["filingDate"] = pd.to_datetime(out["filingDate"])

        if start_date:
            out = out[out["filingDate"] >= pd.to_datetime(start_date)]

        if end_date:
            out = out[out["filingDate"] <= pd.to_datetime(end_date)]

        out = out.sort_values("filingDate", ascending=False)

        if limit:
            out = out.head(limit)

        return out

    @staticmethod
    def clean_filename(name: str) -> str:
        name = re.sub(r"[^\w\-.]+", "_", name)
        return name[:180]

    def build_filing_url(self, cik: str, accession_number: str, primary_document: str) -> str:
        """
        SEC filing URL 格式：
        https://www.sec.gov/Archives/edgar/data/{CIK without leading zeros}/{accession without dashes}/{primary_document}
        """
        cik_no_zero = str(int(cik))
        accession_no_dash = accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dash}/{primary_document}"

    def build_complete_submission_txt_url(self, cik: str, accession_number: str) -> str:
        """
        完整 submission txt 文件 URL
        """
        cik_no_zero = str(int(cik))
        accession_no_dash = accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dash}/{accession_number}.txt"

    def html_to_plain_text(self, html: str) -> str:
        """
        简单把 HTML 财报转为纯文本
        """
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        return "\n".join(lines)

    def download_filings(
        self,
        ticker: str,
        forms=("10-Q", "10-K"),
        start_date=None,
        end_date=None,
        limit=10,
        output_dir="sec_filings",
        save_html=True,
        save_text=True,
        save_complete_submission_txt=False
    ) -> pd.DataFrame:
        """
        下载指定公司的 filing 文件

        参数：
        ticker: 股票代码，例如 AAPL, MSFT, NVDA
        forms: 要下载的表格类型，例如 ("10-Q", "10-K")
        start_date: 开始日期，例如 "2020-01-01"
        end_date: 结束日期，例如 "2025-12-31"
        limit: 最多下载多少份
        output_dir: 保存目录
        save_html: 保存主文档 HTML
        save_text: 保存纯文本 TXT
        save_complete_submission_txt: 是否保存完整 submission txt
        """

        ticker = ticker.upper()
        cik = self.ticker_to_cik(ticker)
        submissions = self.get_submissions(cik=cik)
        company_name = submissions.get("name", ticker)

        df_all = self.filings_to_dataframe(submissions)

        df = self.filter_filings(
            df_all,
            forms=forms,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        if df.empty:
            print("没有找到符合条件的 filing")
            return df

        company_dir = os.path.join(output_dir, f"{ticker}_{cik}")
        os.makedirs(company_dir, exist_ok=True)

        downloaded_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Downloading {ticker} filings"):
            accession = row["accessionNumber"]
            filing_date = row["filingDate"].strftime("%Y-%m-%d")
            form = row["form"]
            primary_doc = row["primaryDocument"]

            filing_url = self.build_filing_url(cik, accession, primary_doc)
            complete_txt_url = self.build_complete_submission_txt_url(cik, accession)

            base_name = self.clean_filename(
                f"{ticker}_{form}_{filing_date}_{accession}_{primary_doc}"
            )

            html_path = os.path.join(company_dir, base_name + ".html")
            text_path = os.path.join(company_dir, base_name + ".txt")
            complete_path = os.path.join(company_dir, base_name + "_complete_submission.txt")

            try:
                html = self._get_text(filing_url, data_sec=False)

                if save_html:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)

                if save_text:
                    plain_text = self.html_to_plain_text(html)
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(plain_text)

                if save_complete_submission_txt:
                    complete_txt = self._get_text(complete_txt_url, data_sec=False)
                    with open(complete_path, "w", encoding="utf-8") as f:
                        f.write(complete_txt)

                downloaded_rows.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik,
                    "form": form,
                    "filingDate": filing_date,
                    "reportDate": row["reportDate"],
                    "accessionNumber": accession,
                    "primaryDocument": primary_doc,
                    "filing_url": filing_url,
                    "complete_submission_txt_url": complete_txt_url,
                    "html_path": html_path if save_html else None,
                    "text_path": text_path if save_text else None,
                    "complete_submission_txt_path": complete_path if save_complete_submission_txt else None
                })

            except Exception as e:
                print(f"下载失败: {ticker} {form} {filing_date} {accession}")
                print(e)

        result_df = pd.DataFrame(downloaded_rows)

        metadata_path = os.path.join(company_dir, f"{ticker}_filings_metadata.csv")
        result_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

        print(f"\n完成。文件保存在: {company_dir}")
        print(f"metadata 保存为: {metadata_path}")

        return result_df

    def get_company_facts(self, ticker: str = None, cik: str = None) -> dict:
        """
        获取公司 XBRL 财务指标数据。
        例如 Revenue, NetIncomeLoss, Assets, Liabilities 等。
        """
        if cik is None:
            if ticker is None:
                raise ValueError("ticker 和 cik 至少提供一个")
            cik = self.ticker_to_cik(ticker)

        cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        return self._get_json(url, data_sec=True)

    def extract_us_gaap_fact(
        self,
        company_facts: dict,
        concept: str,
        unit: str = "USD"
    ) -> pd.DataFrame:
        """
        从 companyfacts 里提取某个 US-GAAP 指标。
        例如：
        concept="Revenues"
        concept="NetIncomeLoss"
        concept="Assets"
        """
        facts = company_facts.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        if concept not in us_gaap:
            raise ValueError(f"找不到 concept: {concept}")

        concept_data = us_gaap[concept]
        units = concept_data.get("units", {})

        if unit not in units:
            raise ValueError(f"{concept} 没有单位 {unit}，可用单位: {list(units.keys())}")

        df = pd.DataFrame(units[unit])

        cols = [
            "fy", "fp", "form", "filed", "start", "end",
            "val", "accn", "frame"
        ]

        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols].copy()

        if "filed" in df.columns:
            df["filed"] = pd.to_datetime(df["filed"])

        if "end" in df.columns:
            df["end"] = pd.to_datetime(df["end"])

        df = df.sort_values(["end", "filed"], ascending=[False, False])

        return df


if __name__ == "__main__":
    downloader = SECFilingDownloader(
        user_agent=USER_AGENT,
        sleep_time=0.15
    )

    # =========================
    # 示例 1：下载 Apple 最近 8 份 10-Q 和 10-K
    # =========================
    df_downloaded = downloader.download_filings(
        ticker="AAPL",
        forms=("10-Q", "10-K"),
        start_date="2020-01-01",
        end_date=None,
        limit=8,
        output_dir="sec_filings",
        save_html=True,
        save_text=True,
        save_complete_submission_txt=False
    )

    print(df_downloaded)

    # =========================
    # 示例 2：下载 Nvidia 最近 10 份 10-Q
    # =========================
    # df_nvda_10q = downloader.download_filings(
    #     ticker="NVDA",
    #     forms=("10-Q",),
    #     limit=10,
    #     output_dir="sec_filings"
    # )

    # =========================
    # 示例 3：获取 XBRL 财务指标
    # =========================
    facts = downloader.get_company_facts(ticker="AAPL")

    # Apple 营收
    try:
        revenue_df = downloader.extract_us_gaap_fact(
            company_facts=facts,
            concept="Revenues",
            unit="USD"
        )

        print("\nRevenue:")
        print(revenue_df.head(20))

        revenue_df.to_csv(
            "AAPL_Revenues.csv",
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:
        print("提取 Revenues 失败:", e)

    # Apple 净利润
    try:
        net_income_df = downloader.extract_us_gaap_fact(
            company_facts=facts,
            concept="NetIncomeLoss",
            unit="USD"
        )

        print("\nNet Income:")
        print(net_income_df.head(20))

        net_income_df.to_csv(
            "AAPL_NetIncomeLoss.csv",
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:
        print("提取 NetIncomeLoss 失败:", e)