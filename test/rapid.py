import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Optional


# =========================
# 1. RapidAPI 配置
# =========================

RAPIDAPI_KEY = '11dc6cd96amsh3936588991807edp1ca8d4jsn4790a17ab297'

if not RAPIDAPI_KEY:
    raise ValueError("请先设置环境变量 RAPIDAPI_KEY")

# 常见 API Dojo Seeking Alpha host
# RapidAPI Playground 里一般能看到这个 host
RAPIDAPI_HOST = "seeking-alpha.p.rapidapi.com"

BASE_URL = f"https://{RAPIDAPI_HOST}"

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST,
}

# 这两个 endpoint 是 API Dojo / Seeking Alpha 常见结构
# 如果你的 RapidAPI provider 不一样，只需要改这里
TRANSCRIPT_LIST_ENDPOINT = "/transcripts/v2/list"
TRANSCRIPT_DETAIL_ENDPOINT = "/transcripts/v2/get-details"


# =========================
# 2. 通用请求函数
# =========================

def rapidapi_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    sleep: float = 0.3,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    通用 RapidAPI GET 请求。
    """
    url = BASE_URL + endpoint

    response = requests.get(
        url,
        headers=HEADERS,
        params=params or {},
        timeout=timeout,
    )

    time.sleep(sleep)

    if response.status_code != 200:
        raise RuntimeError(
            f"Request failed: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"Response: {response.text[:1000]}"
        )

    try:
        return response.json()
    except Exception:
        raise RuntimeError(f"返回内容不是 JSON:\n{response.text[:1000]}")


# =========================
# 3. 获取某个股票的 transcript 列表
# =========================

def get_transcript_list(
    symbol: str,
    size: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    """
    获取某个股票的 earnings call transcript 列表。

    常见参数：
    - symbol: 股票代码，例如 AAPL, MSFT, TSLA
    - size: 每页数量
    - number/page: 页码，不同 provider 可能叫 number 或 page
    """
    params = {
        "symbol": symbol.upper(),
        "size": size,
        "number": page,
    }

    return rapidapi_get(TRANSCRIPT_LIST_ENDPOINT, params=params)


# =========================
# 4. 从列表结果中抽取 transcript ID
# =========================

def extract_transcript_items(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    尽量兼容不同 RapidAPI 返回格式。
    常见格式可能是：
    raw["data"] = [...]
    或 raw["data"]["attributes"]
    """
    items = []

    data = raw.get("data", [])

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            item_id = item.get("id")
            attributes = item.get("attributes", {})

            title = attributes.get("title") or item.get("title")
            publish_on = attributes.get("publishOn") or attributes.get("publish_on")
            url = attributes.get("url") or attributes.get("link")

            items.append({
                "id": item_id,
                "title": title,
                "publishOn": publish_on,
                "url": url,
                "raw": item,
            })

    elif isinstance(data, dict):
        # 防止某些 provider 把列表藏在 data["items"] 里
        possible_lists = [
            data.get("items"),
            data.get("articles"),
            data.get("transcripts"),
        ]

        for possible in possible_lists:
            if isinstance(possible, list):
                for item in possible:
                    item_id = item.get("id")
                    title = item.get("title")
                    publish_on = item.get("publishOn") or item.get("publish_on")
                    url = item.get("url") or item.get("link")

                    items.append({
                        "id": item_id,
                        "title": title,
                        "publishOn": publish_on,
                        "url": url,
                        "raw": item,
                    })

    return items


# =========================
# 5. 获取单篇 transcript 详情
# =========================

def get_transcript_detail(transcript_id: str) -> Dict[str, Any]:
    """
    根据 transcript id 获取全文。
    """
    params = {
        "id": transcript_id
    }

    return rapidapi_get(TRANSCRIPT_DETAIL_ENDPOINT, params=params)


# =========================
# 6. 从详情结果中抽取正文
# =========================

def extract_transcript_text(raw: Dict[str, Any]) -> str:
    """
    兼容不同 JSON 结构，尽量抽取正文。
    """
    # 常见结构 1：data.attributes.content
    try:
        content = raw["data"]["attributes"].get("content")
        if content:
            return content
    except Exception:
        pass

    # 常见结构 2：data.attributes.body
    try:
        body = raw["data"]["attributes"].get("body")
        if body:
            return body
    except Exception:
        pass

    # 常见结构 3：data.content
    try:
        content = raw["data"].get("content")
        if content:
            return content
    except Exception:
        pass

    # 常见结构 4：content/body/transcript 在顶层
    for key in ["content", "body", "transcript", "text"]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # 实在找不到，就保存原始 JSON，方便你检查
    return ""


# =========================
# 7. 下载单个股票的 transcripts
# =========================

def download_symbol_transcripts(
    symbol: str,
    max_pages: int = 3,
    page_size: int = 20,
    output_dir: str = "seeking_alpha_transcripts",
) -> pd.DataFrame:
    """
    下载某个股票多个页面的 earnings call transcripts。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for page in range(1, max_pages + 1):
        print(f"\nFetching {symbol} transcript list page {page}...")

        list_raw = get_transcript_list(
            symbol=symbol,
            size=page_size,
            page=page,
        )

        items = extract_transcript_items(list_raw)

        if not items:
            print(f"No transcript items found on page {page}.")
            break

        for item in tqdm(items, desc=f"{symbol} page {page}"):
            transcript_id = item.get("id")

            if not transcript_id:
                continue

            try:
                detail_raw = get_transcript_detail(transcript_id)
                text = extract_transcript_text(detail_raw)

                row = {
                    "symbol": symbol.upper(),
                    "transcript_id": transcript_id,
                    "title": item.get("title"),
                    "publishOn": item.get("publishOn"),
                    "url": item.get("url"),
                    "text": text,
                    "text_length": len(text) if text else 0,
                    "detail_raw": json.dumps(detail_raw, ensure_ascii=False),
                }

                all_rows.append(row)

            except Exception as e:
                print(f"Failed transcript id {transcript_id}: {e}")

    df = pd.DataFrame(all_rows)

    output_path = os.path.join(output_dir, f"{symbol.upper()}_transcripts.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path}")
    print(f"Total transcripts: {len(df)}")

    return df


# =========================
# 8. 批量下载多个股票
# =========================

def download_many_symbols(
    symbols: List[str],
    max_pages: int = 3,
    page_size: int = 20,
    output_dir: str = "seeking_alpha_transcripts",
) -> pd.DataFrame:
    """
    批量下载多个股票的 transcripts。
    """
    all_dfs = []

    for symbol in symbols:
        try:
            df = download_symbol_transcripts(
                symbol=symbol,
                max_pages=max_pages,
                page_size=page_size,
                output_dir=output_dir,
            )
            all_dfs.append(df)

        except Exception as e:
            print(f"Failed symbol {symbol}: {e}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    final_path = os.path.join(output_dir, "all_transcripts.csv")
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    print(f"\nAll saved: {final_path}")

    return final_df


# =========================
# 9. 主程序示例
# =========================

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "TSLA"]

    df = download_many_symbols(
        symbols=symbols,
        max_pages=2,
        page_size=10,
        output_dir="seeking_alpha_transcripts",
    )

    print(df[["symbol", "title", "publishOn", "text_length"]].head())