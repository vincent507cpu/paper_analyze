import re
import time
import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def fetch_latest_cs_papers(target_count=100, batch_size=200):
    """
    从 arXiv 的 HTML 页面获取最新的计算机科学（CS）领域论文信息。

    参数：
        target_count (int): 目标获取的论文数量。
        batch_size (int): 每次请求的论文数量。

    返回：
        List[dict]: 包含论文元信息的字典列表。
    """
    base_url = "https://arxiv.org/list/cs/recent?skip=0&show=2000"
    papers = []
    skip = 0

    while len(papers) < target_count:
        params = {"skip": skip, "show": batch_size}

        try:
            # 请求 HTML 页面
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            # 解析 HTML 页面
            soup = BeautifulSoup(response.content, "html.parser")
            titles = soup.find_all("div", class_="list-title mathjax")
            authors = soup.find_all("div", class_="list-authors")
            categories = soup.find_all("div", class_="list-subjects")
            links = soup.find_all("a", title="Abstract")

            # 确保字段长度一致
            min_length = min(len(titles), len(authors), len(categories), len(links))
            if min_length == 0:
                print("未找到有效的论文信息，检查页面结构或请求参数。")
                break

            for i in tqdm(range(min_length)):
                if len(papers) >= target_count:
                    break

                # 提取标题、作者和分类信息
                title = titles[i].text.replace("Title:", "").strip()
                author_text = authors[i].text.replace("Authors:", "").strip()
                category_text = categories[i].text.strip()

                # 提取主分类和次要分类
                categories_list = [
                    re.findall(r"([a-z]{2}\.[A-Z]{2})", c.strip())[0] if re.findall(r"([a-z]{2}\.[A-Z]{2})", c.strip()) else "Unknown"
                    for c in category_text.split(";")
                ]
                # primary_category = categories_list[0]
                # secondary_categories = ", ".join(categories_list[1:])

                # 提取 arXiv ID 和摘要链接
                arxiv_id = links[i]["href"].split("/")[-1]

                # 获取摘要和日期
                abstract, date = fetch_abstract_and_date(arxiv_id)

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": author_text,
                    # "primary_category": primary_category,
                    # "secondary_categories": secondary_categories,
                    "category": ','.join(categories_list),
                    "abstract": abstract,
                    "date": date
                })

            skip += batch_size
            time.sleep(1)  # 避免请求频率过高

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            break

    return papers

def fetch_abstract_and_date(arxiv_id):
    """
    获取单篇论文的摘要和发表日期。

    参数：
        arxiv_id (str): 论文的 arXiv ID。

    返回：
        Tuple[str, str]: (摘要内容, 格式化的日期)，若无法提取则返回 ('No abstract available.', 'Unknown date').
    """
    paper_url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        response = requests.get(paper_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        abstract_tag = soup.find("blockquote", class_="abstract")
        date_tag = soup.find("div", class_="dateline")

        # 提取摘要
        abstract = abstract_tag.text.replace("Abstract:", "").strip() if abstract_tag else "No abstract available."

        # 提取日期并格式化
        raw_date = date_tag.text.replace("Submitted on", "").strip() if date_tag else "Unknown date"
        formatted_date = format_date(raw_date)

        print(f"获取 {arxiv_id} 元信息成功")
        return abstract, formatted_date

    except requests.exceptions.RequestException as e:
        print(f"获取 {arxiv_id} 元信息失败: {e}")
        return "No abstract available.", "Unknown date"


def format_date(raw_date):
    """
    格式化日期为标准格式 (YYYY-MM-DD)。

    参数：
        raw_date (str): 原始日期字符串。

    返回：
        str: 格式化的日期，若无法解析，则返回 'Unknown date'.
    """
    try:
        match = re.search(r"(\d{1,2} \w+ \d{4})", raw_date)
        if match:
            date_obj = datetime.datetime.strptime(match.group(1), "%d %b %Y")
            return date_obj.strftime("%Y-%m-%d")
        else:
            return "Unknown date"
    except ValueError:
        return "Unknown date"
    
def save_to_csv(papers, output_path):
    """
    将论文元信息保存到 CSV 文件。

    参数：
        papers (List[dict]): 包含论文元信息的字典列表。
        output_path (str): 输出 CSV 文件路径。
    """
    df = pd.DataFrame(papers)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"论文信息已保存到 {output_path}")


if __name__ == "__main__":
    max_results = 1000  # 目标获取的论文数量
    batch_size = 1100  # 每次请求的论文数量

    print("正在获取最新的 CS 领域论文...")
    papers = fetch_latest_cs_papers(target_count=max_results, batch_size=batch_size)

    output_file = Path(__file__).parent.parent.absolute().joinpath('data/latest_cs_papers_metadata.csv')
    save_to_csv(papers, output_file)
