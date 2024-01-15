import io
import time
import requests
import os
import re
import argparse

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from lxml import html

from config import Config
from utils.parser import PaperRetrievor

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', default=None, required=False, help='年份的后两位', type=str)
parser.add_argument('-m', '--month', default=None, required=False, help='两位月份', type=str)
parser.add_argument('-p', '--save_dir', default=None, required=False, type=str)
parser.add_argument('-s', '--show', default=None, required=False, type=int)
parser.add_argument('-k', '--skip', default=None, required=False, type=int)
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
parser.add_argument('--download_papers', action=argparse.BooleanOptionalAction)
# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
parser.add_argument('-t', '--target_category', default=None, required=False, nargs='+')

args = parser.parse_args()
year = args.year or Config.target_year
month = args.month or Config.target_month
save_dir = args.save_dir or Config.save_dir
target_category = args.target_category or Config.target_category
show = args.show if isinstance(args.show, int) else Config.show
skip = args.skip if isinstance(args.skip, int) else Config.skip
download_papers = args.download_papers if args.download_papers in [True, False] else Config.download_papers

assert len(year) == 2
assert len(month) == 2


if __name__=='__main__':
    if isinstance(target_category, str):
        print(f'开始爬取 {year} 年 {month} 月 {target_category} 的论文...')
    elif isinstance(target_category, list):
        print(f'开始爬取 {year} 年 {month} 月 {"、".join(target_category)} 的论文...')
    retrievor = PaperRetrievor(save_dir, skip, show, target_category)
    retrievor.collect_paper_info(year, month, download_papers)
    time.sleep(100)