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

from utils.parser import PaperRetrievor

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', default='23', required=False, help='年份的后两位', type=str)
parser.add_argument('-m', '--month', default='01', required=False, help='两位月份', type=str)
parser.add_argument('-r', '--save_dir', default='fetched_paper_info', required=False)
parser.add_argument('-t', '--target_category', default=['cs.AI', 'cs.CL', 'cs.IR', 'cs.LG', 'cs.NE'], required=False, type=list)
parser.add_argument('-s', '--show', default=2000, required=False, type=int)
parser.add_argument('-k', '--skip', default=0, required=False, type=int)
parser.add_argument('-d', '--download_papers', default=False, required=False, type=bool)

args = parser.parse_args()
year = args.year
month = args.month
save_dir = args.save_dir
show = args.show
target_category = args.target_category
skip = args.skip
download_papers = args.download_papers

assert len(year) == 2
assert len(month) == 2


if __name__=='__main__':
    print(f'开始爬取 {year} 年 {month} 月的论文...')
    retrievor = PaperRetrievor(year, month, save_dir, skip, show, target_category)
    retrievor.collect_paper_info(download_papers)
    time.sleep(100)