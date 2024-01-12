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

class PaperRetrievor:
    """论文下载器。
    
    工作流程：
    1. 遍历所有页，检验每一篇文献是否包含目标类别
        - 是：爬取论文信息
        - 不是：跳过
    2. 保存论文信息到一个 csv 文件
    3. 根据 csv 文件里的信息下载论文
    """
    def __init__(self, year, month, save_dir, skip, show, target_category) -> None:
        self.query = 'https://arxiv.org/list/cs/{}?skip={}&show={}'
        self.date = year + month
        self.save_dir = os.path.join(save_dir, self.date)
        self.skip = skip
        self.show = show
        self.target_category = target_category
        
        self.create_folder(save_dir)
        self.create_folder(self.save_dir)
        
    def create_folder(self, path):
        if not os.path.exists(path):
            print(f'创建 {path} 文件夹')
            os.mkdir(path)
        
        
    def download_pdf(self, pdf_name, pdf_url):
        """下载 PDF

        Args:
            save_path (_type_): _description_
            pdf_name (_type_): _description_
            pdf_url (_type_): _description_
        """
        if not pdf_url:
            return
        response = requests.get(pdf_url)
        bytes_io = io.BytesIO(response.content)
        with open(os.path.join(self.save_dir, f"{pdf_name}.pdf"), mode='wb') as f:
            f.write(bytes_io.getvalue())
            print(f'{pdf_name}.PDF 下载成功！')
            
            
    def collect_paper_info(self, download_papers=False):
        start_time = time.time()
        titles, addresses, categories = [], [], []
        authors, abstracts, comments = [], [], []
        
        response = requests.get(self.query.format(self.date, self.skip, self.show))
        tree = html.fromstring(response.content)
        target_value_xpath = '//*[@id="dlpage"]/small[1]'
        result = tree.xpath(target_value_xpath)[0].text_content().strip()
        num_entries = int(re.search('total of (\d+)', result).group(1))
        print(f'{year} 年 {month} 月一共有 {num_entries} 篇论文，开始判断是否与目标类别相关...')
        
        for i in range(1, num_entries  // self.show + 2):
            response = requests.get(self.query.format(self.date, self.skip, self.show))
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f'第 {i} / {num_entries  // self.show + 1} 个页面的论文信息获取成功，解析中...')
            
            titles_ = soup.find_all('div', class_="list-title mathjax")
            titles_ = [t.select_one('span.descriptor').next_sibling.strip() for t in titles_]
            titles_ = [t.replace('  ', ' ') for t in titles_]
            titles_ = [t.replace('/', '') for t in titles_]
            
            addresses_ = soup.find_all('dt')
            addresses_ = [a.find_all('a')[2].attrs['href'].split('/')[-1] for a in addresses_]
            
            categories_ = soup.find_all('span', class_='primary-subject')
            categories_ = [c.contents[0] for c in categories_]
            categories_ = [re.search(r'\((.+?)\)', c).group(1) for c in categories_]
            
            for title, address, cat in tqdm(zip(titles_, addresses_, categories_)):
                if cat in self.target_category:
                    authors_, abstract, comment = self.fetch_one_paper(title, address, download_papers)
                    titles.append(title)
                    addresses.append(address)
                    categories.append(cat)
                    authors.append(authors_)
                    abstracts.append(abstract)
                    comments.append(comment)
                
            self.skip += self.show
            print(f'第 {i} / {num_entries  // self.show + 1} 页爬取完毕，耗时 {(time.time() - start_time)/60:.2f} min')
            
        df = pd.DataFrame({'title':titles, 'index':addresses, 'category': categories, 'authors':authors, 'abtract':abstracts, 'comment':comments})
        df.to_csv(os.path.join(self.save_dir, self.date+'.csv'), index=False)
        
        print(f'{year} 年 {month} 月论文爬取 & 下载完成，共获取 {df.shape[0]} 篇论文 ，共耗时 {(time.time() - start_time)/60:.2f} min')
    
    def fetch_one_paper(self, title, address, download_papers):
        web_page = 'https://arxiv.org/abs/{}'.format(address)
        response = requests.get(web_page)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到包含作者信息的<div>标签
        authors_div = soup.find('div', class_='authors')

        # 提取作者信息
        if authors_div:
            # 使用正则表达式提取作者名字
            authors_text = authors_div.get_text(strip=True)
            authors_match = re.search(r'Authors:(.*)', authors_text)
            
            if authors_match:
                authors = [author.strip() for author in authors_match.group(1).split(',')]
                authors = '#'.join(authors)
        else:
            authors = ''        

        if soup.find('span', class_='descriptor', string='Abstract:'):
            abstract = soup.find('span', class_='descriptor', string='Abstract:').next_sibling
        else:
            abstract = ''
        
        if soup.find('td', class_='tablecell comments mathjax'):
            comment = soup.find('td', class_='tablecell comments mathjax').string
        else:
            comment = ''
            
        if download_papers:
            self.download_pdf(address + ' ' + title, f'https://arxiv.org/pdf/{address}.pdf')
        return authors, abstract, comment
    
if __name__=='__main__':
    print(f'开始爬取 {year} 年 {month} 月的论文...')
    retrievor = PaperRetrievor(year, month, save_dir, skip, show, target_category)
    retrievor.collect_paper_info(download_papers)
    time.sleep(100)