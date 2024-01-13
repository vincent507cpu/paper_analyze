import os
import io
import time
import re

import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from lxml import html

def create_folder(path):
    if not os.path.exists(path):
        print(f'创建 {path} 文件夹')
        os.mkdir(path)
    
    
def download_pdf(save_dir, paper_title, pdf_url):
    """下载 PDF

    Args:
        save_path (_type_): _description_
        paper_title (_type_): _description_
        pdf_url (_type_): _description_
    """
    if not pdf_url:
        return
    response = requests.get(pdf_url)
    bytes_io = io.BytesIO(response.content)
    with open(os.path.join(save_dir, f"{paper_title}.pdf"), mode='wb') as f:
        f.write(bytes_io.getvalue())
        print(f'{paper_title}.PDF 下载成功！')
        
        
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
        self.month = month
        self.year = year
        self.save_dir = os.path.join(save_dir, self.year + self.month)
        self.skip = skip
        self.show = show
        self.target_category = target_category

        create_folder(save_dir)
        create_folder(self.save_dir)
        
        
    def collect_paper_info(self, download_papers=False):
        start_time = time.time()
        titles, addresses, categories = [], [], []
        authors, abstracts, comments = [], [], []
        
        response = requests.get(self.query.format(self.date, self.skip, self.show))
        tree = html.fromstring(response.content)
        target_value_xpath = '//*[@id="dlpage"]/small[1]'
        result = tree.xpath(target_value_xpath)[0].text_content().strip()
        num_entries = int(re.search('total of (\d+)', result).group(1))
        print(f'{self.year} 年 {self.month} 月一共有 {num_entries} 篇论文，开始判断是否与目标类别相关...')
        
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
            
        df = pd.DataFrame({'title':titles, 'index':addresses, 'category': categories, 'authors':authors, 'abstract':abstracts, 'comment':comments})
        df.to_csv(os.path.join(self.save_dir, self.date+'.csv'), index=False)
        
        print(f'{self.year} 年 {self.month} 月论文爬取 & 下载完成，共获取 {df.shape[0]} 篇论文 ，共耗时 {(time.time() - start_time)/60:.2f} min')
    
    
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
            download_pdf(address + ' ' + title, f'https://arxiv.org/pdf/{address}.pdf')
        return authors, abstract, comment