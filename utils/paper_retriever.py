import os
import io
import time
import re

from string import punctuation

import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from lxml import html

def create_folder(path):
    if not os.path.exists(path):
        print(f'\t创建 {path} 文件夹')
        os.mkdir(path)
    
    
def download_pdf(save_dir, paper_title, file_name, pdf_url):
    """下载 PDF

    Args:
        save_path (_type_): _description_
        paper_title (_type_): _description_
        pdf_url (_type_): _description_
    """
    response = requests.get(pdf_url)
    bytes_io = io.BytesIO(response.content)
    
    with open(os.path.join(save_dir, f"{file_name}"), mode='wb') as f:
        f.write(bytes_io.getvalue())
        print(f'\t{paper_title}.PDF 下载成功！')
        
def generate_file_name(address, title):
    title = re.sub(f"[{punctuation}]+", ' ', title)
    title = re.sub('\s+', ' ', title)
    return address + ' ' + title
    
class PaperRetrievor:
    """论文下载器。
    
    工作流程：
    1. 遍历所有页，检验每一篇文献是否包含目标类别
        - 是：爬取论文信息
        - 不是：跳过
    2. 保存论文信息到一个 csv 文件
    3. 根据 csv 文件里的信息下载论文
    """
    def __init__(self, save_dir) -> None:
        self.save_dir = save_dir
        
        create_folder(save_dir)
        
    def collect_paper_info(self, year, month, skip, show, target_category, download_papers=False):
        query = 'https://arxiv.org/list/cs/{}?skip={}&show={}'
        cache = {
            'title':[],
            'address':[],
            'category':[],
            'authors':[],
            'abstract':[],
            'comment':[],
            'file_name':[]
        }
        stored_paper = set()

        start_time = time.time()
        response = requests.get(query.format(year + month, skip, show))
        tree = html.fromstring(response.content)
        target_value_xpath = '//*[@id="dlpage"]/small[1]'
        result = tree.xpath(target_value_xpath)[0].text_content().strip()
        num_entries = int(re.search('total of (\d+)', result).group(1))
        print(f'{year} 年 {month} 月一共有 {num_entries} 篇论文，开始判断是否与目标类别相关...')
        
        for i in range(1, num_entries  // show + 2):
            start_time_ = time.time()
            if i != 1:
                response = requests.get(query.format(year + month, skip, show))
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f'第 {i} / {num_entries  // show + 1} 个页面的论文信息获取成功，解析中...')
            
            titles = soup.find_all('div', class_="list-title mathjax")
            titles = [t.select_one('span.descriptor').next_sibling.strip() for t in titles]
            titles = [t.replace('  ', ' ') for t in titles]
            
            addresses = soup.find_all('dt')
            addresses = [a.find_all('a')[2].attrs['href'].split('/')[-1] for a in addresses]
            
            categories = soup.find_all('span', class_='primary-subject')
            categories = [c.contents[0] for c in categories]
            categories = [re.search(r'\((.+?)\)', c).group(1) for c in categories]
            
            for title, address, cat in tqdm(zip(titles, addresses, categories)):
                if cat != target_category:
                    continue
                if address in stored_paper:
                    continue
                try:
                    file_name = generate_file_name(address, title)
                    authors, abstract, comment = self.fetch_one_paper(year, month, title, cat, address, file_name, download_papers)
                    cache['title'].append(title)
                    cache['address'].append(address)
                    cache['category'].append(cat)
                    cache['authors'].append(authors)
                    cache['abstract'].append(abstract)
                    cache['comment'].append(comment)
                    cache['file_name'].append(file_name+'.pdf')

                    stored_paper.add(address)
                except:
                    print(f'\t{address} 论文爬取出错')
                
            skip += show
            print(f'第 {i} / {num_entries  // show + 1} 页爬取完毕，耗时 {(time.time() - start_time_)/60:.2f} min')
            
        df = pd.DataFrame(cache)
        df.to_csv(os.path.join(self.save_dir, year + month + ' ' + target_category + '.csv'), index=False)
        
        print(f'{year} 年 {month} 月论文爬取 & 下载完成，共获取 {df.shape[0]} 篇论文 ，共耗时 {(time.time() - start_time)/60:.2f} min')
    
    
    def fetch_one_paper(self, year, month, title, cat, address, file_name, download_papers):
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
            abstract = soup.find('span', class_='descriptor', string='Abstract:').next_sibling.strip()
        else:
            abstract = ''
        
        if soup.find('td', class_='tablecell comments mathjax'):
            comment = soup.find('td', class_='tablecell comments mathjax').string
        else:
            comment = ''

        if download_papers:
            create_folder(os.path.join(self.save_dir, year + month + ' ' + cat))
            if not os.path.exists(os.path.join(self.save_dir, year + month + ' ' + cat, file_name + '.pdf')):
                download_pdf(os.path.join(self.save_dir, year + month + ' ' + cat), address + ' ' + title, file_name + '.pdf', f'https://arxiv.org/pdf/{address}.pdf')
        return authors, abstract, comment