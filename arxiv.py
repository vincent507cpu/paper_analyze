import io
import time
import requests
import sys
import os
import argparse
from bs4 import BeautifulSoup
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start_date', default='2023-01-01', required=False)
parser.add_argument('-e', '--end_date', default='2023-12-31', required=False)
parser.add_argument('-r', '--save_root', default='fetched_papers', required=False)
parser.add_argument('-t', '--target_category', default=['cs.AI', 'cs.CL', 'cs.IR', 'cs.LG', 'cs.NE'], required=False, type=list)
parser.add_argument('-i', '--size', default=200, required=False, type=int)
parser.add_argument('-a', '--start_index', default=0, required=False, type=int)
args = parser.parse_args()
start_date = args.start_date
end_date = args.end_date
save_root = args.save_root
size = int(args.size)
target_category = args.target_category
start_index = int(args.start_index)


class PaperRetrievor:
    """论文下载器。
    
    工作流程：
    1. 获取源代码块
    2. 校验信息是否完整
        - 是：首先下载论文，然后进行第 3 步；
        - 不是：跳过
    3. 爬取论文信息
    4. 遍历同一页的源代码块
    5. 遍历所有页，直至全部论文遍历完毕
    6. 保存论文信息至 csv
    """
    def __init__(
            self, start_date=start_date, end_date=end_date, save_root=save_root, size=size, target_category=target_category, start_index=start_index
        ) -> None:
        self.query = "https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={}&date-to_date={}&date-date_type=announced_date_first&abstracts=show&size={}&order=announced_date_first&start={}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
            "Connection": "keep-alive",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8"
        }
        self.start_date = start_date
        self.end_date = end_date
        self.save_root = save_root
        self.size = size
        self.target_category = set(target_category)
        self.start_index = start_index
        
    def get_eligeble_snip_w_pdf(self, snippet):
        ...
        
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
        with open(os.path.join(self.save_root, f"{pdf_name}.pdf"), mode='wb') as f:
            f.write(bytes_io.getvalue())
            print(f'{pdf_name}.PDF 下载成功！')
    
    def category_check(self, cats):
        """检查论文类别是否在目标

        Args:
            cats (_type_): _description_

        Returns:
            _type_: _description_
        """
        return bool(set(cats) & self.target_category)
    
    def fetch_one_paper(self, snippet):
        """下载单篇论文，如果没有目标类型直接跳过

        Args:
            snippet (_type_): _description_
            save_root (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            paper_categories = [cat for cat in snippet.find('div', class_='tags').strings if cat != '\n']
            if not self.category_check(paper_categories, target_category):
                return None
            else:
                paper_categories = ','.join(paper_categories)
        except:
            paper_categories = ''
        
        try:
            title = snippet.find('p', class_="title is-5 mathjax").contents[0].strip()
        except:
            title = ''
            
        try:
            web_address = snippet.a['href']
        except:
            return ''
        paper_index = web_address.split('/')[-1]
        publish_time = paper_index.split('.')[0]
        pdf_url = web_address + '.pdf'
        
        self.create_folder(self.save_root)
        self.download_pdf(paper_index + ' ' + title.replace('/', ' '), pdf_url)
        time.sleep(1)
            
        try:
            authors = [a.string.strip() for a in snippet.find('p', class_='authors') if a.string.strip() not in ['', ',']][1:]
            authors = ','.join(authors)
        except:
            authors = ''
            
        try:
            abstract = snippet.find('p', class_="abstract mathjax").find('span', class_="abstract-full has-text-grey-dark mathjax").contents[0].strip()
        except:
            abstract = ''
                
        return title, web_address, publish_time, paper_categories, authors, abstract
    
    
    def fetch_one_page(self, data_dict, start_index, total_num, flag=True):
        """下载单页论文

        Args:
            start_date (_type_): _description_
            end_date (_type_): _description_
            start_index (_type_): _description_
            save_root (_type_): _description_
            total_num (_type_): _description_
            flag (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        full_query = self.query.format(self.start_date, self.end_date, self.size, start_index)
        response = requests.get(full_query, headers=self.headers, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            snips = soup.find_all('li', class_='arxiv-result')
            for snippet in snips:
                result = self.fetch_one_paper(snippet)
                if len(result) == 1:
                    continue
                data_dict['title'].append(result[0])
                data_dict['web_web_address'].append(result[1])
                data_dict['publish_time'].append(result[2])
                data_dict['paper_categories'].append(result[3])
                data_dict['authors'].append(result[4])
                data_dict['abstract'].append(result[5])
                total_num += 1
                if not total_num % 1000:
                    print(f"已经完成 {total_num} 篇论文的信息提取和下载")
        else:
            flag = False
        return data_dict, total_num, flag
    
    def fetch_all(
        self, 
        start_index=0, 
        flag=True
    ):
        start_time = time.time()
        data_dict = {
            'title':[],
            'web_web_address':[],
            'publish_time':[],
            'paper_categories':[],
            'authors':[],
            'abstract':[]
        }
        total_num = 0
        while flag:
            data_dict, fetched_num, flag = self.fetch_one_page(data_dict, start_index, total_num, flag)
            start_index += self.size
            total_num += fetched_num
        print(f'一共爬取 {total_num} 篇文献，共用时 {time.time()-start_time:.2}s')
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(self.save_dir, 'paper_info.csv'))
        
        
if __name__=='__main__':
    retrievor = PaperRetrievor()
    retrievor.fetch_all()