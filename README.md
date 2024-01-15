# 论文检索器

目标：制作一个论文检索器，实现：
1. 根据摘要发现相关论文，并自动下载；
2. (TODO)理解有关论文，回答问题。

因爬取/下载论文耗时较长，本 pipeline 分为三部分。
# 1. 论文下载器
```python
python arxiv_paper_parser.py -y yy -m mm -p path -s s -k k --download_papers -t *t
```
- `-y`, `--year`：年份（两位数字）
- `-m`, `--month`：月份（两位数字）
- `-p`, `--save_dir`：文件保存路径
- `-s`, `--show`：每次爬取的论文数
- `-k`, `--skip`：论文检索从第几篇开始
- `--download_papers`：下载论文原文（`--no-download_papers` 为不下载论文原文）
- `-t`, `--target_category`：爬取的论文目标分类

已经下载好的论文（包含 'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG' 四类）上传到 https://github.com/vincent507cpu/arXiv_paper_collection，可随意使用。