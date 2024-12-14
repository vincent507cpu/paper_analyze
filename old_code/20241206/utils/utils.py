import os
import re
import codecs




# stopwords_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                          'stopwords.txt')
stopwords = set()
# fr = codecs.open(stopwords_path, 'r', 'utf-8')
# for word in fr:
#     stopwords.add(word.strip())
# fr.close()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt'),
          encoding='utf-8') as f:
    for line in f.readlines():
        stopwords.add(line.strip())
        
def has_chn(string):
    re_zh = re.compile('([\u4E00-\u9FA5]+)')
    return re_zh.search(string)

def filter_stop(words):
    return [w for w in words if w not in stopwords]

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences

# def logging()