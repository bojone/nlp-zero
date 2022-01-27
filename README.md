# nlp-zero
基于最小熵原理的NLP工具包

https://kexue.fm/archives/5597

可以直接通过
```
pip install nlp-zero==0.1.6
```

进行安装。整个库纯Python实现，没有第三方调用，支持Python2.x和3.x。
## 使用

### 默认分词
库内带了一个词典，可以作为一个简单的分词工具用

```
from nlp_zero import *

t = Tokenizer()
t.tokenize(u'扫描二维码，关注公众号')
```

自带的词典加入了一些通过新词发现挖掘出来的新词，并且经过笔者的人工优化，质量相对来说还是比较高的。

### 词库构建
通过大量的原始语料来构建词库。

首先我们需要写一个迭代容器，这样就不用一次性把所有语料加载到内存中了。迭代器的写法很灵活，比如我的数据存在MongoDB中，那就是：

```
import pymongo
db = pymongo.MongoClient().weixin.text_articles

class D:
    def __iter__(self):
        for i in db.find().limit(10000):
            yield i['text']
```
如果数据存在文本文件中，大概就是

```
class D:
    def __iter__(self):
        with open('text.txt') as f:
            for l in f:
                yield l.strip() # python2.x还需要转编码
```
然后就可以执行了

```
from nlp_zero import *
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

f = Word_Finder(min_proba=1e-8)
f.train(D()) # 统计互信息
f.find(D()) # 构建词库
```

通过Pandas查看结果：

```
import pandas as pd

words = pd.Series(f.words).sort_values(ascending=False)
```

直接用统计出来的词库建立一个分词工具：

```
t = f.export_tokenizer()

t.tokenize(u'今天天气不错')
```

### 句模版构建
跟前面一样，同样要写一个迭代器，这里不再重复。

因为构建句模版是基于词来统计的，因此还需要一个分词函数，可以用自带的分词器，也可以用外部的，比如结巴分词。

```
from nlp_zero import *
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

tokenize = Tokenizer().tokenize # 使用自带的分词工具
# 通过 tokenize = jieba.lcut 可以使用结巴分词

f = Template_Finder(tokenize, window=3)
f.train(D())
f.find(D())
```

通过Pandas查看结果：

```
import pandas as pd

templates = pd.Series(f.templates).sort_values(ascending=False)
idx = [i for i in templates.index if not i.is_trivial()]
templates = templates[idx] # 筛选出非平凡的模版
```

每个模版已经被封装为一个类了。

### 层次分解
基于句模版来进行句子结构解析。

```
from nlp_zero import *

# 建立一个前缀树，并加入模版
# 模版可以通过tuple来加入，
# 也可以直接通过“tire[模版类]=10”这样来加入
trie = XTrie()
trie[(None, u'呢')] = 10
trie[(None, u'可以', None, u'吗')] = 9
trie[(u'我', None)] = 8
trie[(None, u'的', None, u'是', None)] = 7
trie[(None, u'的', None, u'是', None, u'呢')] = 7
trie[(None, u'的', None)] = 12
trie[(None, u'和', None)] = 12

tokenize = Tokenizer().tokenize # 使用自带的分词工具
p = Parser(trie, tokenize) # 建立一个解析器

p.parse(u'鸡蛋可以吃吗') # 对句子进行解析

"""输出：
>>> p.parse(u'鸡蛋可以吃吗')
+---> (鸡蛋)可以(吃)吗
|     +---> 鸡蛋
|     |     +---> 鸡蛋
|     +---> 可以
|     +---> 吃
|     |     +---> 吃
|     +---> 吗
"""
```
为了方便对结果进行调用以及可视化，输出结果已经被封装为一个SentTree类。这个类有三个属性：template（当前主模版）、content（当前主模版覆盖的字符串）、modules（语义块的list，每个语义块也是用SentTree来描述）。总的来说，就是按照《最小熵原理（三）：“飞象过河”之句模版和语言结构》一文中我们对语言结构的假设来设计的。

## 待续
如果有必要，请阅读源码寻求答案～后续有更新会继续在这里演示。

转载到请包括本文地址：https://kexue.fm/archives/5597

更详细的转载事宜请参考：《科学空间FAQ》
