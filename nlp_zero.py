#! -*- coding:utf-8 -*-
"""自然语言处理的一些无监督算法

正如它的名字nlp zero一样，我们希望NLP可以从零开始，从大量的
无标签语料中学习中语言的规律，然后才进一步应用到上层任务中。
所以这里封装了一些NLP的无监督学习算法，目前包括词库构建、句
模版构建，并包含对应的识别工具，能基本起到无监督句法分析的
效果。

这些算法的建立在前辈们的一些成果之上，并且由作者进行了大量的
拓展，并且还在不断拓展中。

苏剑林
https://kexue.fm
2018年05月30日
"""

from math import log as math_log
import re, os
import logging
from itertools import chain
from collections import defaultdict, Iterable
from sys import version_info
import os

is_python2 = (version_info.major == 2)

if not is_python2:
    basestring = str

run_path = os.path.split(os.path.realpath(__file__))[0]

def log(x):
    if x == 0:
        return -100
    else:
        return math_log(x)


class Trie:
    """定义基本的Trie树结构，便于存储词典（词+词频）。
    主要的代码量是基于Python类的特殊方法来定义一些函数，
    达到表面上看起来和dict的使用方法基本一致的效果。
    """

    def __init__(self, path_or_trie=None):
        self.dic = {}
        self.end = True
        self.num_items = 0  # 总词数
        self.total_items = 0  # 总词频
        self.log_total_items = log(self.total_items)  # 对数总词频
        if isinstance(path_or_trie, basestring):  # 从文件中加载，文件的每一行是“词 词频
            with open(path_or_trie) as f:
                for l in f:
                    l = re.split(' +', l.strip())
                    if is_python2:
                        self.__setitem__(l[0].decode('utf-8'), int(l[1]))
                    else:
                        self.__setitem__(l[0], int(l[1]))
        elif path_or_trie != None:
            self.update(path_or_trie)

    def __setitem__(self, item, count):
        if count == 0:  # 设置词频为零则相当于删除该词
            return self.__delitem__(item)

        _ = self.dic
        for c in item:
            if c not in _:
                _[c] = {}
            _ = _[c]

        if self.end in _:  # 调整词频
            self.total_items += (count - _[self.end][1])
        else:  # 增加新词
            self.total_items += count
            self.num_items += 1

        _[self.end] = (item, count)
        self.log_total_items = log(self.total_items)  # 更新对数词频

    def __getitem__(self, item):  # 获取指定词的频率，不存在则返回0
        _ = self.dic
        for c in item:
            if c not in _:
                return 0
            _ = _[c]

        return _.get(self.end, ('', 0))[1]

    def __delitem__(self, item):  # 删除某个词
        _ = self.dic
        for c in item:
            if c not in _:
                return None
            _ = _[c]

        if self.end in _:
            self.num_items -= 1
            self.total_items -= _[self.end][1]
            del _[self.end]

    def __iter__(self, _=None):  # 以(词, 词频)的形式逐一返回所有记录
        if _ == None:
            _ = self.dic

        for c in _:
            if c == self.end:
                yield _[self.end]
            else:
                for i in self.__iter__(_[c]):
                    yield i

    def __str__(self):  # 方便调试的显示
        return '<Trie: %s items, %s frequency>' % (self.num_items,
                                                   self.total_items)

    def __repr__(self):
        return self.__str__()

    def search(self, sent):  # 返回字符串中所有能找到的词语
        result = {}  # 结果是{(start, end): (词, 词频)}的字典
        for i, c1 in enumerate(sent):
            _ = self.dic
            for j, c2 in enumerate(sent[i:]):
                if c2 in _:
                    _ = _[c2]
                    if self.end in _:
                        result[i, i + j + 1] = _[self.end]
                else:
                    break

        return result

    def update(self, tire):  # 用一个词典更新当前trie树
        for i, j in tire:
            self.__setitem__(i, j)

    def get_proba(self, w, logit=True):  # 算词频
        _ = self.__getitem__(w)
        if logit:
            return log(_) - self.log_total_items
        else:
            return _ / self.total_items


class DAG:
    """定义一般的有向无环图（Directed Acyclic Graph）对象，
    便于在各种场景下使用。其中optimal_path方法使用viterbi
    算法来给出最优路径。
    """

    def __init__(self, nb_node, null_score=-100):
        self.edges = {}
        self.nb_node = nb_node
        self.null_score = null_score

    def __setitem__(self, start_end, score):  # 构建图上的加权边
        start, end = start_end  # key是(start, end)下标对
        if start not in self.edges:
            self.edges[start] = {}
        self.edges[start][end] = score

    def optimal_path(self):
        """动态规划求最优路径
        result的key是当前字的下标，代表截止到前一字的规划结果，
        result的第一个值是list，表示匹配片段的(start, end)下标对；
        result的第二个值是路径的分数
        """
        result = {0: ([], 1)}
        start = 0  # 当前字的下标
        length = self.nb_node
        while start < length:
            if start in self.edges:  # 如果匹配得上
                for i, j in self.edges[start].items():  # 这里i是终止下标
                    score = result[start][1] + j  # 当前路径分数
                    # 如果当前路径不在result中，或者它的分数超过已有路径，则更新
                    if i not in result or (score > result[i][1]):
                        result[i] = result[start][0] + [(start, i)], score

            # 为了下一步的匹配，如果下一字还不在result中，
            # 就按单字来插入，概率为null_score
            if start + 1 not in result:
                score = result[start][1] + self.null_score
                result[start
                       + 1] = result[start][0] + [(start, start + 1)], score

            start += 1

        return result[self.nb_node][0]

    def _all_paths(self, n):  # all_paths的辅助函数，递归获取从n开始的所有路径
        if n in self.edges:  # 如果成立则意味着n还不是终点
            paths = []
            for m in self.edges[n]:
                paths.extend([[n] + _ for _ in self._all_paths(m - 1)])
        else:  # 意味着n是终点
            paths = [[n]]

        return paths

    def all_paths(self):  # 返回所有连通路径（包括孤立节点）
        ends = set(chain(*self.edges.values()))
        starts = [n for n in range(self.nb_node) if n + 1 not in ends]
        paths = []

        for n in starts:
            paths.extend(self._all_paths(n))

        return paths


class Tokenizer:
    """基于Trie树和动态规划的分词工具，
    仅仅是基于词典和一元模型完成分词，不具备新词识别能力。
    """

    def __init__(self, word_trie=None):
        if word_trie == None:
            self.words = Trie(run_path + '/dic.txt')
        else:
            self.words = word_trie
        self.Aa123 = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    def simple_tokenize(self, sent):  # 利用动态规划完成分词的基本函数
        if not sent:
            return []

        dag = DAG(len(sent))
        for i, j in self.words.search(sent).items():
            dag[i] = log(j[1]) - self.words.log_total_items

        return [sent[i:j] for i, j in dag.optimal_path()]

    def combine_Aa123(self, simple_result):  # 将连续的字母或者数字合并
        result = []
        combine = False
        for w in simple_result:
            if w[-1] in self.Aa123:
                if combine:
                    if len(re.findall('[a-zA-Z\d\.]', w)) == len(w):
                        result[-1] += w
                    else:
                        combine = False
                        result.append(w)
                else:
                    combine = True
                    result.append(w)
            else:
                combine = False
                result.append(w)

        return result

    def tokenize(self, sent, combine_Aa123=True):
        """比较完整的分词函数，包括分词前和分词后的处理。
        分词前：按照标点或者换行符来切分长句，然后逐一分词，能提高效率；
        分词后：合并连续的数字或字母（可选）
        """
        result = []
        start = 0
        for i in re.finditer(u'[。，、！？\n ]+', sent):
            result.extend(self.simple_tokenize(sent[start:i.start()]))
            result.extend(sent[i.start():i.end()])
            start = i.end()

        result.extend(self.simple_tokenize(sent[start:]))
        if combine_Aa123:
            result = self.combine_Aa123(result)

        return result


class Word_Finder:
    """基于切分和去冗余理论的无监督词库构建

    可以通过train函数增量训练，可以自行通过word_finder.words=defaultdict(int)来覆盖
    旧的词表。
    """

    def __init__(self, min_proba=1e-7, min_pmi=1):
        self.min_proba = min_proba  # 最低共现频率
        self.min_pmi = min_pmi  # 最低互信息
        self.words = defaultdict(int)  # 当前词库
        self.chars = defaultdict(int)  # 单字频率
        self.pairs = defaultdict(int)  # 邻字对频率

    def texts(self, sents, filtering=True):
        if filtering:
            for sent in sents:
                yield re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', sent)
        else:
            for sent in sents:
                yield [sent]

    def train(self, sents, filtering=True, remove_weak_pairs=True):
        """训练步骤，也就是统计字频和字对频率。

        【参数含义】
        filtering：是否过滤无关字符，只保留中文、数字和英文字母；
        """
        logger = logging.getLogger(u'统计频数')
        for n, sent in enumerate(self.texts(sents, filtering)):
            if n % 10000 == 0 and n > 0:
                logger.info(u'已统计了 %s 个句子' % n)
            for s in sent:
                for i, c in enumerate(s):
                    self.chars[c] += 1
                    if i < len(s) - 1:
                        self.pairs[s[i:i + 2]] += 1

        logger.info(u'共统计了 %s 个句子' % (n + 1))

    def find(self, sents, filtering=True, remove_weak_pairs=True):
        """基本的新词识别算法

        在train的基础上，计算互信息，通过互信息找出强相关的邻字，然后
        通过这些邻字实现粗糙的分词，从粗糙的分词结果中筛选出可能的词语。

        【参数含义】
        filtering：是否过滤无关字符，只保留中文、数字和英文字母；
        remove_weak_pairs：是否在self.pairs中去掉低频项，以减少低频项的
                          的内存占用，这个参数只影响增量更新。
        """
        log_min_proba = log(self.min_proba)
        log_total_chars = log(sum(self.chars.values()))
        log_total_pairs = log(sum(self.pairs.values()))
        chars = {i: log(j) - log_total_chars for i, j in self.chars.items()}
        pairs = {i: log(j) - log_total_pairs for i, j in self.pairs.items()}
        pairs = {i: j for i, j in pairs.items() if j > log_min_proba}

        if remove_weak_pairs:
            self.pairs = defaultdict(int, {i: j for i, j in self.pairs.items() if i in pairs})

        pairs = {
            i: j
            for i, j in pairs.items()
            if j - chars[i[0]] - chars[i[1]] > self.min_pmi
        }

        logger = logging.getLogger(u'词库构建')
        for n, sent in enumerate(self.texts(sents, filtering)):
            if n % 10000 == 0 and n > 0:
                logger.info(u'已处理了 %s 个句子' % n)
            for s in sent:
                if s:
                    w = s[0]
                    for i, c in enumerate(s[1:]):
                        if s[i:i + 2] in pairs:
                            w += s[i + 1]
                        else:
                            self.words[w] += 1
                            w = s[i + 1]
                    self.words[w] += 1

        logger.info(u'共处理了 %s 个句子' % (n + 1))
        log_total_words = log(sum(self.words.values()))
        self.words.update({
            i: j
            for i, j in self.words.items()
            if log(j) - log_total_words > log_min_proba
        })

        tokenizer = self.export_tokenizer()
        total = len(self.words)
        logger = logging.getLogger(u'词语去冗')

        for n, w in enumerate(sorted(self.words, key=lambda x: -len(x))):
            if n % 100000 == 0 and n > 0:
                logger.info(u'已处理了 %s/%s 个词语' % (n, total))
            if len(w) == 1:
                continue
            num = tokenizer.words[w]
            pmi = tokenizer.words.get_proba(w)
            del tokenizer.words[w]
            w_subwords = tokenizer.tokenize(w)
            for k in w_subwords:
                pmi -= tokenizer.words.get_proba(k)
            if pmi < len(w) - 1:
                for k in w_subwords:
                    tokenizer.words[k] += num
                    self.words[k] += num
                del self.words[w]
            else:
                tokenizer.words[w] = num

    def export_trie(self):
        return Trie(self.words.items())

    def export_tokenizer(self):
        return Tokenizer(self.export_trie())


class Template:
    """封装一个类，作为句模版的容器
    """

    def __init__(self, container=None):
        self.words = []
        self.last = None
        self.length = 0  # 非占位符的词数
        if isinstance(container, Iterable):
            for w in container:
                self.add(w)

    def add(self, value):
        if value != None:
            self.words.append(value)
            self.last = value
            self.length += 1
        elif (not self.words) or self.words[-1] != None:
            self.words.append(value)

    def is_trivial(self):
        """是否平凡模版，单个词且两边都是占位符的模版是平凡的。
        """
        return (len(self.words) == 3 and self.words[0] == None
                and self.words[2] == None)

    def __iter__(self):
        for w in self.words:
            yield w

    def __hash__(self):
        return tuple(self.words).__hash__()

    def __eq__(self, t):
        return self.__hash__() == t.__hash__()

    def __str__(self):
        _ = u'< Template: %s >' % (
            ''.join([i if i else ' [X] ' for i in self.words]))

        if is_python2:
            return _.encode('utf-8')
        else:
            return _

    def __repr__(self):
        return self.__str__()


class Template_Finder:
    """基于 Skip Gram 互信息构建句子模版
    """

    def __init__(self, tokenize, window=5, min_proba=1e-5, min_pmi=2):
        self.tokenize = tokenize  # 自定义分词函数，输入字符串，输出词列表
        self.window = window  # skip gram的统计需要一个窗口大小，以便及时截断
        self.min_proba = min_proba
        self.min_pmi = min_pmi
        self.templates = defaultdict(int)
        self.words = defaultdict(int)
        self.skip_grams = defaultdict(int)

    def texts(self, sents, filtering=True):
        if filtering:
            for sent in sents:
                yield [
                    self.tokenize(s)
                    for s in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', sent) if s
                ]
        else:
            for sent in sents:
                yield [
                    self.tokenize(s) for s in re.split(u'[。，、！？\n ]+', sent)
                    if s
                ]

    def train(self, sents, filtering=True):
        """训练步骤，也就是统计词频和skip gram。

        【参数含义】
        filtering：是否过滤无关字符，只保留中文、数字和英文字母；
        """
        logger = logging.getLogger(u'统计频数')
        for n, sent in enumerate(self.texts(sents, filtering)):
            if n % 10000 == 0 and n > 0:
                logger.info(u'已统计了%s个句子' % n)
            for ws in sent:
                for i, w1 in enumerate(ws):
                    self.words[w1] += 1
                    for w2 in ws[i + 1:i + 1 + self.window]:
                        self.skip_grams[w1, w2] += 1

        logger.info(u'共统计了%s个句子' % (n + 1))

    def find(self, sents, filtering=True, remove_weak_grams=True):
        """基本的句模版构建算法

        在train的基础上，然后计算互信息，通过互信息给每个词找出最邻近的
        强相关的词语，然后将它们连接起来，就构成一个模版。

        【参数含义】
        filtering：是否过滤无关字符，只保留中文、数字和英文字母；
        remove_weak_grams：训练完成后是否在self.skip_grams去掉低频项，这个
                           参数只影响增量更新。
        """
        log_min_proba = log(self.min_proba)
        log_total_words = log(sum(self.words.values()))
        log_total_skip_grams = log(sum(self.skip_grams.values()))
        words = {i: log(j) - log_total_words for i, j in self.words.items()}
        skip_grams = {
            i: log(j) - log_total_skip_grams
            for i, j in self.skip_grams.items()
        }
        skip_grams = {i: j for i, j in skip_grams.items() if j > log_min_proba}

        if remove_weak_grams:
            self.skip_grams = {
                i: j
                for i, j in self.skip_grams if i in skip_grams
            }

        skip_grams = {
            i: j
            for i, j in skip_grams.items()
            if j - words[i[0]] - words[i[1]] > self.min_pmi
        }

        logger = logging.getLogger(u'模版构建')
        for n, sent in enumerate(self.texts(sents, filtering)):
            if n % 10000 == 0 and n > 0:
                logger.info(u'已处理了%s个句子' % n)
            for ws in sent:
                dag = DAG(len(ws))
                for i, w1 in enumerate(ws):
                    for j, w2 in enumerate(ws[i + 1:i + 1 + self.window]):
                        if (w1, w2) in skip_grams:
                            dag[i, i + 1 + j + 1] = 1
                paths = dag.all_paths()
                for p in paths:
                    t = Template()
                    if p[0] != 0:
                        t.add(None)
                    t.add(ws[p[0]])
                    for i in range(1, len(p)):
                        if p[i] - p[i - 1] > 1:
                            t.add(None)
                        t.add(ws[p[i]])
                    if p[-1] != len(ws) - 1:
                        t.add(None)
                    self.templates[t] += 1

        logger.info(u'共处理了%s个句子' % (n + 1))

        log_total_templates = log(sum(self.templates.values()))
        self.templates = {
            i: j
            for i, j in self.templates.items()
            if log(j) - log_total_templates > log_min_proba
        }


class XTrie(Trie):
    """新的Trie树，以tuple为输入（原来是字符串），
    并且修改了search函数，实现对模版的查找
    """

    def search(self, sent):
        """重定义search函数，返回所有可能的匹配结果
        输出output的格式为{(开始位置, 终止位置): {(匹配模版的词tuple): 模版频数}}
        """
        output = defaultdict(dict)
        for i in range(len(sent)):
            # search函数从头查找所有可能的匹配模版
            self._search(sent[i:], output, 0, i, self.dic)
        return output

    def _search(
            self,
            sent,  # 句子词组输入（分好词）
            output,  # 输出（defaultdict）
            in_start=0,  # 内部起点（函数内部使用）
            out_start=0,  # 外部起点，即sent对应原始输入的起始位置
            trie_dic={},  # 当前层级的用来查找的trie_dic
            matching=()  # 当前匹配情况
    ):
        """查找原理：按照常规trie树的查找原则，
        但是每个单词有两个匹配可能：1.匹配单词本身；2.匹配占位符None。
        因此只好递归匹配。
        """
        if sent:
            c = sent[0]
            if c in trie_dic:  # 匹配上当前词
                matching_ = matching + (out_start + in_start,
                                        )  # 将词添加到匹配tuple中
                # out_start+in_start
                # 为词在原句子中的实际下标
                if self.end in trie_dic[c]:  # 如果完整匹配一个模版
                    output[out_start, out_start + in_start + 1][
                        matching_] = trie_dic[c][self.end][1]  # 添加到output
                self._search(
                    sent[1:],
                    output,
                    in_start + 1,
                    out_start,
                    trie_dic[c],
                    matching=matching_)  # 递归查找

            if None in trie_dic:  # 匹配上占位符，不需要修改matching
                if self.end in trie_dic[None]:  # 如果完整匹配一个模版
                    output[out_start, out_start + in_start + 1][
                        matching] = trie_dic[None][self.end][1]  # 添加到output
                self._search(
                    sent[1:],
                    output,
                    in_start + 1,
                    out_start,
                    trie_dic[None],
                    matching=matching)  # 递归查找

            # 重点！！！
            # 连续几个词都可以匹配上占位符，但形式上只算一个。
            # 反过来说，就是前面已经匹配了一个占位符，后面可以反复匹配占位符，trie_dic不用递归
            if (not matching  # matching为空，即为起点
                    or matching[-1] != out_start + in_start  # 这意味着前面匹配了占位符
                ):
                if self.end in trie_dic:  # 如果完整匹配一个模版
                    output[out_start, out_start + in_start + 1][
                        matching] = trie_dic[self.end][1]
                self._search(
                    sent[1:],
                    output,
                    in_start + 1,
                    out_start,
                    trie_dic,  # trie_dic不用递归，因为多个占位符也算一个
                    matching=matching)  # 递归查找


class SentTree:
    """句子结构树
    用于输出句子层次结构的解析结果并可视化
    """

    def __init__(self, parsing, words, start, end):
        """SentTree有三个属性：
        template是当前主模版，
        content是当前主模版覆盖的字符串
        modules是语义块列表，每个元素也是一个SentTree
        """
        _ = [words[i] if i in parsing[0] else None for i in range(start, end)]
        self.template = Template(_)
        self.content = ''.join(words[start:end])
        self.modules = []
        marks = (start - 1, ) + parsing[0] + (end, )  # 所有间隙
        k = 0
        for i in range(len(marks) - 1):
            if marks[i] + 1 < marks[i + 1]:
                # 递归创建
                st = SentTree(parsing[1][k], words, marks[i] + 1, marks[i + 1])
                self.modules.append(st)
                k += 1

    def plot(self, prefix='+---> '):  # 可视化函数
        strings = []
        k = 0
        s = ''
        for w in self.template:
            if w == None:
                strings.append(self.modules[k].plot('|     ' + prefix))
                s += '(' + self.modules[k].content + ')'
                k += 1
            else:
                s += w
                strings.append('|     ' + prefix + w)
        strings = [prefix + s] + strings
        return '\n'.join(strings)

    def __str__(self):
        if is_python2:
            return self.plot().encode('utf-8')
        else:
            return self.plot()

    def __repr__(self):
        return self.__str__()


class Parser:
    """基于句模版的句子层次结构解析
    试图用句模版覆盖整个句子，并且使得概率对数之和最大，
    句模版可以相互嵌套，但是不允许交叉，
    因此可以抽取出句子的层次结构。
    """

    def __init__(self, templates, tokenize, null_score=-20):
        self.templates = templates  # 模版trie树，需要XTrie类型
        self.tokenize = tokenize  # 分词函数
        self.null_score = null_score  # 不成模版的词的打分

    def max(self, x):  # 自定义最大值函数
        if x:
            return sorted(x, key=lambda s: (-round(s[1], 4), -len(s[0][0])))[0]
        else:
            return (-1, 0)

    def parse(self, sent):
        ws = self.tokenize(sent)
        matches = self.templates.search(ws)
        p = self._parse((0, len(ws)), matches)
        return SentTree(p[0], ws, 0, len(ws))  # 返回SentTree对象

    def _parse(self, span, matches):
        """解析思路：递归进行——如果一种层次结构是最优的，那么它每个语义块的
        层次结构都是最优的。利用这种特性，可以写成递层优化的算法。
        """
        tpls = []  # 候选层次结构集，每个元素代表一种层次结构
        # 格式为[((主模版词集, [语义块1, 语义块2]), 得分)]
        # 语义块1、语义块2依然为
        # (语义块主模版词集, [语义块的语义块1, 语义块的语义块2])
        # 的格式，即无限递归直到遍历完句子

        if span in matches:  # 意味着当前语义块能析出一个模版
            for t, f in matches[span].items():
                score = log(f) - log(self.templates.total_items)  # 主模版的打分
                modules = []  # 所有子语义块
                marks = (span[0] - 1, ) + t + (span[1], )  # 所有间隙
                for i in range(len(marks) - 1):  # 枚举每个语义块
                    if marks[i] + 1 < marks[i + 1]:  # 如果不满足，则是连续的两个词，
                        # 则不是间隔，无法构成语义块。
                        _tpls, _score = self._parse(
                            (marks[i] + 1, marks[i + 1]), matches)
                        modules.append(_tpls)
                        score += _score  # 主模版的分数+每个语义块的分数等于总分
                tpls.append(((t, modules), score))
            return self.max(tpls)  # 返回最优结构
        else:  # 意味着没有模版能解析当前语义块，那么将它视为一系列孤立的词
            return ((tuple(range(span[0], span[1])), []),
                    self.null_score * (span[1] - span[0] - 1))
