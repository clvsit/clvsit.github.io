---
title: n-gram 词频统计
date: 2019-12-15 14:33:01
tags:
category:
- 数据处理
---

# n-gram 词频统计
借助 sklearn 提供的 CountVectorizer 可以实现 n-gram 的词频统计。

## 实现过程
首先，导入所需的包以及数据。

```python
from sklearn.feature_extraction.text import CountVectorizer
from collections import ChainMap
import tqdm


with open("/nfs/users/chenxu/common_word_mining/dataset_word_cut_small.json", "r", encoding="utf-8") as file:
    content_list = json.load(file)
```

然后，调用 CountVectorizer，以获得每段文本的文本向量。
```python
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2,2), min_df=5)
X = vectorizer.fit_transform(content_list)
```
- min\_df：指定最小阈值；
- ngram\_range：指定要获取 n-gram 的范围。例如 ngram\_range=(1,2) 单词量的大小等于 ngram\_range=(1,1) 与 ngram\_range=(2,2) 单词量大小之和。

【注意】：
- 在处理中文时，token\_pattern 参数需要设置为 `(?u)\b\w+\b` ，即允许单个汉字。
- CountVectorizer 的 fit\_transform() 方法会将输入的文本转化为形似 ont-hot 向量（各维度的数值可以超过 1）。

接下来，我们可以使用 get\_feature\_names() 来获取所有的词语（即单词表）。
```
vocab_list = vectorizer.get_feature_names()
```

接着，借助 pandas 库生成一张列为单词表、行为文本形似 ont-hot 向量的表格。此时，我们只需要调用 DataFrame 对象的 sum() 方法即可得到每个词语的词频。
```python
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(df.sum())
```

![n-gram 词频](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/%7BC7D5BEA0-8615-4281-AB4F-624B5059ED07%7D_20191202110504.jpg)

当数据量较大时，我们无法将所有的数据都填充到 pandas 的 DataFrame 对象中，此时就需要对表格进行切分。

【步骤】：推荐按列（单词表）进行切分。
1. 每次对一部分单词生成 DataFrame 对象；
2. 调用当前 DataFrame 对象的 sum() 方法，并将结果转换为 dict（词语：词频）；
3. 每轮迭代过程中将上一轮的 dict 整合到当前 dict 中，最后整合成一个完整的 dict。

【关于 dict 合并的优化】：我们可以使用 collections 包中的 ChainMap 来帮助我们加快 dict 的合并操作。需要注意的是，ChainMap 返回的是 ChainMap 对象，我们还需要将其转换为 dict。
```python
vocab_dict = dict(ChainMap(vocab_dict, dict(df.sum().items())))
```

【完整代码】：
```python
from sklearn.feature_extraction.text import CountVectorizer
from collections import ChainMap
import tqdm


with open("/nfs/users/chenxu/common_word_mining/dataset_word_cut_small.json", "r", encoding="utf-8") as file:
    content_list = json.load(file)

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2,2), min_df=5)
X = vectorizer.fit_transform(content_list)


def summary(X, vocab_list, batch_size=1000):
    rows, cols = X.shape
    iters = cols // batch_size + 1    
    vocab_dict = {}
    
    for i in tqdm.tqdm(range(iters)):
        start, end = batch_size * i, batch_size * (i + 1)
        df = pd.DataFrame(X[:, start:end].toarray(), columns=vocab_list[start:end])        
        vocab_dict = dict(ChainMap(vocab_dict, dict(df.sum().items())))
    return vocab_dict


vocab_list = vectorizer.get_feature_names()
vocab_dict = summary(X, vocab_list, 1000)
```

## 与 Counter 比较
使用 Counter 需要先对文本数据做一些处理。
```python
content_split_list = [content.split(" ") for content in content_list]
content_split_list[:10]
```
![Counter 性能](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/Counter%E6%80%A7%E8%83%BD.jpg)

![CountVectorizer 性能](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/n-gram%E6%80%A7%E8%83%BD.jpg)

通过比对 CountVectorizer 与 Counter 的执行时间，可知 CountVectorizer 在执行效率上要高于 Counter。

## 问题

### CountVectorizer 词频统计所遇到的坑
在使用 sklearn 的 CountVectorizer 方法过程中发现，使用 CountVectorizer 统计获得的 char 个数要少于 Counter 方法统计的 char 个数。

#### 准备工作
【数据】：/nfs/users/chenxu/common_word_mining/data/char_total_list.json
```
[['日', '前'],
 ['捷', '豹', '路', '虎'],
 ['中', '国'],
 ...
 ['的', '要', '求'],
 ['向', '质', '检', '总', '局', '备', '案', '了', '召', '回', '计', '划'],
 ['将', '自', '2017', '年', '12', '月', '22', '日', '起'],
 ['召', '回', '部', '分', '进', '口', '路', '虎', '新', '揽', '胜']]
```

【引入的包】：
```python
import pandas as pd
import numpy as np
import json
import tqdm
from collections import Counter, ChainMap
```

【读取数据】：
```python
with open("/nfs/users/chenxu/common_word_mining/data/char_total_list.json", "r", encoding="utf-8") as file:
    char_total_list = json.load(file)

content_char_list = [" ".join(char_list) for char_list in char_total_list]
```

拼接后的数据 content_char_list：
```
['日 前',
 '捷 豹 路 虎',
 '中 国',
 ...
 '的 要 求',
 '向 质 检 总 局 备 案 了 召 回 计 划',
 '将 自 2017 年 12 月 22 日 起',
 '召 回 部 分 进 口 路 虎 新 揽 胜']
```

#### CountVectorizer
【执行代码】：
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), stop_words=None, lowercase=None)
X = vectorizer.fit_transform(content_char_list)
char_list_cv = vectorizer.get_feature_names()
print(len(char_list_cv))  # 2483
```

#### Counter
【执行代码】：
```python
from collections import Counter


def summary_char(char_total_list):
    char_counter = Counter(char_total_list[0])
    
    for index in tqdm.tqdm(range(1, len(char_total_list))):
        char_counter += Counter(char_total_list[index])
    
    return dict(char_counter)

char_dict = summary_char(char_total_list)
print(len(char_dict))  # 2529
```

我们可以直接使用 set 的方式去统计 char 的个数。
```python
char_list_all = []

for char_list in char_total_list:
    char_list_all.extend(char_list)

char_set = set(char_list_all)
print(len(char_set))  # 2529
```

再做进一步的验证：
```python
>>> char_set - set(char_dict)
set()
```

可知 Counter 方法统计的 char 个数没有问题。

#### 两者进行对比
我们再来对比 CountVectorizer 方法获得的字集合与 Counter 方法获得的字集合的差异。
```
>>> set(char_dict) - set(vocab_dict)
{'+',
 '+10',
 '+12',
 '+7',
 '-',
 '-25℃\xa0',
 '-35',
 '-AMGA35',
 ...
  '——',
 '——18T',
 '——Ascent',
 '——Nautilus',
 '——Urus'}
```

#### 产生原因
CountVectorizer 默认会将英文转换为小写，例如 AMG 转换为 amg，这导致我误认为 CountVectorizer 会过滤部分英文和数字，但实际上只是因为这些英文和数字不匹配 token_pattern 对应的正则表达式。我们只需要将其修改为 `r"(?u)\b\w+[-+]?\w*\b"` 即可统计 "Wi-Fi" 这一类的文本内容。