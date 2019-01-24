import nltk

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')#符号集

stemmer = nltk.stem.PorterStemmer()


def stem(word): #单词压缩  能把一个单词的不同形式映射到一个单词上 例："create" 和 "created" ，都得到"creat"
    return stemmer.stem(word)


def clean_sentence(text, stemming=False): #格式化句子， 输入文本， 分割单词， 是否进行压缩
    for token in punct: #从text中去除punct包含的字符
        text = text.replace(token, "") 
    words = text.split() #按空白字符分割
    if stemming: # 如果压缩， 则会将单词 “归一化”
        stemmed_words = [] #列表
        for w in words: #枚举单词w
            stemmed_words.append(stem(w)) # stem 压缩单词， 对于英文单词 来说， 可以将其不同形态 映射到一个单词上
        words = stemmed_words
    return " ".join(words) 


def clean_name(name): #格式化名称
    if name is None:
        return "" #strip() 移除首尾空白字符
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()] #按'.', '-'分割开字符 小写化
    return "_".join(x)
