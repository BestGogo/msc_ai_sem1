import pycrfsuite
import string
import nltk, re, math
from nltk import tag
import pandas as pd
import numpy as np
import textstat
import collections, itertools
from spellchecker import SpellChecker
from flair.data import Sentence
from flair.models import SequenceTagger
flatten = itertools.chain.from_iterable
TAGGER = SequenceTagger.load('pos')
tagger = pycrfsuite.Tagger()
tagger.open('../model/model.crf.tagger')
POS_DICTIONARY = {}
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet
punct = set(string.punctuation)
lemmatizer = nltk.WordNetLemmatizer()
stopwords = stopwords.words('english')
porter = PorterStemmer()
SPELL = SpellChecker()
SPELL.word_frequency.load_words(["'s", "'m", "'re", "'ll" , "'ve", "'t", "'d"])
from modules import factor_analysis

def lemmatize(token, tag):
    """
    Use NLTK Lemmatizer to lemmatize 
    params: token, tags
    return lemma
    """
    tag = {
        'N': wordnet.NOUN,
        'B': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(tag[0], wordnet.NOUN)

    return lemmatizer.lemmatize(token, tag)

def tokenNormalizeText(text):
    """
    return sentence in Normalized form
    features in stopword removal, and lemmatization
    params: text, string
    return: text, string
    """
    # Remove distracting single quotes
    text = re.sub(r"\'", "", text)
    # Remove distracting double quotes
    text = re.sub(r'\"', "", text)
    # Remove new line characters
    text = re.sub(r'\s+', ' ', text)
    # word normalisation
    text = re.sub(r"(\w)([.,;:!?'/\"”\)])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?'/\"“\(])(\w)", r"\1 \2", text)
    # normalisation
    text = re.sub(r"(\S)\1\1+",r"\1\1\1", text)
    #tokenising
    
    tokens = list(flatten([re.split(r"\s+",t) for t in re.split('(\d+)',text)]))
    tokens = [re.sub(r'[^A-Za-z]+','',t) for t in tokens]
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in ' ']# and len(t) > 2]
    tokens = [w for w in tokens if w not in stopwords ]
    tokens = [lemmatize(token, tag) for token, tag in nltk.pos_tag(nltk.wordpunct_tokenize(' '.join(tokens)))]
#     tokens = [str(porter.stem(w)) for w in tokens]
    return ' '.join(tokens)

def normalizeText(text):
    """
    returns tokenize text in its original form
    """
#     s = s.lower()
#     s = re.sub('\s\W',' ',s) #  hyphens, apostrophes
#     s = re.sub('\W\s',' ',s)
#     s = re.sub('\s+',' ',s) # double spaces
    tokens = nltk.word_tokenize(str(text).lower())
    return ' '.join(tokens)

def textTokenizer(text):
    """
    returns tokenize text with special character parsing
    """
    text = re.sub("[/%-._]", " ", text)
    text = re.sub("[,()!;$?:~*]","", text)
    text = text.replace('"', '')
    text = text.replace(" '", '')
    text = text.replace("' ", '')
    tokens = nltk.word_tokenize(text)
    return tokens


def createDiagActFeatures(sentence):
    """
    Diagloue Act Feature generator, feature are of form
    TOKEN_TOKEN POS_TOKEN TOKEN1 TOKEN2
    params: sentence, string
    return: nested feature list
    """
    features = []
    if len(sentence.split())>=2:
        tagged_sent = tag.pos_tag(nltk.word_tokenize(sentence))
        for tagset in tagged_sent:
            features.append('TOKEN_'+tagset[0])
        for tagset in tagged_sent:
            features.append('POS_'+tagset[1])
        for words in sentence.split():
            features.append(words)
        features.append('/')
    return [[features]]

def POSTag(words,type="flair"):
    """
    Two types of POS taggers, Flair(https://github.com/zalandoresearch/flair) and NLTK
    # Accuracy was merely increased by 0.5 - 1 % using flair
    # Inorder to use flair use data/pos_dict.npy
    params: words, a sentence string
            tagger, type of tagger string
    return: postags, list of pos tag for the senetnce
    """
    if type=="NLTK":
        if words in POS_DICTIONARY:
            return POS_DICTIONARY[words]
        else:
            postags = []
            text = str(words).replace(',', ', ').replace('.', '. ')
            tagged_sent = tag.pos_tag(nltk.word_tokenize(text))
            postags = [tags[1] for tags in tagged_sent]
            sentence = Sentence(text, use_tokenizer=True)
            POS_DICTIONARY[words] = postags
            return postags
    if type=="flair":
        if words in POS_DICTIONARY:
            return POS_DICTIONARY[words]
        else:
            postags = []
            text = str(words).replace(',', ', ').replace('.', '. ')
            sentence = Sentence(text, use_tokenizer=True)
            TAGGER.predict(sentence)
            for token in sentence:
                pos = token.get_tag('pos').value
                postags.append(pos)
            POS_DICTIONARY[words] = postags
            return postags


def calcFMeasure(text):
    """
          Get F measure which is defined as
          f = 0.5[(freq.noun+freq.adj+freq.prep+freq.art)-(freq.pron+freq.verb+freq.adv+freq.int)+100]
          freq is the frequency
          params: text
          return: integer score
    """

    tagged = POSTag(text, type='NLTK')

    grammar_freq = {'noun':0,'adj':0,'prep':0,'art':0,'pron':0,'verb':0,'adv':0,'int':0}
    grammer_type = {
                    'noun' : ['NN', 'NNS', 'NNP', 'NNPS'],
                    'adj' : ['JJ', 'JJR', 'JJS'],
                    'prep':['IN'],
                    'art':['DET', 'DT', 'PDT', 'WDT'],
                    'pron':['PRP', 'PRP$', 'WP', 'WP$'],
                    'verb':['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                    'adv':['RB', 'RBR', 'RBS', 'WRB'],
                    'int':['UH']
                    }

    count = 0
    for i in range(len(tagged)):
        pos = tagged[i]
        if pos in grammer_type['noun']:
            grammar_freq['noun'] += 1
        elif pos in grammer_type['adj']:
            grammar_freq['adj'] += 1
        elif pos in grammer_type['prep']:
            grammar_freq['prep'] += 1
        elif pos in grammer_type['art']:
            grammar_freq['art'] += 1
        elif pos in grammer_type['pron']:
            grammar_freq['pron'] += 1
        elif pos in grammer_type['verb']:
            grammar_freq['verb'] += 1
        elif pos in grammer_type['adv']:
            grammar_freq['adv'] += 1
        elif pos in grammer_type['int']:
            grammar_freq['int'] += 1

        if pos not in ['$', "'", '(', ')', ',', '-', '.', ':', 'SYM', "''", '``']:
            count += 1

    for key in grammar_freq:
        grammar_freq[key] = (grammar_freq[key] / count) * 100

    fmeasure = 0.5 * ( (grammar_freq['noun'] + grammar_freq['adj'] + grammar_freq['prep'] + grammar_freq['art']) - (grammar_freq['pron'] + grammar_freq['verb'] + grammar_freq['adv'] + grammar_freq['int']) + 100 )

    return fmeasure



def POSFeatures(text):
    """
    return POS tags for given input text sentence string
    """
    pos_tags = POSTag(text, type='NLTK')
    return ' '.join(pos_tags)

def POSTaggedFeatures(text, type="flair"):
    """
    Gives POS features in form of TOKEN_POS
    params: text, a sentence string
            type, type of POS tagger to use (flair/NLTK)
    return: tagged_sent, a string of form TOKEN_POS
    """
    if type=="NLTK":
        pos_tags = POSTag(str(text),type="NLTK")
        tagged_sent = []

        cleaned_text = str(text).replace(',', ', ').replace('.', '. ')
    #     sentence = Sentence(cleaned_text, use_tokenizer=True)
        sentence = nltk.word_tokenize(cleaned_text)

        for i in range(len(pos_tags)):
            tagged_sent.append(sentence[i] + '_' + pos_tags[i])
        return ' '.join(tagged_sent)
    if type=="flair":
        pos_tags = POSTag(str(text),type="flair")
        tagged_sent = []

        cleaned_text = str(text).replace(',', ', ').replace('.', '. ')
        sentence = Sentence(cleaned_text, use_tokenizer=True)

        for i in range(len(pos_tags)):
            tagged_sent.append(sentence[i].text + '_' + pos_tags[i])

        return ' '.join(tagged_sent)


def genderFavouredFeatures(text):
    """
    Gender Favoured features are count of words 
    which are specifically ending with certain 
    characters or sentences have words in them.
    param: text, a sentence string
    return: f, a feature list of count
    """
    gf = []
    word_types = {
                    'f0':'able',
                    'f1':'al',
                    'f2':'ful',
                    'f3':'ible',
                    'f4':'ic',
                    'f5':'ive',
                    'f6':'less',
                    'f7':'ly',
                    'f8':'ous',
                    'f9':['sorry', 'penitent', 'contrite', 'repentant', 'remorseful', 
                        'regretful', 'compunctious', 'touched', 'melted', 'sorrowful',
                        'apologetic', 'softened','sad', 'greived', 'mournful']

                }
    for i in range(10):
        gf.append(0)
    for word in textTokenizer(str(text).lower()):
        for ftype,fword in word_types.items():
            if ftype=='f0' and word.endswith((fword)):
                gf[0] += 1
            elif ftype=='f1' and word.endswith((fword)):
                gf[1] += 1
            elif ftype=='f2' and word.endswith((fword)):
                gf[2] += 1
            elif ftype=='f3' and word.endswith((fword)):
                gf[3] += 1
            elif ftype=='f4' and word.endswith((fword)):
                gf[4] += 1
            elif ftype=='f5' and word.endswith((fword)):
                gf[5] += 1
            elif ftype=='f6' and word.endswith((fword)):
                gf[6] += 1
            elif ftype=='f7' and word.endswith((fword)):
                gf[7] += 1
            elif ftype=='f8' and word.endswith((fword)):
                gf[8] += 1
            if ftype=='f9' and word in fword:
                gf[9] += 1
    return gf


factors = list(factor_analysis.word_type.keys())
def factorAnalysis(text):
    """
    finding groups of similar words that tend to occur in similar documents
    group of similar words present in factor_analysis->word_tyoe
    params: text, a sentence string
    return: fa, a list of occurance of counts of given word in sentence
    """
    fa = []    
    for i in range(len(factors)):
        fa.append(0)
    for word in textTokenizer(str(text).lower()):
        for  i,fact in enumerate(factors):
            if word in factor_analysis.word_type[fact]:
                fa[i] += 1
    return fa


def textStatistics(text):
    """
    returns text statistics such as lexicon count and text standard in a tuple
    """
    le_c = textstat.lexicon_count(text, removepunct=True)
    ts = textstat.text_standard(text, float_output=True)

    return le_c, ts


def countIncorrectWordChars(text):
    """
    return len of word chars which are incorrect or does not exits in english,(combination of several words).
    """
    tokens = textTokenizer(text)

    misspelled = SPELL.unknown(tokens)


    misspelled.discard('')
    return len(misspelled)


# POS_DICTIONARY = np.ndarray.tolist(np.load("data/pos_dict.npy",allow_pickle=True))
df = pd.read_csv("../data/training.csv",names=['text','character','gender'])

df['text_norm'] = df.apply(lambda x: normalizeText(str(x.text)),axis = 1)
df['token_text_norm'] = df.apply(lambda x: tokenNormalizeText(str(x.text)),axis = 1)
df['POS'] = df.apply(lambda x: POSFeatures(str(x.text)), axis = 1)
df['POS_tagged'] = df.apply(lambda x: POSTaggedFeatures(str(x.text),type='NLTK'), axis = 1)
df['f_measure'] = df.apply(lambda x: calcFMeasure(str(x.text)), axis=1)
np.save("../data/pos_dict.npy", POS_DICTIONARY)
df['word_count'] = df.apply(lambda x: len(textTokenizer(str(x.text))) , axis=1)
df['length'] = df.apply(lambda x: len(str(x.text)), axis = 1)
df['gf'] = df.apply(lambda x: genderFavouredFeatures(str(x.text)), axis = 1)
# df[["GPF0","GPF1","GPF2","GPF3","GPF4","GPF5","GPF6","GPF7","GPF8","GPF9"]] = pd.DataFrame(df.gf.values.tolist(), index= df.index)
df['fa'] = df.apply(lambda x: factorAnalysis(str(x.text)), axis = 1)
# df[["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22"]] = pd.DataFrame(df.fa.values.tolist(), index= df.index)
df['diag_act'] = df.apply(lambda x: [tagger.tag(xseq) for xseq in createDiagActFeatures(str(x.text))][0][0], axis=1)
mostCommonDiagAct = collections.Counter(df.diag_act.values.tolist()).most_common(8)
mostCommonDiagAct = [i[0] for i in mostCommonDiagAct]
def helper(a):
    if a not in mostCommonDiagAct:
        return 'othr'
    else: return a
    
df['diag_act'] = df.apply(lambda x: helper(x.diag_act),axis=1)
act_types=df['diag_act'].value_counts()
df = pd.get_dummies(df, columns=["diag_act"], prefix=["diag_act"])
df['diag_act']= df[['diag_act_'+act_types[act_types==i].index[0] for i in act_types]].values.tolist()
for i in act_types:
    df = df.drop(['diag_act_'+act_types[act_types==i].index[0]],axis=1)

df['LE_C'] = df.apply(lambda x: textStatistics(str(x.text))[0], axis = 1)
df['TS'] = df.apply(lambda x: textStatistics(str(x.text))[1], axis = 1)
df['mispelled'] = df.apply(lambda x: countIncorrectWordChars(str(x.text)), axis = 1)
df['gender'] = [1 if x =='male' else -1 for x in df['gender']] 
# encoder = LabelEncoder()
# df['character'] = encoder.fit_transform(df['character'])

df.drop(['gf','fa','character'],axis=1).to_csv('../data/train_gender.csv',index=False)
df.drop(['gf','fa','gender'],axis=1).to_csv('../data/train_character.csv',index=False)


df['index'] = df.index
df_train = df.copy()


train_dict = {}
for cols in list(df_train.columns):
    train_dict[cols] = df_train[cols].values.tolist()
np.save('../data/train_dict.npy', train_dict)


# from MinePOSPats import MinePOSPats
# pos_list = df['POS'].values.tolist()
# print(pos_list[0:10])
# mine_obj = MinePOSPats(pos_list, 0.3, 0.2)
# pos_pats = mine_obj.MinePOSPats()

# # Write POS Patterns to Text
# with open('../data/POSPatterns.txt', 'w') as file:
#     patterns = []
#     for pos_pat in pos_pats:
#         pattern = ' '.join(pos_pat)
#         patterns.append(pattern)
#     file.write('\n'.join(patterns))


df = pd.read_csv("../data/test.csv",names=['text','character','gender'])

df['text_norm'] = df.apply(lambda x: normalizeText(str(x.text)),axis = 1)
df['token_text_norm'] = df.apply(lambda x: tokenNormalizeText(str(x.text)),axis = 1)
df['POS'] = df.apply(lambda x: POSFeatures(str(x.text)), axis = 1)
df['POS_tagged'] = df.apply(lambda x: POSTaggedFeatures(str(x.text),type='NLTK'), axis = 1)
df['f_measure'] = df.apply(lambda x: calcFMeasure(str(x.text)), axis=1)
# np.save("data/pos_dict.npy", POS_DICTIONARY)
df['word_count'] = df.apply(lambda x: len(textTokenizer(str(x.text))) , axis=1)
df['length'] = df.apply(lambda x: len(str(x.text)), axis = 1)
df['gf'] = df.apply(lambda x: genderFavouredFeatures(str(x.text)), axis = 1)
# df[["GPF0","GPF1","GPF2","GPF3","GPF4","GPF5","GPF6","GPF7","GPF8","GPF9"]] = pd.DataFrame(df.gf.values.tolist(), index= df.index)
df['fa'] = df.apply(lambda x: factorAnalysis(str(x.text)), axis = 1)
# df[["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22"]] = pd.DataFrame(df.fa.values.tolist(), index= df.index)
df['diag_act'] = df.apply(lambda x: [tagger.tag(xseq) for xseq in createDiagActFeatures(str(x.text))][0][0], axis=1)
mostCommonDiagAct = collections.Counter(df.diag_act.values.tolist()).most_common(8)
mostCommonDiagAct = [i[0] for i in mostCommonDiagAct]
def helper(a):
    if a not in mostCommonDiagAct:
        return 'othr'
    else: return a
    
df['diag_act'] = df.apply(lambda x: helper(x.diag_act),axis=1)
act_types=df['diag_act'].value_counts()
df = pd.get_dummies(df, columns=["diag_act"], prefix=["diag_act"])
df['diag_act']= df[['diag_act_'+act_types[act_types==i].index[0] for i in act_types]].values.tolist()
for i in act_types:
    df = df.drop(['diag_act_'+act_types[act_types==i].index[0]],axis=1)

df['LE_C'] = df.apply(lambda x: textStatistics(str(x.text))[0], axis = 1)
df['TS'] = df.apply(lambda x: textStatistics(str(x.text))[1], axis = 1)
df['mispelled'] = df.apply(lambda x: countIncorrectWordChars(str(x.text)), axis = 1)
df['gender'] = [1 if x =='male' else -1 for x in df['gender']] 
# encoder = LabelEncoder()
# df['character'] = encoder.fit_transform(df['character'])

df.drop(['gf','fa','character'],axis=1).to_csv('../data/test_gender.csv',index=False)
df.drop(['gf','fa','gender'],axis=1).to_csv('../data/test_character.csv',index=False)


df['index'] = df.index
df_test = df.copy()

test_dict = {}
for cols in list(df_train.columns):
    test_dict[cols] = df_test[cols].values.tolist()
np.save('../data/test_dict.npy', test_dict)