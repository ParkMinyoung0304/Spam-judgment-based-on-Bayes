from spam.spamEmail import spamEmailBayes
import re
import pickle

spam=spamEmailBayes()

#保存每封邮件中出现的词
wordsList=[]
wordsDict={}
stopList=spam.getStopWords()

a_file = open("Model.pkl", "rb")
normDict,spamDict,normFilelen,spamFilelen = pickle.load(a_file)
a_file.close()

for line in open("../data/test.txt"):   #在这里输入你想测试的邮件路径，或者去替换test.txt的文件内容，注意是相对路径
    rule=re.compile(r"[^\u4e00-\u9fa5]")
    line=rule.sub("",line)
    spam.get_word_list(line,wordsList,stopList)

spam.addToDict(wordsList, wordsDict)
testDict=wordsDict.copy()

#通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
wordProbList=spam.getTestWords(testDict, spamDict,normDict,normFilelen,spamFilelen)

#对每封邮件得到的15个词计算贝叶斯概率  
p=spam.calBayes(wordProbList, spamDict, normDict)
if(p>0.9):
    print("这是一封垃圾邮件")
else:
    print("这是一封正常邮件")  