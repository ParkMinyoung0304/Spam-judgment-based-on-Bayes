#encoding=utf-8
from spam.spamEmail import spamEmailBayes
import re
import pickle
#spam类对象
spam=spamEmailBayes()
#保存词频的词典
spamDict={}
normDict={}

#保存每封邮件中出现的词
wordsList=[]
wordsDict={}

#分别获得正常邮件、垃圾邮件及测试文件名称列表
normFileList=spam.get_File_List(r"D:\\校用\大三下\\垃圾邮件分类\\BayesSpam-master\\BayesSpam-master\\data\normal")  #正常邮件路径，注意替换
spamFileList=spam.get_File_List(r"D:\\校用\大三下\\垃圾邮件分类\\BayesSpam-master\\BayesSpam-master\\data\\spam")   #垃圾邮件路径，注意替换

#获取训练集中正常邮件与垃圾邮件的数量
normFilelen=len(normFileList)
spamFilelen=len(spamFileList)

#获得停用词表，用于对停用词过滤
stopList=spam.getStopWords()

#获得正常邮件中的词频
for fileName in normFileList:
    wordsList.clear()
    for line in open("../data/normal/"+fileName):
        #过滤掉非中文字符
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        #将每封邮件出现的词保存在wordsList中
        spam.get_word_list(line,wordsList,stopList)
    #统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
normDict=wordsDict.copy()  

#获得垃圾邮件中的词频
wordsDict.clear()
for fileName in spamFileList:
    wordsList.clear()
    for line in open("../data/spam/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
spamDict=wordsDict.copy()

a_file = open("Model.pkl", "wb")
pickle.dump((normDict,spamDict,normFilelen,spamFilelen), a_file)
a_file.close()
print("Has Built Model Finish!")