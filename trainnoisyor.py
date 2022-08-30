from collections import Counter
import string
import nltk
import pandas as pd


tunigrams=[]
tbigrams=[]
narrLength=[]
   
def countWOrdUni(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    unig = Counter(nltk.ngrams(text.split(),1))
    dic={''.join(ch for ch in key if ch not in exclude):value for key, value in unig.items()}
    unigm=[''.join(ch for ch in key if ch not in exclude) for key, value in unig.items()]
    return dic,unigm

def countWOrdBi(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    bigmC = Counter(nltk.bigrams(text.split()))
    dic={' '.join(ch for ch in key if ch not in exclude):value for key, value in bigmC.items()}
    bigm=[' '.join(ch for ch in key if ch not in exclude) for key, value in bigmC.items()]

    return dic,bigm

def WordPro(P_word,N_word,thrs):
    uniProb={}    
    for tg, n_count in N_word.items():        
        if tg in P_word.keys() and P_word[tg]>=thrs:
            p_count=P_word[tg]
            c2 = float(p_count)
            p = (c2+1)/(c2+float(n_count)+2)
            uniProb[tg ]=(p,p_count,n_count)
           #print (uniProb[tg ],p)
    return uniProb

def classifier( thrs, unigrams, bigrams, U_list,B_list):    

    WordProbMult = 1
    
    Positive_Bigams = []
    Positive_Unigams = []
    for unigram in unigrams:
        if unigram in U_list.keys():
            pr = float(U_list[unigram][0])
            #print(unigram,pr) 
            
            if pr >= thrs:
                
                WordProbMult = WordProbMult * (1 - pr)
                #print(WordProbMult)
                
                Positive_Unigams.append(unigram)
    #print(WordProbMult)

    for bigram in bigrams:
        if bigram in B_list.keys():
            pr = float(B_list[bigram][0])  
            #print(bigram,pr)  
            
            if pr >= thrs :
                WordProbMult = WordProbMult * (1 - pr)
                #print(WordProbMult)
                Positive_Bigams.append(bigram)
            else:
                Positive_Bigams.append(bigram)
    NoisyOR = 1 - WordProbMult

    return NoisyOR, Positive_Unigams, Positive_Bigams
def classifierDefaultModel( thrs, unigrams, bigrams, U_list,B_list):    

    WordProbMult = 1
    
    Positive_Bigams = []
    Positive_Unigams = []
    for unigram in unigrams:
        if unigram in U_list.keys():
            pr = float(U_list[unigram])
            
            if pr >= thrs:
                
                WordProbMult = WordProbMult * (1 - pr)
                
                Positive_Unigams.append(unigram)
    #print(WordProbMult)

    for bigram in bigrams:
        if bigram in B_list.keys():
            pr = float(B_list[bigram])  
            
            
            if pr >= thrs :
                WordProbMult = WordProbMult * (1 - pr)
                Positive_Bigams.append(bigram)
            else:
                Positive_Bigams.append(bigram)
    NoisyOR = 1 - WordProbMult

    return NoisyOR, Positive_Unigams, Positive_Bigams
def classify(testDataDf,uniPro, biPro,labelFieldName):
    clsThrss=[0.25,0.35,0.45,0.5,0.6,0.7,.8]
    #clsThrss=[0.80]
    #wordProThrss=[0.65]
    TN_WZ=0
    FN_WZ=0
    TP_WZ=0
    FP_WZ=0 
    recT=[]
    record=[]
    
    for Cls in clsThrss:        
        recordTemp=[]
        for row2 in range (0, testDataDf.shape[0]):                            
            C1_S, C1_U, C1_B= classifier( Cls, tunigrams[row2], tbigrams[row2],uniPro, biPro) 
            Glebel=testDataDf.iloc[row2][labelFieldName] 
            recordTemp.append((C1_S,Glebel))
        recT.append(recordTemp)


    for ClsThrs in [0.5,0.6,0.7,0.8,0.9,0.95]:
        n=0
        for clscore in recT:
            saveClsThrs=(clsThrss[n],ClsThrs)
            n+=1
            for C1_S in clscore:
                
                        
                if  C1_S[0]<ClsThrs:
                        #print(testDataDf.iloc[row2]['DISCON'])
                        
                    if C1_S[1]==0:
                        TN_WZ+=1
                    else:
                        FN_WZ+=1 
                        
            
                else: 
                    if C1_S[1]==1:
                        TP_WZ+=1
                    else:
                        FP_WZ+=1
    
            preci=(TP_WZ)/(TP_WZ+FP_WZ+0.000001)
            rec=(TP_WZ)/(TP_WZ+FN_WZ+0.000001)
            accuracy=(TN_WZ+TP_WZ)/(TN_WZ+TP_WZ+FN_WZ+FP_WZ+0.000001)         
            FScore=2*preci*rec/(preci+rec+0.000001)        
            TN_WZ=0
            FN_WZ=0
            TP_WZ=0
            FP_WZ=0
            record.append([saveClsThrs,round(FScore,3),round(preci,3),round(rec,3),round(accuracy,3)])

    return record

            
def Robust_Classifier(inputcsv,labelFieldName,textFieldName):

    inputcsv[labelFieldName].astype('int')                             
    for row2 in inputcsv[textFieldName]:
        _,tug = countWOrdUni(row2)
        _,tbg=countWOrdBi(row2)
        tunigrams.append(tug) 
        tbigrams.append(tbg)
        narrLength.append(len(tug))
    Pdf=inputcsv[inputcsv[labelFieldName]==1]
    Ndf=inputcsv[inputcsv[labelFieldName]==0]
    #print(Pdf[textFieldName])

    PText= ' '.join ( str(i) for i in Pdf[textFieldName])
    NText= ' '.join(str(i) for i in Ndf[textFieldName]) 
    #print(PText)              
    P_word_uni,_=countWOrdUni(PText)
    N_word_uni,_=countWOrdUni(NText)                  
    P_word_bi,_=countWOrdBi(PText)
    N_word_bi,_=countWOrdBi(NText) 
                   
    uniPROB=WordPro(P_word_uni,N_word_uni,1) 
    biPROB=WordPro(P_word_bi,N_word_bi,1) 
    #TOPuniPROB=sorted(uniPROB.items(),key=lambda x:x[1][0],reverse=True)
    #TOPbiPROB=sorted(biPROB.items(),key=lambda x:x[1][0],reverse=True)
  
   # print( narrLength)
    return classify(inputcsv,uniPROB, biPROB,labelFieldName),uniPROB,biPROB,narrLength

#inputdata='CRU_DT4000_final_2021_narrative_NarrAdded.csv'  
#df=pd.read_csv(inputdata,encoding="utf-8",nrows=2000)
#df=df.dropna(subset=['OFFRNARR'])

#print(Robust_Classifier(df,'DISCON','OFFRNARR'))