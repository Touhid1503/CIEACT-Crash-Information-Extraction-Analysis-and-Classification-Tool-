from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse
from datetime import datetime
from django.contrib import messages
from .models import UserDatabase,AdminDatabase,UserQuery
from .models import DistractedUnigram,DistractedBigram,DistractedTrigram
from .models import InattentiveUnigram,InattentiveBigram,InattentiveTrigram
from .models import WorkzoneUnigram,WorkzoneBigram,WorkzoneTrigram
import csv, io
from .classifiers import ClassifierWthrs
from .modelNameQuery import pullWordProbability
from asgiref.sync import sync_to_async,async_to_sync
import time,asyncio
from django.template import loader
from .trainnoisyor import Robust_Classifier,classifier,classifierDefaultModel
from plotly.offline import plot
import plotly.express as go
from .CIEACT_viz import textViz
from .trainnoisyor import countWOrdUni,countWOrdBi
# Create your views here.

    
def homePage(request):
    template_name = "first_page.html"
    return render(request, template_name)
 
def helpPage(request):
    template_name = "help.html"   
    return render(request, template_name)

def contactPage(request):
    template_name = "contact.html"   
    return render(request, template_name)

def aboutPage(request):
    template_name = "about.html"   
    return render(request, template_name)

        

# welcome to custom model
def customTrainTestUpload(request):
    template = loader.get_template('DataUploadCustomMethod_test.html') 
    sesID=request.POST.get('sesID')
    #print('sesID2: ',sesID)
    context = {'sesID':sesID}
    return HttpResponse(template.render(context, request))
        
        
# welcome to custom model
def DataUploadDefaultMethod(request):
    template_name = "DataUploadDefaultMethod.html"
    return render(request, template_name)
# page to upload csv for train

def UpldCvsFORTrn(request):
    template_name = "DataUploadCustomMethod_train.html"
    return render(request, template_name)
    
# welcome to custom model
def TrainStatistic(request):

    # declaring template
    template_name = "DataUploadCustomMethod_train.html"
    data = UserDatabase.objects.all()
    
    # prompt is a context variable that can have different values         depending on their context
    prompt = {
    
        'order': 'Order of the CSV should be OFFRNARR,CRSCATGRYINT,CRSHDATE,CRASHTIME,CRSCATGRY,CRSHTYPE,CNTYNAME,MUNINAME,MUNITYPE,INJSVR,LATDECDG,LONDECDG',
        'profiles': data
            }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template_name, prompt)
       
    elif request.method == 'POST':
        emailId=request.POST.get('email')
        dataId=request.POST.get('dataID')
        label=request.POST.get('label')        
        narrative=request.POST.get('narrative') 
        print('i am here')
        #print(emailId,dataId,label,narrative)
        

        sesID=emailId+ dataId  
        #print(   'sesID: ',sesID)     
        #ret=DistractedUnigram.objects.raw('select * FROM "CIEACTapp_distractedunigram"')

        print(request.POST.getlist('modelname'))     
        csv_file = request.FILES['file']        
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect(reverse('fileupload'))
        import pandas as pd
        df=pd.read_csv(csv_file)
        df=df.dropna(subset=[narrative])
        #print(df.head(3))
        classymatrix,Uni,Bi,narrLen=Robust_Classifier(df,label,narrative)
        request.session['Uni'+sesID] = Uni
        request.session['Bi'+sesID] = Bi
        #print(classymatrix)
        layout = {  
        'title':'jghjjh',        
        'xaxis_title': 'Narrative Length',
        'yaxis_title': 'Frequency',

         'margin': {'t': 3}
        }
        book_name = request.session.get('book_name')
        print(book_name)
        graphs=go.histogram(narrLen,width=500, height=250).update_layout(
                    margin=dict(l=20, r=20, t=0, b=20),
                    yaxis={'title_text':"Frequency"},
                    xaxis={'title_text':"Text Length"},
                    paper_bgcolor="white",
                    showlegend=False,)
                    #title={
                        #'text': "Text Summary",
                        #'y':0.9,
                        #'x':0.5,
                        #'xanchor': 'center',
                        #'yanchor': 'top'}


        # Getting HTML needed to render the plot.
        plot_div = plot({'data': graphs, 'layout': layout,}, 
                    output_type='div')  
        Uni=sorted(Uni.items(),key=lambda x:x[1][0],reverse=True)
        Bi=sorted(Bi.items(),key=lambda x:x[1][0],reverse=True)
        context = {'m':classymatrix,'TOPUni':Uni[:10],'TOPBi':Bi[:10],'plot_div': plot_div,'sesID':sesID}
        messages.success(request, 'CSV file updated.') 
        template = loader.get_template('Trained-Models-Summary-Results.html')        

        return HttpResponse(template.render(context, request))
        
       
        
# use trainned model to classiffy
def UploadViewCustomClassify(request):
 
    if request.method == 'GET':
 
       template = loader.get_template('DataUploadCustomMethod_test.html')
       context  ={'sms':'custom'}            
       return HttpResponse(template.render(context,request))
       
    elif request.method == 'POST':
        #emailId=request.POST.get('email')
        #dataId=request.POST.get('dataID')
        NarrID=request.POST.get('NarrID')        
        narrative=request.POST.get('narrative') 
        sesID=request.POST.get('sesID')              
        cutoff= float(request.POST.get('cutoff')) 
        lat=request.POST.get('lattitude') 
        lon=request.POST.get('longitude') 
       
        print(sesID)

        print('cutoff',cutoff)
        #ret=DistractedUnigram.objects.raw('select * FROM "CIEACTapp_distractedunigram"')
        #U_list,B_list,T_list,hybrid=pullWordProbability(request.POST.getlist('modelname')) 
             
        csv_file = request.FILES['file']        
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect(reverse('fileupload'))
        import pandas as pd
        df=pd.read_csv(csv_file)
        df=df.dropna(subset=[narrative])
        #print(df.head(3))
        #classymatrix=Robust_Classifier(df,label,narrative)
        U_list=request.session['Uni'+sesID]
        #print('ulist:',U_list)
        B_list=request.session['Bi'+sesID]
        
       

        df['NoisyOR']=df[narrative].apply( lambda x:(classifier( cutoff, countWOrdUni(x)[0], countWOrdBi(x)[0], U_list,B_list))[0])
        
        request.session['narrIDName'+sesID]=NarrID
        request.session['narrName'+sesID]=narrative
        request.session['narrID'+sesID]=df[NarrID].tolist()
        request.session['narr'+sesID]=df[narrative].tolist()
        request.session['result'+sesID] = df['NoisyOR'].tolist()
        
        df=df[df['NoisyOR']>=cutoff]
        U_list=dict([(i,j[0]) for i,j in U_list.items()])
        B_list=dict([(i,j[0]) for i,j in B_list.items()])
            
        threshold=0.35
        U_list=dict([(i,j) for i ,j in U_list.items() if j>=threshold])
        B_list=dict([(i,j) for i ,j in B_list.items() if j>=threshold])
        df_latlon=df[[lat,lon]]
        df_latlon.dropna(inplace=True)
  
        latlon=[[i,j] for i,j in zip(df_latlon[lat],df_latlon[lon])]
        df=df.iloc[:2000]
        
        
        df['NoisyORVIZ']=df[narrative].apply(lambda x:textViz(x,U_list,B_list))
        RESULT=[[h,i,round(j,3),str(k)+','+str(l)] for h,i,j,k,l in zip(df[NarrID],df['NoisyORVIZ'],df['NoisyOR'],df[lat],df[lon])]
        #print(RESULT[1])

        
        #print(classymatrix)
        
        context = {'RESULT':RESULT,'mapdata':latlon,'sesID':sesID}
        messages.success(request, 'CSV file updated.') 
        template = loader.get_template('custom_models.html')        

        return HttpResponse(template.render(context, request))



 
        
# use default model to classiffy
def UploadViewDefaultClassify(request):

 
    if request.method == 'GET':
 
       template = loader.get_template('DataUploadDefaultMethod.html')
       context  ={'sms':'custom'}            
       return HttpResponse(template.render(context,request))
       
    elif request.method == 'POST':
        emailId=request.POST.get('email')
        dataId=request.POST.get('dataID')
        sesID=emailId+dataId
        NarrID=request.POST.get('NarrID')        
        narrative=request.POST.get('narrative') 
        cutoff= float(request.POST.get('cutoff')) 
        lat=request.POST.get('lattitude') 
        lon=request.POST.get('longitude')
        modelName=request.POST.getlist('modelname')
        print( modelName,sesID)
        
        if  cutoff<0.35:
            cutoff=0.35
        print('cutoff',cutoff)
        #ret=DistractedUnigram.objects.raw('select * FROM "CIEACTapp_distractedunigram"')
        #U_list,B_list,T_list,hybrid=pullWordProbability(request.POST.getlist('modelname')) 
             
        csv_file = request.FILES['file']        
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect(reverse('fileupload'))
        import pandas as pd
        df=pd.read_csv(csv_file)
        df=df.dropna(subset=[narrative])
        #print(df.head(3))
        #classymatrix=Robust_Classifier(df,label,narrative)
        U_list,B_list,T_list,hybrid=pullWordProbability(modelName)

        print('ulist:', U_list,B_list,T_list,hybrid)       

        df['NoisyOR']=df[narrative].apply( lambda x:(classifierDefaultModel( cutoff, countWOrdUni(x)[0], countWOrdBi(x)[0], U_list,B_list))[0])

        request.session['narrIDName'+sesID]=NarrID
        request.session['narrName'+sesID]=narrative
        request.session['narrID'+sesID]=df[NarrID].tolist()
        request.session['narr'+sesID]=df[narrative].tolist()
        request.session['result'+sesID] = df['NoisyOR'].tolist()
        

        
        df=df[df['NoisyOR']>=cutoff]
        U_list=dict([(i,j) for i,j in U_list.items()])
        B_list=dict([(i,j) for i,j in B_list.items()])
            
        threshold=0.35
        U_list=dict([(i,j) for i ,j in U_list.items() if j>=threshold])
        B_list=dict([(i,j) for i ,j in B_list.items() if j>=threshold])
        df_latlon=df[[lat,lon]]
        df_latlon.dropna(inplace=True)
  
        latlon=[[i,j] for i,j in zip(df_latlon[lat],df_latlon[lon])]
        df=df.iloc[:2000]
        
        
        df['NoisyORVIZ']=df[narrative].apply(lambda x:textViz(x,U_list,B_list))
        RESULT=[[h,i,round(j,3),str(k)+','+str(l)] for h,i,j,k,l in zip(df[NarrID],df['NoisyORVIZ'],df['NoisyOR'],df[lat],df[lon])]
        #print(RESULT[1])

        
        #print(classymatrix)
        
        context = {'RESULT':RESULT,'mapdata':latlon,'sesID':sesID}
        messages.success(request, 'CSV file updated.') 
        template = loader.get_template('default_models.html')      
        return HttpResponse(template.render(context, request))        
        
        
def send_file(request):

  import os, tempfile, zipfile
  from wsgiref.util import FileWrapper
  from django.conf import settings
  import mimetypes,io
  import pandas as pd
  sesID=request.POST.get('sesID')   
  print(sesID)
  uni=request.session['Uni'+sesID]
  bi=request.session['Bi'+sesID]
  ngram={**uni,**bi }

  results=pd.DataFrame.from_dict(ngram).T
  results=results.reset_index(level=0)
  results.columns=['token','probability','posCount','negCount']
  
  response = HttpResponse(content_type='text/csv')
  response['Content-Disposition'] = 'attachment; filename=wordProbability.csv'
  results.to_csv(path_or_buf=response,index=None)
  return response
  
def result_dload(request):

  import os, tempfile, zipfile
  from wsgiref.util import FileWrapper
  from django.conf import settings
  import mimetypes,io
  import pandas as pd
  sesID=request.POST.get('sesID')  
  print(sesID) 
  narrIDName=request.session['narrIDName'+sesID]
  narrName=request.session['narrName'+sesID]
  res=request.session['result'+sesID]
  narrID=request.session['narrID'+sesID]
  narr=request.session['narr'+sesID]
  
  
  dfres = pd.DataFrame({narrIDName:narrID,narrName:narr,'NoisyOR':res})
  response = HttpResponse(content_type='text/csv')
  response['Content-Disposition'] = 'attachment; filename=results.csv'
  dfres.to_csv(path_or_buf=response,index=None)
  return response

def classifiedPage(request):

    # declaring template
    template_name = "first page.html"
    data = UserDatabase.objects.all()
    request.session['book_name'] = 'Sherlock Holmes'
    
    # prompt is a context variable that can have different values         depending on their context
    prompt = {
    
        'order': 'Order of the CSV should be OFFRNARR,CRSCATGRYINT,CRSHDATE,CRASHTIME,CRSCATGRY,CRSHTYPE,CNTYNAME,MUNINAME,MUNITYPE,INJSVR,LATDECDG,LONDECDG',
        'profiles': data
            }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template_name, prompt)
    elif request.method == 'POST':
        emailId=request.POST.get('email')
        dataId=request.POST.get('dataID')
        label=request.POST.get('label')        
        narrative=request.POST.get('narrative') 
        print(emailId,dataId,label,narrative)        
        #ret=DistractedUnigram.objects.raw('select * FROM "CIEACTapp_distractedunigram"')
        U_list,B_list,T_list,hybrid=pullWordProbability(request.POST.getlist('modelname')) 
        print(request.POST.getlist('modelname'))     
        csv_file = request.FILES['file']        
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect(reverse('fileupload'))
        import pandas as pd
        df=pd.read_csv(csv_file)
        df=df.dropna(subset=[narrative])
        #print(df.head(3))
        classymatrix=Robust_Classifier(df,label,narrative)
        #print(classymatrix)
        
        context = {'m':classymatrix,}
        messages.success(request, 'CSV file updated.') 
        template = loader.get_template('models.html')        

        return HttpResponse(template.render(context, request))