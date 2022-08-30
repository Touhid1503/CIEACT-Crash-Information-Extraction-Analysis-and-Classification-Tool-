from django.conf.urls import  include
from django.urls import path
from .views import aboutPage,helpPage,homePage, contactPage,TrainStatistic,UploadViewDefaultClassify,UploadViewCustomClassify,send_file,classifiedPage,result_dload,UploadViewDefaultClassify,customTrainTestUpload,UpldCvsFORTrn,DataUploadDefaultMethod#,modelRun

urlpatterns = [
   #path('train/statistic', UploadView2, name = 'Train-button'),# welcome to custom model
   path('train/', UpldCvsFORTrn, name = 'Train-button'),# welcome to custom model
   path('default/', DataUploadDefaultMethod, name = 'default_data_upload'),# welcome to default model
   path('default/classify', UploadViewDefaultClassify,name='classify'),# welcome to default model
   path(r'train/statistics', TrainStatistic,name='TrainStatistic'),
   path(r'about', aboutPage,name='aboutPage'),
   path(r'help', helpPage,name='helpPage'),
   path(r'contact', contactPage,name= 'contactPage'),
   path(r'train/statistics/trainClassify', customTrainTestUpload,name='Trainclassify'),
   path(r'train/statistics/trainClassify/testclassify', UploadViewCustomClassify,name='classify'),
   #path('classifyD/', UploadViewDefaultClassify,name='classifyD'),
   path('', homePage,name='homePage'),#welcome page

   path('train/statistics/k?a2', send_file,name='csv_download'),
   path('train/statistics/trainClassify/testclassify/k?oa', result_dload,name='result_dload'),
   #path('train/traincustomModel/', UploadViewCustomClassify,name='UploadViewCustomClassify'),# trained model to test classify
   #path('modelsurl', modelRun,name='runmodel'),


  ]     