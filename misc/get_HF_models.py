"""
@ Author: Omar
"""
from transformers import BertModel, BertTokenizerFast
from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import CLIPProcessor, CLIPModel,CLIPImageProcessor,CLIPTokenizer,CLIPTextModel,CLIPVisionModel,CLIPFeatureExtractor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import  Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer
import os

model_mappings = {   

'bert-base-uncased':{'model':BertModel ,'config':None, 'tokenizer': BertTokenizerFast, 'path' : 'bert-base-uncased' },

'dpr_question_encoder_triviaqa_without_viquae':{'model':DPRQuestionEncoder ,'config':None, 'tokenizer': None, 'path' : 'PaulLerner/dpr_question_encoder_triviaqa_without_viquae' },

'dpr_context_encoder_triviaqa_without_viquae':{'model':DPRContextEncoder ,'config':None, 'tokenizer': None, 'path' : 'PaulLerner/dpr_context_encoder_triviaqa_without_viquae' },

'clip-vit-base-patch32':{'model':CLIPModel ,'config':None, 'tokenizer': CLIPProcessor, 'path' : 'openai/clip-vit-base-patch32' },                               

't5_small':{'model':T5ForConditionalGeneration ,'config':None, 'tokenizer': T5Tokenizer, 'path' : 'google-t5/t5-small' },            

't5_large':{'model':T5ForConditionalGeneration ,'config':None, 'tokenizer': T5Tokenizer, 'path' : 'google-t5/t5-large' },             

'blip2':{'model':Blip2ForConditionalGeneration ,'config':None, 'tokenizer': Blip2Processor, 'path' : 'Salesforce/blip2-flan-t5-xl' },                  
               }

def save_tokenizer_model_to_local(transf_model_map, model_name, cache_dir, kwargs={}):    
    if not os.path.exists(os.path.join(cache_dir, model_name)):
       os.makedirs(os.path.join(cache_dir, model_name))  
    
    if transf_model_map[ model_name ]['tokenizer'] != None:
      if  transf_model_map[ model_name ].get('tokenizer_path', None):
          model = transf_model_map[ model_name ]['tokenizer'].from_pretrained( transf_model_map[ model_name ]['tokenizer_path'], **kwargs)
      else:
        if  transf_model_map[ model_name ]['path'] != None:
                model = transf_model_map[ model_name ]['tokenizer'].from_pretrained( transf_model_map[ model_name ]['path'], **kwargs)           
        else:
                model = transf_model_map[ model_name ]['tokenizer'].from_pretrained(model_name, **kwargs)           
      model.save_pretrained(os.path.join(cache_dir, model_name))
       
    if transf_model_map[ model_name ]['model'] != None:  
       if  transf_model_map[ model_name ]['path'] != None:
           model = transf_model_map[ model_name ]['model'].from_pretrained( transf_model_map[ model_name ]['path'], **kwargs)           
       else:
           model = transf_model_map[ model_name ]['model'].from_pretrained(model_name, **kwargs)           
       
       model.save_pretrained(os.path.join(cache_dir, model_name))       

def save_HF_models( local_cache_dir, models_key, kwargs={}): 
   save_tokenizer_model_to_local(model_mappings, models_key, local_cache_dir, kwargs)     

   
