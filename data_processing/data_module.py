"""
Created on Sun May 29 10:12:23 2022

@author: Omar
"""
import os
import utils
from torch.utils.data import (DataLoader)
import pytorch_lightning as pl
import torch
import os 
from data_processing.batch_factory import * 
from misc import get_HF_models
import utils
from utils import utils
from transformers import RobertaTokenizer,XLMRobertaTokenizer,XLNetTokenizer,BertTokenizer,BertTokenizerFast, CLIPProcessor, CLIPModel,CLIPFeatureExtractor,CLIPTokenizer,CLIPTokenizerFast
from transformers import BlipModel, BlipProcessor
from utils.image.image_processing import *

'''======================================================================================'''
def get_pretrained_model(args, **kwargs):
    def load_checkpoint(pretrained_model, args,**kwargs ):  
        if 'checkpoint' in kwargs: # in case of resuming training from chpt, splitted encoders might be missing          
            pretrained_model.load_state_dict(torch.load( kwargs['checkpoint'] ), strict=True)         
            print('loaded checkpoint:', kwargs['checkpoint'], '\n')  
              
        return pretrained_model
    
    Class = get_class(args, **kwargs)      
    if Class.__bases__[0] is nn.Module:   
    #    print('base classes', Class.__bases__[0])
       pt_model = Class(args,**kwargs) 
       pt_model = load_checkpoint(pt_model, args,**kwargs )
       return pt_model       
    else: 
        if 'pretrained_model_name_or_path' in kwargs:
           loacal_path = os.path.join(args.local_cache, os.path.basename(kwargs['pretrained_model_name_or_path']) )        
           pt_model = Class.from_pretrained(loacal_path)
           pt_model = load_checkpoint(pt_model, args,**kwargs )
        elif 'base_encoder_kwargs' in kwargs:
            loacal_path = os.path.join(args.local_cache, os.path.basename(kwargs['base_encoder_kwargs']['pretrained_model_name_or_path']) )        
            pt_model = Class.from_pretrained(loacal_path)
            pt_model = load_checkpoint(pt_model, args,**kwargs )
        else:            
            configuration_class = get_config_class(args, **kwargs) 
            configuration = configuration_class()        
            pt_model = Class(configuration)        
     
    return pt_model   

def get_class(args, class_name, **kwargs):
    modules = dict(globals().items())    
    Class = modules[class_name]                       
    return Class 

def get_config_class(args, config_class_name, **kwargs):
    modules = dict(globals().items())    
    Class = modules[config_class_name]                       
    return Class       
'''======================================================================================'''  
class DataModule(pl.LightningDataModule):    
    def __init__(self, args, dp, all_kwargs, dataloader_kwargs={}  ):
      super().__init__()

      # NOTE in multigpu lr = lr*nbr_gpu
      self.args = args      
      self.input_key ='input' 
      self.dp = dp  
      self.kwargs = all_kwargs['data_module_kwargs'] 
      self.all_kwargs = all_kwargs

      self.use_image = self.kwargs.get('use_image', None) 
      self.imageFormatter = None
      self.train_batch_size = args.train_batch_size 
      self.dataloader_kwargs = dataloader_kwargs

      if  self.kwargs.get('tokenizer_kwargs', None):
          if self.kwargs['tokenizer_kwargs'].get('pretrained_model_name_or_path', None):          
             self.tokenizer = get_pretrained_model(args,**self.kwargs['tokenizer_kwargs'])        
          else:              
              tokenizer_class = get_class(args, **self.kwargs['trainee_kwargs'])
              self.tokenizer = tokenizer_class(args, **self.kwargs['trainee_kwargs'])    
          # is BLIP processor    
          if self.kwargs['tokenizer_kwargs']['class_name'] == 'BlipProcessor' or self.kwargs['tokenizer_kwargs']['class_name'] == 'Blip2Processor':               
             self.tokenizer = self.tokenizer.tokenizer             
          elif self.kwargs['tokenizer_kwargs']['class_name'] == 'FlavaProcessor':               
              self.tokenizer = self.tokenizer.tokenizer
      else:
        get_HF_models.save_HF_models( args.local_cache,  self.args.transformer_model_name)    
        self.tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)                   
      
      get_HF_models.save_HF_models( args.local_cache,  self.args.transformer_model_name) 
      self.bert_tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)                   
     
      if self.kwargs.get('image_processor_kwargs', None):
        image_processor_class = get_class(args, **self.kwargs['image_processor_kwargs'])
        self.imageFormatter = image_processor_class(args, self.kwargs['image_processor_kwargs'])  

      if self.kwargs.get('validation_image_processor_kwargs', None): # corner case: different image processor for validation set
        image_processor_class = get_class(args, **self.kwargs['validation_image_processor_kwargs'])
        self.validation_imageFormatter = image_processor_class(args, self.kwargs['validation_image_processor_kwargs'])    
   
    def prepare_data(self,):
      if self.args.train_sanity_run:
         self.dp.train_dataset = self.dp.train_dataset.select([0])
         
      if self.args.sanity_run:
         self.args.num_train_epochs = 1
         from torch.utils.data import Subset
         self.dp.train_dataset = self.dp.train_dataset.select( list(range( min( len(self.dp.train_dataset), 2 * self.args.train_batch_size)  )) )
         self.dp.dev_dataset = self.dp.dev_dataset.select( list(range( min(len(self.dp.dev_dataset), 2 * self.args.eval_batch_size) )) )         
         if self.dp.cross_modal_validation_dataset_path != None:            
            self.dp.cross_modal_validation_dataset = self.dp.cross_modal_validation_dataset.select( list(range(2 * self.args.eval_batch_size)) )
         if self.dp.test_dataset != None:
            self.dp.test_dataset = Subset( self.dp.test_dataset, list(range( 2 * self.args.eval_batch_size)))

    def train_dataloader(self):        
        return DataLoader( self.dp.train_dataset, batch_size=self.train_batch_size, shuffle = False, collate_fn=self.collate_fn, **self.dataloader_kwargs)

    def val_dataloader(self):
        if self.dp.dev_dataset != None:         
           return DataLoader( self.dp.dev_dataset, batch_size=self.args.eval_batch_size, shuffle = False, collate_fn=self.collate_fn, **self.dataloader_kwargs)   
        else:
            return None

    def test_dataloader(self):
        if self.dp.test_dataset != None:        
           return DataLoader( self.dp.test_dataset, batch_size=self.args.eval_batch_size, shuffle = False, collate_fn=self.collate_fn, **self.dataloader_kwargs)   
        else:
           return None

    def predict_dataloader(self):
        return  DataLoader( self.kwargs['predict_dataset'], batch_size=self.args.eval_batch_size, shuffle = False, collate_fn=self.collate_fn, **self.dataloader_kwargs) 



class multi_modal_DataModule(DataModule): 
    """
    Parameters
    ----------
    """
    def __init__(self, args, dp, all_kwargs, dataloader_kwargs={}):
        super().__init__(args, dp, all_kwargs, dataloader_kwargs={})         
        # self.tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)                   
        self.batch_index = -1 
        self.all_kwargs = all_kwargs
        self.kwargs = all_kwargs['data_module_kwargs']
        if self.kwargs.get('augmented', None): 
           # Add  special tokens
           self.tokenizer.add_tokens(['[MASK]'], special_tokens=True)       
           self.augment_vocab()       
    
    def collate_fn(self, items):         
        batch = build_multimodal_batch( self.args, items, self.tokenizer, self.dp, self.batch_index, self.imageFormatter, self.kwargs,
                                        self.trainer.training, all_kwargs = self.all_kwargs)         
        return batch




