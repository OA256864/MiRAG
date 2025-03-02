"""
@author: Omar
"""
import os
import datasets
import os
import json
from utils.utils import remove_unused_columns

class data_processor:
   def __init__(self, args, kwargs):
      super().__init__() 
      
      self.kwargs = kwargs
      self.args = args
      self.input_key = None            

      ''' Datesets, KB '''
      self.train_dataset = None
      self.dev_dataset = None
      self.test_dataset = None
      self.kb = None
      self.passages = None
      ''' loaders '''
      self.train_dataloader = None
      self.dev_dataloader = None
      self.test_dataloader = None
      self.train_len = None
      self.map_entity_descriptions = {}      
      self.viquae_dev_dataset = None
      self.new_vocab  = {}
      self.article2passage = None

      self.dataset_path = self.kwargs['data_module_kwargs']['data_processor'].get('dataset_path', None)
      self.train_path = self.kwargs['data_module_kwargs']['data_processor'].get('train_path', None)
      self.val_path = self.kwargs['data_module_kwargs']['data_processor'].get('val_path', None)
      self.test_path = self.kwargs['data_module_kwargs']['data_processor'].get('test_path', None)

      self.kb_path = self.kwargs['data_module_kwargs']['data_processor'].get('kb_path', None)
      self.entity_kb_path = self.kwargs['data_module_kwargs']['data_processor'].get('entity_kb_path', None)
      self.cross_modal_validation_dataset_path = self.kwargs['data_module_kwargs']['data_processor'].get('cross_modal_validation_dataset_path', None)  

      if self.kwargs['data_module_kwargs']['data_processor'].get('article2passage', None):
         self.article2passage = json.load( open(self.kwargs['data_module_kwargs']['data_processor'].get('article2passage', None), 'r'))
   
   def load_data(self,):        
       raise NotImplementedError("Subclass and implement load_data.")   

class viquae_data_processor(data_processor):
   def __init__(self, args, kwargs):
      super().__init__(args, kwargs) 

   def load_data(self,):
      if self.dataset_path:          
         self.train_dataset = datasets.load_from_disk( os.path.join(self.dataset_path,'train') , keep_in_memory=self.args.keep_in_memory) 
         self.dev_dataset = datasets.load_from_disk( os.path.join(self.dataset_path,'val') , keep_in_memory=self.args.keep_in_memory) 
         self.test_dataset = datasets.load_from_disk( os.path.join(self.dataset_path,'test') , keep_in_memory=self.args.keep_in_memory)          
      else:
          if  self.train_path:
              self.train_dataset = datasets.load_from_disk( self.train_path , keep_in_memory=self.args.keep_in_memory) 
              self.train_dataset = remove_unused_columns( self.train_dataset, to_be_removed=['clip-RN50']) 

          if  self.val_path:    
              self.dev_dataset = datasets.load_from_disk( self.val_path , keep_in_memory=self.args.keep_in_memory) 
              self.dev_dataset = remove_unused_columns( self.dev_dataset, to_be_removed=['clip-RN50']) 

          if  self.test_path:
              self.test_dataset = datasets.load_from_disk( self.test_path , keep_in_memory=self.args.keep_in_memory)
              self.test_dataset = remove_unused_columns( self.test_dataset, to_be_removed=['clip-RN50'])        

      if  self.kb_path:
          self.kb = datasets.load_from_disk( self.kb_path , keep_in_memory=self.args.keep_in_memory) 

      if self.entity_kb_path:    
         self.entity_kb = datasets.load_from_disk( self.entity_kb_path , keep_in_memory=self.args.keep_in_memory) 
         self.entity_kb = remove_unused_columns( self.entity_kb, to_keep=['passage_index', 'wikidata_info', 'entity_type_qid', 'entity_type', 'image', 'wikipedia_title']) 