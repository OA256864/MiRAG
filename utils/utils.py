"""
@ Author: Omar
"""
import os
import json
from pytorch_lightning import seed_everything
import numpy as np
import torch
import random
from models.trainee import *
from transformers.tokenization_utils_base import BatchEncoding
import logging
logging.basicConfig(format = '%(message)s', level =  logging.INFO)
logger = logging.getLogger(__name__)
from transformers import BertTokenizerFast

tokenizer_map= {            
            'bert-base-uncased':BertTokenizerFast ,            
            }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)       
    seed_everything(seed)

def update_experiment_params(args,experiment_dir):
   new_param_dicts = {k:v for k,v in args.__dict__.items()}     
   with open( os.path.join( experiment_dir, 'experiment_params.json') , 'w') as configfile:
      json.dump(new_param_dicts, configfile, indent=2)


def get_device():
    if torch.cuda.is_available():
       device_nbr = torch.cuda.current_device()
       logger.info('cuda.current_device() = %s', torch.cuda.current_device())   
       logger.info('torch.cuda.get_device_name(device_nbr) = %s', torch.cuda.get_device_name(device_nbr))     
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    logger.info('device = %s', device)    
    return device

device = get_device()
def prepare_inputs(data):
    """
    Moves tensors in data to `device`, be it a tensor or a nested list/dictionary of tensors.
    Adapted from transformers.Trainer
    """
    if isinstance(data, (dict, BatchEncoding)):
        # N. B. BatchEncoding does not accept kwargs, not sure what happens in Trainer
        return dict(**{k: prepare_inputs(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        raise TypeError(f"Unexpected type '{type(data)}' for data:\n{data}")
    return data

def update_args(args, json_dict, excluded_args=[]):
  for key_arg in json_dict:
    if key_arg not in excluded_args:
      vars(args)[key_arg] = json_dict[key_arg]
  
  return args  

def create_experiment_dir(args, experiment_dir, **kwargs):
   
   if os.path.exists(experiment_dir):   
      for i in range(1000):
         if i > 99:
            print('\ntoo many same experiments dirs')
            exit()
         new_experiment_dir = experiment_dir + '_' + str(i)
         new_output_dir = args.output_dir + '_' + str(i)
         if not(os.path.exists(new_experiment_dir)):
            experiment_dir = new_experiment_dir 
            args.output_dir = new_output_dir
            os.makedirs(new_experiment_dir)
            break
   else:
      os.makedirs(experiment_dir)

   excluded_params = ['nbr_gpus', 'new_vocab_embeddings']
   param_dicts = {k:v for k,v in args.__dict__.items() if k not in excluded_params }
   # if not(os.path.exists(os.path.join( experiment_dir, 'experiment_params.json'))):
   with open( os.path.join( experiment_dir, 'experiment_params.json') , 'w') as configfile:
      json.dump(param_dicts, configfile, indent=2)

   with open( os.path.join( experiment_dir, 'config_kwargs.json') , 'w') as configfile:
      for model_kwargs_key in kwargs:         
         if 'model' in model_kwargs_key or  'encoder' in model_kwargs_key or 'generator' in model_kwargs_key or 'reader' in model_kwargs_key and 'base' not in model_kwargs_key:     
            if 'checkpoint_name' in kwargs[model_kwargs_key] :
               kwargs[model_kwargs_key]['inference_path'] =  os.path.join(*[ args.output_dir , kwargs[model_kwargs_key]['checkpoint_name'] ])
            else:
               kwargs[model_kwargs_key]['inference_path'] =  args.output_dir 
      json.dump(kwargs, configfile, indent=2)   
   return experiment_dir  


def append_dict_list_values(target_dict , source_dict):
   for k in source_dict:
      if k not in target_dict:
         target_dict[k] = []
      target_dict[k].append(source_dict[k])   
   return target_dict   

def extend_dict_list_values(target_dict , source_dict):
   for k in source_dict:
      if k not in target_dict:
         target_dict[k] = []
      target_dict[k].extend(source_dict[k]) 
   return target_dict   

def concat_dict_list_values(dict_1 , dict_2):
   concat_dict = {}
   for k in dict_1:
      concat_dict[k] = dict_1[k] +  dict_2[k]  
   return concat_dict 


def format_kawargs(all_kwargs, args):
      # set full path for pretrained models
   
   if 'pretrained_model_name_or_path' in all_kwargs:
      all_kwargs['pretrained_model_name_or_path'] = os.path.join(args.local_cache, all_kwargs['pretrained_model_name_or_path'] )
   if 'base_encoder_kwargs' in all_kwargs:
      if 'pretrained_model_name_or_path' in all_kwargs['base_encoder_kwargs']:
         all_kwargs['base_encoder_kwargs']['pretrained_model_name_or_path'] = os.path.join(args.local_cache, all_kwargs['base_encoder_kwargs']['pretrained_model_name_or_path'] )            

   if args.checkpoint != None:                       
      # model_type_name = model_kwargs_key.replace('_kwargs', '') # question_model, context_model... 
      if 'checkpoint_name' in all_kwargs:                     
         checkpoint_name = all_kwargs['checkpoint_name']    
         print('\nAssert any checkpoint_name ', checkpoint_name, ' in ', os.listdir( args.checkpoint), '\n')      
         assert(any(checkpoint_name in  os.listdir( args.checkpoint) for f in  os.listdir( args.checkpoint)))          
         for file_or_folder in os.listdir( args.checkpoint):                 
            if file_or_folder == checkpoint_name:
               all_kwargs['checkpoint'] =  os.path.join(*[args.checkpoint , checkpoint_name , 'pytorch_model.bin'])  
   else:
      True
      # if 'checkpoint' in all_kwargs:            
      #    all_kwargs['checkpoint'] =  os.path.join(*[ all_kwargs['checkpoint'] , 'pytorch_model.bin']) 
         

def get_checkpoint_file_name(checkpoint_path):
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            if 'latest' in file:
                continue
            return file

def set_checkpoint_kwargs_embedding(all_kwargs, args):
   for model_kwargs_key in all_kwargs:         
      if 'model' in model_kwargs_key or  'encoder' in model_kwargs_key or 'generator' in model_kwargs_key or 'reader' in model_kwargs_key and 'base' not in model_kwargs_key:                              
         # all_kwargs[model_kwargs_key]['checkpoint'] =  os.path.join(*[ all_kwargs[model_kwargs_key]['inference_path'] , 'pytorch_model.bin'])     
         all_kwargs[model_kwargs_key]['checkpoint'] =  os.path.join( args.checkpoint, get_checkpoint_file_name(args.checkpoint))     
   return all_kwargs

def set_checkpoint_kwargs(all_kwargs, args):
   '''
   Recursively set checkpoints info in all sub models
   '''
   format_kawargs(all_kwargs, args)
   for model_kwargs_key in all_kwargs:         
      if 'model' in model_kwargs_key or  'encoder' in model_kwargs_key or 'generator' in model_kwargs_key or 'reader' in model_kwargs_key and 'base' not in model_kwargs_key:
         # if 'inference_path' in all_kwargs[model_kwargs_key]:
         #    all_kwargs[model_kwargs_key]['checkpoint'] =  all_kwargs[model_kwargs_key]['inference_path']
         #    continue
         format_kawargs(all_kwargs[model_kwargs_key], args)
         for sub_model_kwargs_key in all_kwargs[model_kwargs_key]:               
            if 'model' in sub_model_kwargs_key or  'encoder' in sub_model_kwargs_key  and 'base' not in sub_model_kwargs_key:                  
               temp_kwargs = set_checkpoint_kwargs(all_kwargs[model_kwargs_key][sub_model_kwargs_key], args)                  
               all_kwargs[model_kwargs_key][sub_model_kwargs_key] = temp_kwargs
   return all_kwargs  


def build_embedding_kwargs(args, input_kwargs=None):
   if input_kwargs:
      kwargs = set_checkpoint_kwargs_embedding(input_kwargs, args) 
   else:
      kwargs = json.load(open(args.config)) 
      kwargs = set_checkpoint_kwargs_embedding(kwargs, args) 
   # print('\n',kwargs, '\n')
   return kwargs

def build_kwargs(args):   
   kwargs = json.load(open(args.config)) 
   for model_kwargs_key in kwargs:         
      if 'model' in model_kwargs_key or  'encoder' in model_kwargs_key or 'generator' in model_kwargs_key or 'reader' in model_kwargs_key and 'base' not in model_kwargs_key:                                      
         if args.checkpoint != None: 
            kwargs[model_kwargs_key]['checkpoint'] =  os.path.join( args.checkpoint, get_checkpoint_file_name(args.checkpoint))     
            kwargs[model_kwargs_key].pop('checkpoint_prefix', None)
   
   # exit()
   # kwargs = set_checkpoint_kwargs(kwargs, args)
   # print('\n',kwargs, '\n')
   return kwargs

def build_split_dual_encoders_kwargs(args):   
   kwargs = json.load(open(args.config))   
   return kwargs 


def remove_unused_columns( DaTAsEt, to_be_removed=None, to_keep=None):
      '''
      DaTAsEt: input dataset
      to_be_removed: list of columns' dataset to discard after loading
      to_keep: list of columns' dataset to keep after loading
      '''
      if to_be_removed != None:
         removable_columns = []
         for c in to_be_removed:
             if c in DaTAsEt.column_names:
                removable_columns.append(c)
                
         DaTAsEt = DaTAsEt.remove_columns(removable_columns)
         return DaTAsEt
      elif to_keep != None:
         removable_columns = []
         for c in DaTAsEt.column_names:
             if c not in to_keep:
                removable_columns.append(c)
                
         DaTAsEt = DaTAsEt.remove_columns(removable_columns)
         return DaTAsEt
      else:
           print('please set a list of to_be_removed or to_keep columns')   
           exit()

def get_map_wikidataId_2_passageId(kb, output_file_name):
    map_wikidataId_2_passageId = {}
    for idx, item in enumerate(kb):
        wikidata_id = item['wikidata_info']['wikidata_id']
        map_wikidataId_2_passageId[wikidata_id] = item['passage_index']
    json.dump(map_wikidataId_2_passageId, open(output_file_name, "w"))  

def get_map_wikidataId_2_KB_Id(kb, output_file_name):
    map_wikidataId_2_KB_Id = {}
    for idx, item in enumerate(kb):
        wikidata_id = item['wikidata_info']['wikidata_id']
        map_wikidataId_2_KB_Id[wikidata_id] = idx
    json.dump(map_wikidataId_2_KB_Id, open(output_file_name, "w")) 










