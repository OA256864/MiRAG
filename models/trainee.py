import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import NLLLoss,CrossEntropyLoss, KLDivLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput, ModelOutput
from transformers import BertForQuestionAnswering
from transformers import DPRContextEncoder,DPRQuestionEncoder, DPRConfig
from models.losses import *
import os
import timeit
import json

from transformers import CLIPProcessor, CLIPModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoImageProcessor
from utils import utils
from utils.utils import prepare_inputs as prepare_inputs_device
from utils.image.image_processing import *
import sys
from pathlib import Path    

try:
  if len(Path(os.getcwd()).resolve().parents) > 1:
    meerqat_main_path =  os.path.join( Path(os.getcwd()).resolve().parents[2], "meerqat_main")
    if os.path.exists(meerqat_main_path):
       print('meerqat_main_path',meerqat_main_path)
       sys.path.append(str(meerqat_main_path))
       from meerqat.ir.metrics import find_relevant
       from meerqat.data.loading import answer_preprocess
       from meerqat.train.optim import _calc_mml
       from meerqat.models.qa import get_best_spans 
       from meerqat.ir.metrics import find_valid_numerical_answers
       from meerqat.data.infoseek import QuestionType
except:
   print("No meerqat_main rep")

import pickle
import ranx
from misc import get_HF_models
import datasets



'''=================================   Load pretraind checkpoints ====================================================='''
def build_new_state_dict(model_state_dict, old_key_string, new_key_string):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items(): 
        if old_key_string in k:            
           new_key = k.replace(old_key_string, new_key_string)
           new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def remove_prefix_state_dict(pretrained_state_dict, encoder_prefix):      
   encoder_state_dict = {k[len(encoder_prefix):]: v for k, v in pretrained_state_dict.items() if k.startswith(encoder_prefix)}
   return encoder_state_dict   

def apply_lora(pretrained_model, args, **kwargs):
    from peft import LoraConfig, get_peft_model, TaskType
    from peft import prepare_model_for_int8_training
    lora_config = LoraConfig(r=8,lora_alpha=32,target_modules=["q", "v"],lora_dropout=0.1,bias="none")
    pretrained_model = prepare_model_for_int8_training(pretrained_model)
    # pretrained_model = prepare_model_for_kbit_training(pretrained_model)
    pretrained_model.enable_input_require_grads()
    # add LoRA adaptor
    pretrained_model = get_peft_model(pretrained_model, lora_config)
    return pretrained_model

def get_pretrained_model(args, **kwargs):
    lora_num_param = 1e9
    def load_checkpoint(pretrained_model, args,**kwargs ): 
            if 'checkpoint' in kwargs:                
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                if args.cpu:
                   device = torch.device('cpu') 
                
                num_parameters = sum(p.numel() for p in pretrained_model.parameters())    
                if  num_parameters > lora_num_param: 
                    pretrained_model = apply_lora(pretrained_model, args, **kwargs) 

                checkpoint = torch.load( kwargs['checkpoint'] ,  map_location=device)
                pretrained_dict = checkpoint['state_dict']                               
                
                # Remove prefix from state dict
                if 'checkpoint_prefix' in kwargs:
                    new_state_dict = remove_prefix_state_dict(pretrained_dict,  kwargs['checkpoint_prefix'])
                    pretrained_model.load_state_dict(new_state_dict, strict=False if args.non_strict_load else True)
                    print('\n[LOADED CHECKPOINT]:', '[', kwargs['checkpoint'], ']', '[', kwargs['checkpoint_name'], ']' ,'\n')   
                    return pretrained_model
                else:
                    # select specific submodel parameters                    
                    if any(kwargs['checkpoint_name'] in  k for k, v in pretrained_dict.items() ):
                       pretrained_dict = {k: v for k, v in pretrained_dict.items() if  kwargs['checkpoint_name'] in k}               

                       if any('base_model.model.base_model.model' in  k for k, v in pretrained_dict.items() ):
                          new_state_dict = remove_prefix_state_dict(pretrained_dict, 'model.' + kwargs['checkpoint_name'] + '.' + 'base_model.model.')
                       else:
                           new_state_dict = remove_prefix_state_dict(pretrained_dict, 'model.' + kwargs['checkpoint_name'] + '.')
                
                       pretrained_model.load_state_dict(new_state_dict, strict=False if args.non_strict_load else True)
                       print('\n[LOADED CHECKPOINT]:', '[', kwargs['checkpoint'], ']', '[', kwargs['checkpoint_name'], ']' ,'\n')   
                       return pretrained_model
                    else:
                        return pretrained_model
            else:
                return pretrained_model
    
    Class = get_class(args, **kwargs)      
    if Class.__bases__[0] is nn.Module:       
       if 'model_kwargs' in kwargs:  
            try:         
               pt_model = Class(args,**kwargs['model_kwargs'])            
            except:
                  pt_model = Class(**kwargs['model_kwargs'])            
       else:
           pt_model = Class(args,**kwargs) 
    else: 
        if 'pretrained_model_name_or_path' in kwargs:
           # save HF in local cache
           get_HF_models.save_HF_models( args.local_cache,  os.path.basename(kwargs['pretrained_model_name_or_path']), kwargs.get('pretrained_model_kwargs', {})) 
           loacal_path = os.path.join(args.local_cache, os.path.basename(kwargs['pretrained_model_name_or_path']) )                
           pt_model = Class.from_pretrained(loacal_path)

           if 'checkpoint' not in kwargs and hasattr(pt_model, "parameters"): 
              num_parameters = sum(p.numel() for p in pt_model.parameters())                                                 
              if  num_parameters > lora_num_param: 
                  pt_model = apply_lora(pt_model, args, **kwargs)               

        elif 'base_encoder_kwargs' in kwargs:
            loacal_path = os.path.join(args.local_cache, os.path.basename(kwargs['base_encoder_kwargs']['pretrained_model_name_or_path']) )        
            pt_model = Class.from_pretrained(loacal_path)
        else:            
            configuration_class = get_config_class(args, **kwargs) 
            configuration = configuration_class()        
            pt_model = Class(configuration)        

    pt_model = load_checkpoint(pt_model, args,**kwargs )    
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

'''
-------------------------------
CLIP Text Encoder
-------------------------------
'''
class CLIP_Text_Encoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()        
        self.args = args
        self.kwargs = kwargs         
        model = get_pretrained_model(args,**kwargs['base_encoder_kwargs']) 
        self.logit_scale = model.logit_scale
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, return_projection=False, output_attentions =False, **inputs):     
        if return_projection:
            text_outputs = self.text_model(**inputs, output_attentions=output_attentions)
            pooled_output = text_outputs['pooler_output']
            projected_text_outputs = self.text_projection(pooled_output)
            projected_text_outputs = projected_text_outputs / projected_text_outputs.norm(p=2, dim=-1, keepdim=True) 
            if output_attentions:       
               return dict(pooled_output=pooled_output, projected_output=projected_text_outputs, attentions=text_outputs['attentions']) 
            else:
               return dict(pooled_output=pooled_output, projected_output=projected_text_outputs) 
        else:
            text_outputs = self.text_model(**inputs)
            hidden_states = text_outputs['last_hidden_state']
            pooled_output = text_outputs['pooler_output']
            return dict(pooled_output=pooled_output, hidden_states=hidden_states)        

'''
-------------------------------
CLIP Image Encoder
-------------------------------
'''
class CLIP_Image_Encoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()        
        self.args = args
        self.kwargs = kwargs 
        model = get_pretrained_model(args,**kwargs['base_encoder_kwargs'])  
        
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        self.logit_scale = model.logit_scale

    def forward(self, output_attentions=False, **inputs): 
        visual_inputs = {k:inputs[k] for k in inputs if k in ['pixel_values'] }      
        image_outputs = self.vision_model(**visual_inputs, output_attentions=output_attentions)
        hidden_states = image_outputs[0]
        projected_hidden_states = self.visual_projection(hidden_states)   
        projected_image_outputs = self.visual_projection(image_outputs[1])        
        projected_image_outputs = projected_image_outputs / projected_image_outputs.norm(p=2, dim=-1, keepdim=True)
        if output_attentions:
           return projected_image_outputs, projected_hidden_states, image_outputs['attentions'] 
        else:
           return projected_image_outputs, projected_hidden_states   

class CLIP_Encoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()           
        self.args = args
        self.kwargs = kwargs         
        self.text_encoder  =  get_pretrained_model(args,**kwargs['text_encoder_kwargs']) 
        self.image_encoder  = get_pretrained_model(args,**kwargs['image_encoder_kwargs'])                    
        self.log_softmax = nn.LogSoftmax(1) 
        self.loss_fct, self.uniform_L, self.align_L = init_loss_func(args,kwargs)    
        
        if self.kwargs['trainee_kwargs'].get('quantize', None):                  
           VQ_class = get_class(args, kwargs['trainee_kwargs']['codebook_kwargs'].pop('class_name'), **kwargs)
           self.vq = VQ_class(**kwargs['trainee_kwargs']['codebook_kwargs'])

           # init codebook embeddings with input vectors representing KB centroids
           if self.kwargs['trainee_kwargs'].get('codebook_path', None):           
              codebook_path = self.kwargs['trainee_kwargs'].get('codebook_path', None)
              if os.path.exists(codebook_path):
                codebook = torch.load(codebook_path)
                if len(codebook.shape) < 3:
                    codebook = codebook.unsqueeze(0)                
                quantized_text, text_indices, text_commit_loss = self.vq(torch.zeros(1,512))                 
                for rvq in self.vq.rvqs:
                    for lay in rvq.layers:
                        lay._codebook.embed = codebook

    def forward(self, **inputs):         
        text_outputs = self.text_encoder(return_projection=True, **{k:inputs[k] for k in inputs if k  in ['input_ids','attention_mask'] } )        
        image_outputs, image_hidden_states = self.image_encoder(**{k:inputs[k] for k in inputs if k in ['pixel_values'] })

        return dict(text_outputs=text_outputs, image_outputs=image_outputs)

    def compute_loss(self, inputs, pl_model, return_outputs=False):
        local_labels = inputs.pop('labels', None)  # (N, )
        outputs = self(**inputs)               

        text_outputs = outputs['text_outputs']['projected_output'] 
        image_outputs = outputs['image_outputs'] 
        if local_labels == None:
           local_labels  =  torch.range(0,image_outputs.shape[0] - 1,device=image_outputs.device, dtype=int) # Contrastive Labels
       
        if self.kwargs['trainee_kwargs'].get('quantize', None):           
           quantized_text, text_indices, text_commit_loss = self.vq(text_outputs) 
           quantized_image, image_indices, image_commit_loss = self.vq(image_outputs)           
           
           # Sum commitement losses over quantizers
           text_commit_loss = text_commit_loss.sum()
           image_commit_loss = image_commit_loss.sum()
           
           # Reconstruction Loss
           text_rec_loss = (quantized_text - text_outputs).abs().mean()
           image_rec_loss = (quantized_image - image_outputs).abs().mean()
        
           i2t_codebook_loss = loss_forward(self, self.kwargs, inputs, quantized_image, quantized_text, local_labels, CLIP=True ) 
           return i2t_codebook_loss
           i2t_dense_loss = loss_forward(self, self.kwargs, inputs, image_outputs, text_outputs, local_labels, CLIP=True ) 
           total_loss =   i2t_codebook_loss['loss'] + i2t_dense_loss['loss']
           i2t_codebook_loss.update(dict(loss=total_loss))
           return i2t_codebook_loss        

        i2t_loss = loss_forward(self, self.kwargs, inputs, image_outputs, text_outputs, local_labels, CLIP=True ) 
        if self.kwargs['trainee_kwargs'].get('symmetric_CL', None):    
           t2i_loss = loss_forward(self, self.kwargs, inputs, text_outputs, image_outputs, local_labels, CLIP=True )   
           total_loss =   i2t_loss['loss'] + t2i_loss['loss']
           i2t_loss.update(dict(loss=total_loss))

        if self.args.aokvqa and not(self.training):
           i2t_loss.update(dict(image_outputs=image_outputs))
        return i2t_loss


import re
import string

def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def answer_preprocess(answer):
    """Adapted from datasets squad metric. Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(answer.lower())))


class Encoder_Generator(nn.Module):    
    def __init__(self, args, **kwargs):
        super().__init__()
        self._keys_to_ignore_on_save = None
        self.args = args 
        self.kwargs = kwargs 
        self.search_run = None
        self.train_search_run = None
        self.Blip2_imageFormatter = None
        self.qualitative_analysis = open( os.path.join( args.experiment_dir, 'qualitative_analysis.txt' )  , 'w')
        self.batch_idx = 0 

        if os.path.exists(os.path.join('data','wikidata_type_id_2_title.json')):
           self.wikidata_type_id_2_title = json.load( open( os.path.join('data','wikidata_type_id_2_title.json'), 'r'))

        if os.path.exists(os.path.join('data','map_wikidata_id_2_wikidata_type_id.json')):
           self.map_wikidata_id_2_wikidata_type_id = json.load( open( os.path.join('data','map_wikidata_id_2_wikidata_type_id.json'), 'r'))

        if os.path.exists(os.path.join('data','map_entity_descriptions.json')):
           map_entity_descriptions = pickle.load( open( os.path.join('data','map_entity_descriptions'), 'rb'))        
           self.map_descriptions2_wikidataId = {v:k for k, v in map_entity_descriptions.items()}        
        
        if self.kwargs.get('mlp_model_kwargs', None):                                 
           self.mlp_model = get_pretrained_model(args,**kwargs['mlp_model_kwargs']) 

        if self.kwargs.get('mlp_reducer_model_kwargs', None):                                 
           self.mlp_reducer_model = get_pretrained_model(args,**kwargs['mlp_reducer_model_kwargs'])  
           self.LayerNorm = nn.LayerNorm( 128, eps=1e-12)

        if self.kwargs['trainee_kwargs'].get('quantize', None):            
           VQ_class = get_class(args, kwargs['trainee_kwargs']['codebook_kwargs'].pop('class_name'), **kwargs)
           self.vq = VQ_class(**kwargs['trainee_kwargs']['codebook_kwargs'])

        if self.kwargs.get('reader_kwargs', None):                                 
           self.init_reader()          
           self.reader_kwargs = self.kwargs.get('reader_kwargs', {})  
           self.bert_tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)                            
           if self.reader_kwargs.get('search_run', None):                                 
              self.search_run = ranx.Run.from_file(self.reader_kwargs['search_run'])
              print('\nLoaded Search Run:', self.reader_kwargs['search_run'], '\n')              

        if self.kwargs.get('mir_kwargs', None): 
           self.init_mir()
           self.reader_kwargs = self.kwargs.get('mir_kwargs', {})
           self.dpr_tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)
           

        if self.kwargs.get('answer_generator_kwargs', None):
           self.init_generator()
           self.reader_kwargs = self.kwargs.get('answer_generator_kwargs', {})
           if self.reader_kwargs.get('search_run', None):                                 
              self.search_run = ranx.Run.from_file(self.reader_kwargs['search_run'])            
              print('\nLoaded Search Run:', self.reader_kwargs['search_run'], '\n')
           if self.reader_kwargs.get('train_search_run', None):                                 
              self.train_search_run = ranx.Run.from_file(self.reader_kwargs['train_search_run'])            
              print('\nLoaded train Search Run:', self.reader_kwargs['train_search_run'], '\n')   

           if self.reader_kwargs.get('entity_prompt', None):
              self.dpr_tokenizer = utils.tokenizer_map[self.args.transformer_model_name].from_pretrained( os.path.join(args.local_cache , self.args.transformer_model_name ) , truncation=True, do_lower_case=True)                      

        if self.kwargs.get('entity_generator_kwargs', None):
           self.entity_generator  =  get_pretrained_model(self.args,**self.kwargs['entity_generator_kwargs']) 
           self.entity_generator.config.use_cache = False 
           
        if self.kwargs.get('question_model_kwargs', None):
            self.question_model = get_pretrained_model(args,**kwargs['question_model_kwargs'])

        if self.kwargs.get('context_model_kwargs', None):                    
            self.context_model = get_pretrained_model(args,**kwargs['context_model_kwargs']) 

        if self.kwargs.get('image_encoder_kwargs', None):            
            self.image_encoder = get_pretrained_model(args,**kwargs['image_encoder_kwargs'])

        if self.kwargs.get('text_encoder_kwargs', None):
            self.text_encoder = get_pretrained_model(args,**kwargs['text_encoder_kwargs']) 

        if self.kwargs.get('kb_image_encoder_kwargs', None):                                
            self.kb_image_encoder  = get_pretrained_model(args,**kwargs['kb_image_encoder_kwargs'])      
            
        self.log_softmax = nn.LogSoftmax(1)        
        loss_fct_class = get_class(args, **kwargs['trainee_kwargs']['loss'])  
        self.loss_fct = loss_fct_class()  
    
    def init_index(self, dataset_path, column, verbose=True):
        index_data = {}
        embedding_dataset = datasets.load_from_disk( dataset_path, keep_in_memory=self.args.keep_in_memory) 
        '''
        Add Faiss Index
        ''' 
        index_column = column
        index_name = index_column + '_index'
        
        if not (os.path.exists( os.path.join( dataset_path,  index_name + '.faiss' ))):
            embedding_dataset.add_faiss_index(column=index_column, index_name=index_name, string_factory="Flat", device=None, metric_type= 0)
            embedding_dataset.save_faiss_index( index_name , os.path.join( dataset_path, index_name + '.faiss'))    
            if verbose:  
               print('\n[Index Saved]', dataset_path, '\n')     
        else:     
            embedding_dataset.load_faiss_index( index_name , os.path.join( dataset_path, index_name + '.faiss' )) 
            if verbose:
               print('\n[Index Loaded ]', dataset_path, '\n') 

        index_data = dict(embedding_dataset=embedding_dataset, index_column=index_column, index_name=index_name, 
                        dataset_path=dataset_path, column=column) 
        
        return index_data 

    def init_reader(self,):
        self.reader = get_pretrained_model(self.args,**self.kwargs['reader_kwargs'])
        self.k_train = self.kwargs['reader_kwargs']['k_train'] # nbr of retrieved documents per query
        self.k_test = self.kwargs['reader_kwargs']['k_test'] # nbr of retrieved documents per query
        if not(self.kwargs['reader_kwargs'].get('wo_kb', None)):
           self.load_index(self.kwargs['reader_kwargs'])   

    def load_index(self, answer_extractor_kwargs):       

        if answer_extractor_kwargs.get('context_embeddings', None):
            self.context_index_data = self.init_index(
                answer_extractor_kwargs['context_embeddings'], answer_extractor_kwargs['column'])
        
        if answer_extractor_kwargs.get('title_embeddings', None):
            self.title_index_data = self.init_index( 
                answer_extractor_kwargs['title_embeddings'], 'title_embedding')       

    def init_mir(self,):
        self.k_train = self.kwargs['mir_kwargs']['k_train'] # nbr of retrieved passage per query
        self.k_test = self.kwargs['mir_kwargs']['k_test'] # nbr of retrieved passage per query
        self.k_entities = self.kwargs['mir_kwargs'].get('k_entities', None) # nbr of retrieved documents per query

        if self.kwargs['mir_kwargs'].get('search_run', None) == None and  not(self.kwargs['mir_kwargs'].get('wo_kb', None)):           
           self.load_index(self.kwargs['mir_kwargs']) 

    def init_generator(self,):
        start = timeit.default_timer()
        self.answer_generator  =  get_pretrained_model(self.args,**self.kwargs['answer_generator_kwargs'])    
        stop = timeit.default_timer()
        print('\n[INFO] Generator Loading Time', (stop - start)/60 , ' mn', '\n')    

        if self.kwargs['answer_generator_kwargs']['class_name'] ==  "Blip2ForConditionalGeneration":        
            from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
            # Define LoRA Config            
            lora_config = LoraConfig(r=8,lora_alpha=32,target_modules=["q", "v"],lora_dropout=0.1,bias="none")            
            # prepare int-8 model for training
            self.answer_generator = prepare_model_for_int8_training(self.answer_generator)
            self.answer_generator.enable_input_require_grads()
            # add LoRA adaptor
            self.answer_generator = get_peft_model(self.answer_generator, lora_config)            
            self.answer_generator.print_trainable_parameters()

        self.answer_generator.config.use_cache = False   
        if self.kwargs['answer_generator_kwargs']['tokenizer_kwargs']['class_name'] == 'BlipProcessor' or self.kwargs['answer_generator_kwargs']['tokenizer_kwargs']['class_name'] == 'Blip2Processor':               
            self.processor = get_pretrained_model(self.args,**self.kwargs['answer_generator_kwargs']['tokenizer_kwargs'])
            self.tokenizer = self.processor.tokenizer   
        else:      
            self.tokenizer = get_pretrained_model(self.args,**self.kwargs['answer_generator_kwargs']['tokenizer_kwargs'])

        if self.kwargs['answer_generator_kwargs'].get('image_processor_kwargs', None):
           image_processor_class = get_class(self.args, **self.kwargs['answer_generator_kwargs']['image_processor_kwargs'])
           self.Blip2_imageFormatter = image_processor_class(self.args, self.kwargs['answer_generator_kwargs']['image_processor_kwargs'])     

        self.k_train = self.kwargs['answer_generator_kwargs']['k_train'] # nbr of retrieved documents per query
        self.k_test = self.kwargs['answer_generator_kwargs']['k_test'] # nbr of retrieved documents per query
        self.k_entities = self.kwargs['answer_generator_kwargs'].get('k_entities', None) # nbr of retrieved documents per query
        self.input_generator_max_len = self.kwargs['answer_generator_kwargs'].get('input_generator_max_len', 396)                

        if self.kwargs['answer_generator_kwargs'].get('search_run', None) == None and  not(self.kwargs['answer_generator_kwargs'].get('wo_kb', None)):           
           self.load_index(self.kwargs['answer_generator_kwargs']) 

    def get_generator_inputs(self, questions, passages, answers, tokenizer, inputs_embeds=None, answer_max_len=None):  
        answer_tokenized = self.tokenizer(answers, max_length=45 if answer_max_len==None else answer_max_len, padding="longest", 
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
                
        if inputs_embeds != None:
           question_tokenized = self.tokenizer( passages, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
           return {
            #"input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long).to(self.answer_generator.device),
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device),
            "labels": labels.to(self.answer_generator.device),
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device)
            } 
        else:
            if passages != None:
               question_tokenized = self.tokenizer(questions, passages, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation='only_second', pad_to_max_length=True, add_special_tokens=True)
            else:
                question_tokenized = self.tokenizer(questions, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)    

        gen_inputs_dict = {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long).to(self.answer_generator.device),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device),
            "labels": labels.to(self.answer_generator.device),
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device)
            }
               

        return  gen_inputs_dict  
    
    def get_generator_eval_inputs(self, questions, passages, answers, tokenizer, tokenize_answers=False, inputs_embeds=None, output_scores=False, answer_max_len=None):  
        output_dict = {}
            
        if inputs_embeds != None:
           question_tokenized = self.tokenizer( passages, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True) 
           output_dict.update( {                               
                               "inputs_embeds": inputs_embeds,
                                "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device),           
                               })
        else:
            if passages != None:
               question_tokenized = self.tokenizer(questions, passages, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation='only_second', pad_to_max_length=True, add_special_tokens=True)        
            else:
                question_tokenized = self.tokenizer(questions, max_length=self.input_generator_max_len, padding="longest",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)           
                
            output_dict.update( {
                                "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long).to(self.answer_generator.device),
                                "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device),           
                                })
        if tokenize_answers:
            answer_tokenized = self.tokenizer(answers, max_length=45 if answer_max_len==None else answer_max_len, padding="longest", 
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)        
            labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
            labels[labels == 0] = -100
            answer_dict = { "labels": labels.to(self.answer_generator.device),
                            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long).to(self.answer_generator.device)
                          }
            output_dict.update( answer_dict)
        
        if output_scores:
           output_dict.update( dict(output_scores=True)) 
           output_dict.update( dict(return_dict_in_generate=True)) 
        if self.reader_kwargs.get('beam_search', None):          
           output_dict.update( dict(num_beams=5)) 
           output_dict.update( dict(max_length=10)) 
        

        return output_dict      

    def get_retrieval_labels(self, questions, retrieved_passages, original_answers, alternative_answers ):
         new_answers = []
         # retrieval_labels = [-100] * len(questions)        
         retrieval_labels = []        
         total_passage_idx = 0
         retrieval_indices = [ [] for _ in range(len(questions)) ]  

         for question_idx, question in enumerate(questions):                      
            possible_passages = retrieved_passages[question_idx]
            current_batch_labels = [0] * len(possible_passages)
            for psg_idx, passage in enumerate(possible_passages):              
                new_answer = None                
                passage = answer_preprocess(passage)
                answer = answer_preprocess(original_answers[question_idx])                
                
                if re.search(rf'\b{answer}\b', passage) is not None:                      
                    #   retrieval_labels[question_idx] = total_passage_idx
                      current_batch_labels[psg_idx] = 1

                      new_answer = original_answers[question_idx]
                retrieval_indices[question_idx].append(total_passage_idx)                
                
                for alt_answer in alternative_answers[question_idx]:
                        answer = answer_preprocess(alt_answer)
                        if re.search(rf'\b{answer}\b', passage) is not None:
                           current_batch_labels[psg_idx] = 1
                           break
                total_passage_idx += 1 
            retrieval_labels.append(current_batch_labels) 
             
         retrieval_indices = torch.tensor(retrieval_indices)
         
         retrieval_labels = torch.tensor(retrieval_labels)
         return retrieval_labels

    def prepare_rag_inputs(self, questions, retrieved_passages, original_answers, alternative_answers, relevant_passages=None, supervised=False):
        new_questions = []
        new_passages = []
        new_answers = []
        # retrieval_labels = [-100] * len(questions)        
        retrieval_labels = []
        
        total_passage_idx = 0
        retrieval_indices = [ [] for _ in range(len(questions)) ]          
        best_retrieval_indices = [ -100 for _ in range(len(questions)) ] 

        for question_idx, question in enumerate(questions):            
            if supervised:
               possible_passages = retrieved_passages 
            else: 
                possible_passages = retrieved_passages[question_idx]

            current_batch_labels = [0] * len(possible_passages)
            for psg_idx, passage in enumerate(possible_passages):                                              
                new_questions.append(question) 
                new_passages.append(passage)                         
                new_answer = None
                
                passage = answer_preprocess(passage)
                answer = answer_preprocess(original_answers[question_idx])                

                # if retrieval_labels[question_idx] == -100: # consider retrieval label with highest score as relevant               
                if re.search(rf'\b{answer}\b', passage) is not None:                                          
                      current_batch_labels[psg_idx] = 1
                      new_answer = original_answers[question_idx]
                      new_answers.append(new_answer)
                      if best_retrieval_indices[question_idx] == -100:
                         best_retrieval_indices[question_idx] = total_passage_idx                           

                retrieval_indices[question_idx].append(total_passage_idx)
                # if retrieval_labels[question_idx] == -100: # consider retrieval label with highest score as relevant
                for alt_answer in alternative_answers[question_idx]:
                        answer = answer_preprocess(alt_answer)
                        if re.search(rf'\b{answer}\b', passage) is not None:                       
                        #    retrieval_labels[question_idx] = total_passage_idx
                           current_batch_labels[psg_idx] = 1
                           if new_answer ==  None:
                              new_answer = alt_answer
                              new_answers.append(new_answer)

                           if best_retrieval_indices[question_idx] == -100:
                              best_retrieval_indices[question_idx] = total_passage_idx
                           break

                total_passage_idx += 1 
                # if no answer consider the original
                if new_answer ==  None:
                   new_answers.append(original_answers[question_idx]) 

            retrieval_labels.append(current_batch_labels) 
             
        retrieval_indices = torch.tensor(retrieval_indices)
        best_retrieval_indices = torch.tensor(best_retrieval_indices)
        retrieval_labels = torch.tensor(retrieval_labels)        
        return  new_questions, new_passages, new_answers , retrieval_labels, retrieval_indices, best_retrieval_indices           

    def get_retrieved_passages(self, index_data, kb, question_outputs, return_vectors=False, return_batch=False, K=None, kb_column='passage', article2passage=None,
                                return_all_items=False):
        '''
        search closest vectors in KB to queries
        return_batch: return retrived passages arranged by batch (batch_size x K) if not(default): return for each query the string concatenation of the k best passages
        return_vectors: return the vector of retrieved passages
        '''
        def apply_mapping(x):
            passage_indices = article2passage.get(str(x), None)
            if passage_indices:
               passage_index = passage_indices[0] # NOTE Here we take the first passage from article. likely not optimal
            else:
                passage_index = -1
            return passage_index  
        
        output_dict = {}
        if K:
           k_value = K
        else:      
           k_value = self.k_train if self.training else self.k_test   

        indexed_embeddings = index_data['embedding_dataset']
        scores, indices = indexed_embeddings.search_batch( index_data['index_name'], question_outputs.detach().cpu().numpy(), k=k_value)          
        flattened_indices = indices.flatten()
        if article2passage:
            # Convert artice indices to passage indices. Here we take the first passage from article. likely not optimal            
            indices = torch.tensor(indices)
            indices = indices.apply_(apply_mapping)
            flattened_indices = indices.flatten()            
            mask = flattened_indices != -1
            flattened_indices = flattened_indices[mask]            
        
        retrieved_passages_vectors = indexed_embeddings[flattened_indices]        
        retrieved_passages = kb.select(flattened_indices)[kb_column]         
        output_dict.update(dict(scores=scores, indices=indices))
        
        if  return_batch: 
            # arrange by batch
            batched_retrieved_passages  = self.unflatten_inputs(retrieved_passages, k_value)              
            output_dict.update(dict(retrieved_items=batched_retrieved_passages))
            if return_vectors:
               # vectors = torch.tensor([self.context_embeddings[self.index_column ][i] for i in indices.flatten() ])            
               #NOTE: investigate a better way to avoid loading two indexes
            #    vectors = torch.tensor( index_data['embedding_dataset_wo_index'].select(indices.flatten())[index_data['index_column']] )
            #    vectors = torch.tensor( index_data['embedding_dataset'][indices.flatten()][index_data['index_column']] )        
               vectors = retrieved_passages_vectors.pop(index_data['index_column'])
               vectors = torch.tensor(vectors)
               output_dict.update(dict(vectors=vectors))
        else:           
            # Concatenate every k strings        
            concatenated_passages = []              
            for i in range(0, len(retrieved_passages), k_value):
                concatenated_string = ' '.join(retrieved_passages[i:i+k_value])
                concatenated_passages.append(concatenated_string)   
            output_dict.update(dict(retrieved_items=concatenated_passages))    

        if return_all_items:        
           output_dict.update(retrieved_passages_vectors)
        return  output_dict
    
    def augment_questions(self, questions, prompts, entity_types=None):
        augmented_questions = []
        for i, q in enumerate(questions):
            if type(prompts[i])==list:
                for p in prompts[i]:
                    augmented_questions.append( ' '.join([q , p]) )
            else:               
                if entity_types:
                   augmented_questions.append( ' '.join([q , prompts[i], entity_types[i]]) )
                else:                    
                    augmented_questions.append( ' '.join([q , prompts[i]]) )
        return augmented_questions       
    
    
    def augment_questions_for_Generator(self, questions, prompts, entity_types=None):
        augmented_questions = []
        for i, q in enumerate(questions):              
            if entity_types:
               augmented_questions.append( ' '.join([q , 'A picture of Entity', prompts[i], 'with Type', entity_types[i]]) )
            else:                
                augmented_questions.append( ' '.join([q , 'A picture of Entity', prompts[i]]) )

        return augmented_questions    
    
    def diff_fill_mask(self, input_tensor, mask, value):
        # Convert the mask to a float tensor for element-wise multiplication
        mask = mask.float()    
        # Multiply the tensor by the inverted mask to keep original values where mask is False,
        # and multiply by the value where the mask is True.
        filled_tensor = input_tensor * (1 - mask) + value * mask
        filled_tensor = filled_tensor.long()
        return filled_tensor
    '''
    Adapted from: https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering/blob/main/src/models/rag/rag_model_blip.py
    '''
    def compute_ll_loss(self, seq_logprobs, target, ignore_index=-100, K=None ):
        if K:
           k_value = K
        else:
            k_value = self.k_train if self.training else self.k_test   

        n_docs = k_value
        batch_size = seq_logprobs.shape[0] // n_docs      
        
        seq_logprobs = nn.functional.log_softmax(seq_logprobs, dim=-1).view( batch_size, n_docs, -1, seq_logprobs.size(-1))  # batch_size x n_docs x tgt_len x vocab_size
        # seq_logprobs = seq_logprobs.view( batch_size, n_docs, -1, seq_logprobs.size(-1))  # batch_size x n_docs x tgt_len x vocab_size
        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()        
        pad_mask = new_target.eq(ignore_index)
        # if pad_mask.any() and ignore_index < 0:
        #     # fill -100 to be 0, avoid indexing error using gather
        #     new_target.masked_fill_(pad_mask, 0)        
        new_target = self.diff_fill_mask(new_target, pad_mask, 0)
        ll = seq_logprobs.gather(dim=-1, index=new_target)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        
        ll = ll.squeeze(-1) # batch_size x n_docs x seq_len
        ll = ll.sum(2) # batch_size x n_docs
        # ll = -ll
        return ll

    def rag_loss(self, seq_logprobs, doc_scores, target, retrieval_labels=None, ignore_index=-100, K_value=None, entity_scores=None ):
        '''
        args:
            Adapted from: https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering/blob/main/src/models/rag/rag_model_blip.py
            seq_logprobs: log logits from generator  batch_size * n_docs x tgt_len x vocab_size
            doc_logprobs: log logits from retriever  batch_size x n_docs 
            target: tokenized answers
            tgt_len=answer_len
        '''      
        doc_logprobs = self.log_softmax(doc_scores) 
        if entity_scores != None:
           entity_logprobs = self.log_softmax(entity_scores) 
           doc_logprobs = doc_logprobs + entity_logprobs           

        doc_logprobs = doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        
        if K_value:
           k = K_value
        else:
            k = self.k_train if self.training else self.k_test   

        n_docs = k
        batch_size = seq_logprobs.shape[0] // n_docs              
        
        seq_logprobs = nn.functional.log_softmax(seq_logprobs, dim=-1).view( batch_size, n_docs, -1, seq_logprobs.size(-1))  # batch_size x n_docs x tgt_len x vocab_size
        # bos_token_id is None for T5
        bos_token_id = self.answer_generator.config.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()
        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        # print('first_token_scores', first_token_scores.shape)    
        if use_bos:
            second_token_scores = seq_logprobs[:, :, 1:2, :]
            remainder = seq_logprobs[:, :, 2:, :]
            rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        else:
            # print('using T5 doc probs!')
            remainder = seq_logprobs[:, :, 1:, :]
            rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2)     

        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()
        
        pad_mask = new_target.eq(ignore_index)
        if pad_mask.any() and ignore_index < 0:
            # fill -100 to be 0, avoid indexing error using gather
            # new_target.masked_fill_(pad_mask, 0)  # in-place cause error: gradient computation has been modified by an inplace operation                          
            new_target = new_target.masked_fill(pad_mask, 0)        

        # Compute RAG loss
        # reuse new_target
        rag_ll = rag_logprobs.gather(dim=-1, index=new_target)
        # rag_ll = self.diff_fill_mask(rag_ll, pad_mask, 0.0)
        # reuse pad_mask
        if pad_mask.any():
        #    rag_ll.masked_fill_(pad_mask, 0.0)    
           rag_ll = rag_ll.masked_fill(pad_mask, 0.0)    
        
        rag_ll = rag_ll.squeeze(-1) # batch_size x n_docs x seq_len
        # reduce directly since we don't use it elsewhere
        # in training, the marginalisation is performed
        rag_ll = rag_ll.sum(2)  # sum over tokens        
        rag_ll = rag_ll.logsumexp(1)  # sum over docs        
        rag_ll = rag_ll.sum() # sum over batches        
        rag_loss = -rag_ll
        if self.args.debug:
           print('rag_loss', rag_loss) 
        return rag_loss 
    
    def flatten_inputs(self, questions, retrieved_passages, answers):
        new_questions = []
        new_passages = []
        new_answers = []
        
        for question_idx, question in enumerate(questions):            
            possible_passages = retrieved_passages[question_idx]            
            for psg_idx, passage in enumerate(possible_passages):
                new_questions.append(question) 
                new_passages.append(passage)
                if answers != None:        
                   new_answers.append(answers[question_idx])

        return new_questions, new_passages, new_answers   
    
    def unflatten_inputs(self, input_list, K):
        batched_input = []
        for i in range(0, len(input_list), K):
            batch_inp = []                
            for e in input_list[i:i+K]:
                batch_inp.append(e)
            batched_input.append(batch_inp)
        return  batched_input   

    def prompt_augmentation(self, questions, question_outputs,  original_answers, entity_answers, kb, entity_predictions=None, entity_types=None, quesion_images_pixels=None, all_question_items=None ):        
        if entity_predictions == None:
           if self.reader_kwargs.get('best_entity', None):
              retrieved_passages = self.get_retrieved_passages( self.context_index_data, kb,
                                                                question_outputs, K=3, return_batch=True)['retrieved_items']        
              entity_predictions = self.get_best_answers( self.entity_generator, questions, retrieved_passages, K=3)
           else:
               retrieved_passages = self.get_retrieved_passages(self.context_index_data, kb,
                                                                 question_outputs, K=3)['retrieved_items']        
               entity_predictions = self.get_concat_answers( self.entity_generator, questions, retrieved_passages) 

        # add entity-based prompt to questions
        augmented_questions = self.augment_questions( questions, entity_predictions, entity_types=entity_types )
        # augmented_questions_for_Generator = self.augment_questions_for_Generator( questions, entity_predictions, entity_types=entity_types )
        # tokenize  prompted questions
        augmented_questions_tokenized = self.dpr_tokenizer( augmented_questions, max_length=256, padding="longest",
                                        truncation=True, pad_to_max_length=True, add_special_tokens=True)        
        augmented_questions_tokenized = {k:torch.tensor(v,dtype=torch.long).to(self.question_model.device) for k, v in augmented_questions_tokenized.items()}    
        # Compute  prompted questions representations.
        augmented_question_outputs = self.question_model(**augmented_questions_tokenized).pooler_output

        if self.args.debug:
           print('questions',questions, len(questions)) 
           print('augmented_questions', augmented_questions, len(augmented_questions)) 
           print('entity_predictions',entity_predictions, len(entity_predictions)) 
           print('entity reference',entity_answers, len(entity_answers)) 
           print('\n')                           

        if self.reader_kwargs.get('best_answer', None):
           # # Retrieve new passages using prompted questions
           augmented_retrieved_passages = self.get_retrieved_passages( self.context_index_data, kb, 
                                                                      augmented_question_outputs, K=self.k_test, return_batch=True)['retrieved_items']      
           answer_predictions = self.get_best_answers( self.answer_generator, questions, augmented_retrieved_passages, K=self.k_test, quesion_images_pixels=quesion_images_pixels)           
        else:
            # Retrieve new passages using prompted questions
            augmented_retrieved_passages = self.get_retrieved_passages( self.context_index_data, kb,
                                                                        augmented_question_outputs, return_batch=False)['retrieved_items'] 
            # NOTE: Here we can either generate answers using prompted questions with retrieved entities or using the original questions
            # NOTE: preliminary results show that using orginal questions may perform slightly better.
    
            # answer_predictions = self.get_concat_answers( self.answer_generator, augmented_questions_for_Generator, augmented_retrieved_passages)
            answer_predictions = self.get_concat_answers( self.answer_generator, questions, augmented_retrieved_passages, quesion_images_pixels=quesion_images_pixels)
        
        self.write_analysis_file(questions=questions, original_answers=original_answers, entity_predictions=entity_predictions, entity_answers=entity_answers
                            , answer_predictions=answer_predictions, retrieved_passages=augmented_retrieved_passages, all_question_items=all_question_items )
        return answer_predictions
    
    def get_concat_answers(self, generator, questions, retrieved_passages, quesion_images_pixels=None):
        gen_inputs = self.get_generator_eval_inputs( questions, retrieved_passages, None, self.tokenizer, output_scores=False) 
        if quesion_images_pixels:
           gen_inputs.update(quesion_images_pixels) 

        generated_entity_ids = generator.generate(**gen_inputs)
        predictions = [ self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in generated_entity_ids] 
        return predictions

    def get_best_answers(self, generator, questions, retrieved_passages, K=None, quesion_images_pixels=None):
        flattened_questions, flattened_passages, flattened_answers = self.flatten_inputs( questions, retrieved_passages, None) 
        gen_inputs = self.get_generator_eval_inputs( flattened_questions, flattened_passages, flattened_answers, self.tokenizer, output_scores=True)
        if quesion_images_pixels:
           gen_inputs.update(quesion_images_pixels) 

        # generate all answers
        gen_output = generator.generate(**gen_inputs)
        all_generated_answer_ids = gen_output.sequences  

        if self.reader_kwargs.get('beam_search', None):
            sequences_scores = gen_output.sequences_scores
            batch_size = sequences_scores.shape[0] // K 
            sequences_scores = sequences_scores.reshape(batch_size, K)
            generation_outputs_decoded = self.tokenizer.batch_decode(all_generated_answer_ids, skip_special_tokens=True)            
            all_generated_answer_ids = all_generated_answer_ids.reshape(batch_size, K, -1)
            
            outputs = [] 
            best_predictions = []
            for b in range(batch_size):
                # use topk to get indices of top candidates
                top_cand_inds = (sequences_scores[b]).topk(1)[1]                
                outputs.append(all_generated_answer_ids[b, top_cand_inds])
                answer_proposals = generation_outputs_decoded[b*K:(b+1)*K][top_cand_inds]
                best_predictions.append(answer_proposals)                
            outputs = torch.cat(outputs)            
            return best_predictions
            
        else:
            # get ansewer predictions given all retrieved passages
            all_answer_predictions = [ self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in all_generated_answer_ids] 
            
            seq_scores = torch.stack(gen_output.scores)
            seq_scores = torch.swapaxes(seq_scores, 0, 1)
            targets = all_generated_answer_ids[:,1:] # slice and ignore bos token 
            ll = self.compute_ll_loss( seq_scores, targets, ignore_index=-100, K=K )
            doc_scores, doc_indices =  torch.max(ll, dim=-1) 
            
            all_answer_predictions = self.unflatten_inputs( all_answer_predictions, K)  #(batch x n_doc)         
            # select answers with max scores
            best_predictions = []
            for j, pred in enumerate(all_answer_predictions):                   
                best_predictions.append(pred[doc_indices[j]])                         
            
            return best_predictions       

    
    def log_probs_to_answers(self, start_log_probs, end_log_probs, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        passage_indices, start_indices, end_indices = get_best_spans(
            start_probs=start_log_probs.exp(),
            end_probs=end_log_probs.exp(),
            **kwargs
        )
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.bert_tokenizer.batch_decode(answers, skip_special_tokens=True)

    def forward(self, inputs, only_queries=False ,return_dict=None):
        output_dict = {}

        if not(self.reader_kwargs.get('generator_only', None)):
           question_outputs = self.question_model(**inputs['question_inputs']) 

           if self.kwargs['trainee_kwargs'].get('encode_image', None):
              if 'question_image_inputs' in inputs:   
                  question_image_outputs, image_hidden_states = self.image_encoder(**inputs['question_image_inputs'])                  
                  output_dict.update(dict(question_image_outputs=question_image_outputs))                     
                  if self.kwargs.get('kb_image_encoder_kwargs', None):
                     kb_question_image_outputs, image_hidden_states = self.kb_image_encoder(**inputs['question_image_inputs'])                  
                     output_dict.update(dict(kb_question_image_outputs=kb_question_image_outputs))                     

           question_outputs = question_outputs if torch.is_tensor(question_outputs) else  question_outputs.pooler_output
           output_dict.update(question_outputs=question_outputs)

        return output_dict
    

    def write_analysis_file(self, questions, original_answers=None, entity_predictions=None, entity_answers=None, answer_predictions=None,
                            retrieved_passages=None, all_question_items=None ):
        
        self.qualitative_analysis = open(os.path.join(self.args.experiment_dir, 'qualitative_analysis.txt'), 'a')
        
        for i in range(len(questions)):
            question_index = self.batch_idx * len(questions) + i
            self.qualitative_analysis.write(f"\n[Question Index] {question_index}\n")
            self.qualitative_analysis.write("===================================================\n")
            self.qualitative_analysis.write(f"[Question] \n{questions[i]}\n\n")
            
            if entity_predictions:
                self.qualitative_analysis.write("[Predicted Entity]\n--------------------\n")
                self.qualitative_analysis.write(f"{entity_predictions[i]}\n\n")
            
            if entity_answers:
                self.qualitative_analysis.write("[Reference Entity]\n-------------------\n")
                self.qualitative_analysis.write(f"{entity_answers[i]}\n\n")
            
            if answer_predictions:
                self.qualitative_analysis.write("[Predicted Answer]\n-------------------\n")
                self.qualitative_analysis.write(f"{answer_predictions[i]}\n\n")
            
            if original_answers:
                self.qualitative_analysis.write("[Original Answer]\n------------------\n")
                self.qualitative_analysis.write(f"{original_answers[i]}\n\n")
            
            if retrieved_passages:
                self.qualitative_analysis.write("[Retrieved Passages]\n---------------------\n")                
                self.qualitative_analysis.write(f"{retrieved_passages[i]}\n\n")
            
            if all_question_items:
                q_items = all_question_items[i]
                if 'image' in q_items:
                    self.qualitative_analysis.write("[Image]\n---------\n")
                    self.qualitative_analysis.write(f"{q_items['image']}\n\n")
                if 'url' in q_items:
                    self.qualitative_analysis.write("[Image URL]\n------------\n")
                    self.qualitative_analysis.write(f"{q_items['url']}\n\n")
            
            # End of current question block
            self.qualitative_analysis.write("***************************************************\n\n")

        self.qualitative_analysis.close()
        if self.args.debug:
           exit()

    def get_label_vector(self, retrieved, labels):
        label_vector = []
        # Flatten the retrieved titles and keep track of the original indices
        flattened_titles = []
        for batch_idx, sublist in enumerate(retrieved):            
            for i, title in enumerate(sublist):
                if title == labels[batch_idx]:
                   label_vector.append(1)
                else:
                    label_vector.append(0)                   

        return label_vector    
           

    ############################   Score-based Fusion RAG  ################################
    def sbf_rag(self, question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs):
        if self.training: 
           '''
           Training
           '''   
           if self.train_search_run:                                
                passage_indices = []
                for qid in qids:
                   indices = list(map(int, self.train_search_run[qid].keys()))[: self.k_train]
                   scores = list(self.train_search_run[qid].values())[: self.k_train]   
                   passage_indices.extend(indices)         
                #    
                retrieved_passages = kb.select(passage_indices)['passage']
                # batchify retrieved passages
                batched_retrieved_passages = []
                for i in range(0, len(retrieved_passages), self.k_train):
                    b_passages = []                
                    for p in retrieved_passages[i:i+self.k_train]:
                        b_passages.append(p)
                    batched_retrieved_passages.append(b_passages) 
                retrieved_passages = batched_retrieved_passages           

                new_questions, new_retrieved_passages, new_answers, retrieval_labels, retrieval_indices, best_retrieval_indices = self.prepare_rag_inputs( questions, 
                                                            retrieved_passages, original_answers, answers, relevant_passages=relevant_passages)
                t5_inputs = self.get_generator_inputs( new_questions, new_retrieved_passages, new_answers, self.tokenizer)
                if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":
                    #  question_pixel_values = inputs['question_image_inputs']['pixel_values']
                    question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                    question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                    question_pixel_values = torch.stack([tens for tens in question_pixel_values for _ in range(self.k_train)])
                    question_pixel_values = dict( pixel_values=question_pixel_values )  
                    t5_inputs.update(question_pixel_values)

                output = self.answer_generator(**t5_inputs)
                gen_loss = output.loss
                gen_logits = output.logits  # (batch_size, sequence_length, config.vocab_size) 
                return  dict(loss=gen_loss)
        else:   
            if self.search_run:                                
                passage_indices = []
                for qid in qids:
                   indices = list(map(int, self.search_run[qid].keys()))[: self.k_test]
                   scores = list(self.search_run[qid].values())[: self.k_test]   
                   passage_indices.extend(indices)         
                #    
                retrieved_passages = kb.select(passage_indices)['passage']
                # Concatenate every k strings        
                concatenated_passages = []              
                for i in range(0, len(retrieved_passages), self.k_test):
                    concatenated_string = ' '.join(retrieved_passages[i:i+self.k_test])
                    concatenated_passages.append(concatenated_string) 

                if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":                
                    question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                    question_pixel_values = question_image_batch['pixel_values'].to(self.answer_generator.device)
                    quesion_images_pixels = dict( pixel_values=question_pixel_values ) 
                    predictions = self.get_concat_answers( self.answer_generator, questions, concatenated_passages, quesion_images_pixels=quesion_images_pixels)                     
                else:
                    predictions = self.get_concat_answers( self.answer_generator, questions, concatenated_passages) 

                val_loss = torch.tensor([0.0],device=self.answer_generator.device)
                
                return dict(predictions=predictions, answers=answers, original_answers=original_answers, questions=questions, loss=val_loss, log_probs=None)   

    ############################  RAG  ################################
    def rag(self, question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs):
        
        if self.training: 
           '''
           Training
           '''  
           if self.reader_kwargs.get('rag_training', None):
              
              detached_question_outputs = question_outputs.clone().cpu().detach() 
              if self.reader_kwargs.get('multimodal', None):                
                # project to lower dimension space
                question_outputs = self.mlp_reducer_model(question_outputs)
                question_image_outputs = self.mlp_reducer_model(question_image_outputs)
                # question_outputs =  self.LayerNorm( question_outputs + question_image_outputs)
                question_outputs =   question_outputs + question_image_outputs
                detached_question_outputs = question_outputs.clone().cpu().detach()

              retrieval_output_dict = self.get_retrieved_passages( self.context_index_data, kb, detached_question_outputs, return_vectors=True, return_batch=True)
              retrieved_passages, context_embeddings = retrieval_output_dict['retrieved_items'], retrieval_output_dict['vectors'] 
              new_questions, new_retrieved_passages, new_answers, retrieval_labels, retrieval_indices, best_retrieval_indices = self.prepare_rag_inputs( questions, 
                                                            retrieved_passages, original_answers, answers, relevant_passages=relevant_passages)
              '''
              ------------------------------
              RAG
              ------------------------------
              '''
              context_embeddings = context_embeddings.to(question_outputs.device) 
              doc_scores = question_outputs @ context_embeddings.T 

              dpr_logits = self.log_softmax(doc_scores)            
              dpr_labels  =  torch.range(0,question_outputs.shape[0] - 1,device=question_outputs.device, dtype=int) # Contrastive Labels    
              doc_scores_for_rag = torch.gather(doc_scores, 1, retrieval_indices.to(question_outputs.device))   
              
              t5_inputs = self.get_generator_inputs( new_questions, new_retrieved_passages, new_answers, self.tokenizer) 
              if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":                
                 question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                 question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                 question_pixel_values = torch.stack([tens for tens in question_pixel_values for _ in range(self.k_train)])                 
                 question_pixel_values = dict( pixel_values=question_pixel_values )  
                 t5_inputs.update(question_pixel_values)  

              if self.args.debug:
                 for k in  t5_inputs:
                     if torch.is_tensor(t5_inputs[k]):
                        print(k, t5_inputs[k].shape, t5_inputs[k].dtype)

              output = self.answer_generator(**t5_inputs)
              gen_loss = output.loss
              gen_logits = output.logits  # (batch_size, sequence_length, config.vocab_size) 
              
              rag_loss = self.rag_loss(gen_logits, doc_scores_for_rag, t5_inputs['labels'], retrieval_labels=retrieval_labels )
              if self.args.debug:
                 print('gen_loss', gen_loss) 
                  
              total_loss = rag_loss + gen_loss
              return  dict(loss=total_loss)  
           
        else:    
                    if self.reader_kwargs.get('best_answer', None):
                        retrieval_output_dict = self.get_retrieved_passages( self.context_index_data, kb, question_outputs, K=self.k_test, return_batch=True)
                        retrieved_passages  = retrieval_output_dict['retrieved_items']
                        if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":                        
                           question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                           question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                           quesion_images_pixels = dict( pixel_values=question_pixel_values )    
                           predictions = self.get_best_answers( self.answer_generator, questions, retrieved_passages, K=self.k_test, quesion_images_pixels=quesion_images_pixels)
                        else:
                            predictions = self.get_best_answers( self.answer_generator, questions, retrieved_passages, K=self.k_test)
                    else:
                        if self.reader_kwargs.get('wo_kb', None): # do not retrieve passages
                           retrieved_passages = None
                        else:
                            retrieval_output_dict = self.get_retrieved_passages( self.context_index_data, kb, question_outputs, K=self.k_test)        
                            retrieved_passages  = retrieval_output_dict['retrieved_items']     

                        if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":                        
                           question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                           question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                           quesion_images_pixels = dict( pixel_values=question_pixel_values ) 
                           predictions = self.get_concat_answers( self.answer_generator, questions, retrieved_passages, quesion_images_pixels=quesion_images_pixels) 
                        else:
                            predictions = self.get_concat_answers( self.answer_generator, questions, retrieved_passages) 

                
                    loss = torch.tensor([0.0],device=self.answer_generator.device)
                    self.write_analysis_file(questions=questions, original_answers=original_answers, entity_predictions=None, entity_answers=None
                            , answer_predictions=predictions, retrieved_passages=retrieved_passages, all_question_items=all_question_items )
                    self.batch_idx += 1
                    return dict(predictions=predictions, answers=answers, original_answers=original_answers, questions=questions, loss=loss)

    ############################  Generator Only  ################################
    def Generate(self, question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs):        
        if self.training: 
           '''
           Training
           '''
           if self.reader_kwargs.get('generator_only', None):
               if self.reader_kwargs.get('entity_training', None):                    
                  gen_inputs = self.get_generator_inputs( questions, relevant_passages, question_titles , self.tokenizer)                                    
               else:                                       
                   if self.reader_kwargs.get('wo_kb', None): # do not retrieve passages
                       relevant_passages = None
                   gen_inputs = self.get_generator_inputs( questions, relevant_passages, original_answers, self.tokenizer)
                   if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":
                      question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                      question_pixel_values = question_image_batch['pixel_values'].to(self.answer_generator.device)
                      quesion_images_pixels = dict( pixel_values=question_pixel_values )                                            
                      gen_inputs.update(quesion_images_pixels)  
               output = self.answer_generator(**gen_inputs)
               gen_loss = output.loss                              
               gen_logits = output.logits  # (batch_size, sequence_length, config.vocab_size)
               return  dict(loss=gen_loss)
        else:  
                if self.reader_kwargs.get('entity_training', None):             
                   gen_inputs = self.get_generator_eval_inputs( questions, relevant_passages, question_titles, self.tokenizer)                            
                else:
                    if self.reader_kwargs.get('wo_kb', None): # do not retrieve passages
                       relevant_passages = None    
                    gen_inputs = self.get_generator_eval_inputs( questions, relevant_passages, original_answers, self.tokenizer)
                    if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":
                      question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                      question_pixel_values = question_image_batch['pixel_values'].to(self.answer_generator.device)
                      quesion_images_pixels = dict( pixel_values=question_pixel_values )                     
                      gen_inputs.update(quesion_images_pixels)            

                generated_ids = self.answer_generator.generate(**gen_inputs)
                predictions = [ self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in generated_ids]
                loss = torch.tensor([0.0],device=self.answer_generator.device)
                if self.reader_kwargs.get('entity_training', None):
                    return  dict(loss=loss, predictions=predictions, answers=None, original_answers=question_titles, questions=questions)
                else:
                    return  dict(loss=loss, predictions=predictions, answers=answers, original_answers=original_answers, questions=questions)       

    ############################  MiRAG  ################################
    def MiRAG(self, question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs):

        if self.training: 
           '''
           Training
           '''
           if self.reader_kwargs.get('entity_prompt', None):
              if self.k_entities:
                 k_entities = self.k_entities 
              else:
                print('please set k_entities in config file')
                exit()

              if self.reader_kwargs.get('title_embeddings', None):   
                 CM_retrieved_dict = self.get_retrieved_passages(self.title_index_data, entity_kb, question_image_outputs, K=k_entities, kb_column=self.reader_kwargs['title_key'],
                                                           return_vectors=True, return_batch=True)
                 CM_retrieved_titles, CM_entity_embeddings = CM_retrieved_dict['retrieved_items'], CM_retrieved_dict['vectors']
              
              if  self.reader_kwargs.get('title_embeddings', None): 
                  retrieved_titles, entity_embeddings = CM_retrieved_titles, CM_entity_embeddings

              entity_scores = None
              if self.reader_kwargs.get('entity_grad', None):
                 entity_embeddings = entity_embeddings.to(question_image_outputs.device) 
                 entity_scores = question_image_outputs @ entity_embeddings.T  # (batch , batch x k_entities)                            
                 entity_retrieval_indices = torch.tensor( [[ (i*k_entities)+j for j in range(k_entities)] for i in range(entity_scores.shape[0])] )                      
                 entity_scores = torch.gather(entity_scores, 1, entity_retrieval_indices.to(question_outputs.device))                             
                 entity_scores = entity_scores.flatten().unsqueeze(1)                            
                 entity_scores = entity_scores.repeat(1, self.k_train)
                 entity_scores = entity_scores.reshape(question_image_outputs.shape[0], -1)  # (batch , batch x k_train)                                          

              repeated_original_answers = [ans for ans in original_answers for _ in range(k_entities)] # Repeate M times each element of list
              repeated_answers = [ans for ans in answers for _ in range(k_entities)] # Repeate M times each element of list
              repeated_questions = [q for q in questions for _ in range(k_entities)] # Repeate M times each element of list

              # add entity-based prompt to questions
              augmented_questions = self.augment_questions( questions, retrieved_titles, entity_types=None )  
              # tokenize  prompted questions
              augmented_questions_tokenized = self.dpr_tokenizer( augmented_questions, max_length=256, padding="longest",
                                                truncation=True, pad_to_max_length=True, add_special_tokens=True)        
              augmented_questions_tokenized = {k:torch.tensor(v,dtype=torch.long).to(self.question_model.device) for k, v in augmented_questions_tokenized.items()}    
              # Compute  prompted questions representations.
              augmented_question_outputs = self.question_model(**augmented_questions_tokenized).pooler_output              

              retrieval_output_dict = self.get_retrieved_passages( self.context_index_data, kb, augmented_question_outputs, return_vectors=True, return_batch=True)
              retrieved_passages, context_embeddings = retrieval_output_dict['retrieved_items'], retrieval_output_dict['vectors']   
              
              new_questions, new_retrieved_passages, new_answers, retrieval_labels, retrieval_indices, best_retrieval_indices = self.prepare_rag_inputs( repeated_questions, 
                                                            retrieved_passages, repeated_original_answers, repeated_answers, relevant_passages=relevant_passages)              
              '''
              ------------------------------
              RAG
              ------------------------------
              '''
              if not( self.reader_kwargs.get("cross_entropy",None)):
                context_embeddings = context_embeddings.to(question_outputs.device) 
                doc_scores = question_outputs @ context_embeddings.T
                retrieval_indices = torch.tensor( [[ (i*k_entities*self.k_train)+j for j in range(k_entities*self.k_train)] for i in range(question_outputs.shape[0])] )                      
                dpr_logits = self.log_softmax(doc_scores)            
                dpr_labels  =  torch.range(0,question_outputs.shape[0] - 1,device=question_outputs.device, dtype=int) # Contrastive Labels                  
                doc_scores_for_rag = torch.gather(doc_scores, 1, retrieval_indices.to(question_outputs.device))               

              gen_inputs = self.get_generator_inputs( new_questions, new_retrieved_passages, new_answers, self.tokenizer)   
              if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":
                #  question_pixel_values = inputs['question_image_inputs']['pixel_values']
                 question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                 question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                 question_pixel_values = torch.stack([tens for tens in question_pixel_values for _ in range(k_entities*self.k_train)])
                 question_pixel_values = dict( pixel_values=question_pixel_values )  
                 gen_inputs.update(question_pixel_values)

              output = self.answer_generator(**gen_inputs)
              gen_loss = output.loss
              gen_logits = output.logits  # (batch_size, sequence_length, config.vocab_size) 
              if self.args.debug:
                 print('gen_loss', gen_loss) 
              
              if not( self.reader_kwargs.get("cross_entropy",None)):
                 rag_loss = self.rag_loss(gen_logits, doc_scores_for_rag, gen_inputs['labels'], retrieval_labels=retrieval_labels, K_value=self.k_train*k_entities, entity_scores=entity_scores)
                 total_loss =  rag_loss + gen_loss
              else:
                  total_loss = gen_loss            
              return  dict(loss=total_loss)    
        
        else:
                #NOTE For inference: we retrieve only 1 entity before passage retrieval
                if self.reader_kwargs.get('title_embeddings', None):                    
                    inference_nbr_entities = 1
                    CM_retrieved_dict = self.get_retrieved_passages(self.title_index_data, entity_kb,
                                    question_image_outputs, K=inference_nbr_entities, kb_column=self.reader_kwargs['title_key'], return_batch=False)                                                                    
                    retrieved_titles  = CM_retrieved_dict['retrieved_items']
                            
                if self.reader_kwargs['class_name'] ==  "Blip2ForConditionalGeneration":                
                    question_image_batch, question_images = self.Blip2_imageFormatter.format_pixels(all_question_items, invalid_indices=[]) 
                    question_pixel_values = question_image_batch['pixel_values'].to(question_outputs.device)
                    quesion_images_pixels = dict( pixel_values=question_pixel_values ) 
            
                    predictions = self.prompt_augmentation( questions, question_outputs,  original_answers, question_titles, kb,
                                entity_predictions=retrieved_titles, entity_types=None, quesion_images_pixels=quesion_images_pixels, all_question_items=all_question_items ) 
                else:  
                    predictions = self.prompt_augmentation( questions, question_outputs,  original_answers, question_titles, kb,
                                entity_predictions=retrieved_titles, entity_types=None, all_question_items=all_question_items )                                    

                loss = torch.tensor([0.0],device=self.answer_generator.device)
                self.batch_idx += 1
                return dict(predictions=predictions, answers=answers, original_answers=original_answers, questions=questions, loss=loss)

    def compute_loss(self, inputs, pl_model, kb=None, entity_kb=None, dp=None,  return_outputs=False):        
        relevant_passages = inputs.pop('relevant_passages', None)        
        answers = inputs.pop('answers', None)
        original_answers = inputs.pop('original_answers', None)
        questions = inputs.pop('questions', None)        
        question_titles =  inputs.pop('question_titles', None)         
        qids = inputs.pop('qids', None) 
        question_types =  inputs.pop('question_types', None) 
        all_question_items = inputs.pop('all_question_items', None)         
        if self.training:
           outputs = self(inputs, only_queries=False) 
        else:        
            outputs = self(inputs, only_queries=False)         
        question_outputs = outputs.get('question_outputs', None)               

        question_image_outputs = None 
        kb_question_image_outputs = None     
        if self.kwargs['trainee_kwargs'].get('encode_image', None) : #NOTE TODO 
            question_image_outputs = outputs.get('question_image_outputs', None)
            kb_question_image_outputs = outputs.get('kb_question_image_outputs', None)
        
        if self.reader_kwargs.get('sbf_rag', None):
           return self.sbf_rag( question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs)
        
        if self.reader_kwargs.get('generator_only', None):
           return self.Generate( question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs)

        if self.reader_kwargs.get('rag_training', None):
           return self.rag( question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs)
        
        if self.reader_kwargs.get('entity_prompt', None):
           return self.MiRAG( question_outputs, question_image_outputs, qids, questions, question_titles, relevant_passages, answers, original_answers, question_types,
                 kb, entity_kb, all_question_items, inputs, kb_question_image_outputs)             
        




