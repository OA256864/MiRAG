
import os
import pytorch_lightning as pl
import torch
from torch.optim import Adam, SGD
from torch.optim import Adam, RMSprop, SGD, Adagrad, Adadelta, Rprop, ASGD, LBFGS
import numpy as np
from models.trainee import * 
import warnings
import tqdm
import json
from pathlib import Path
from evaluation.metrics import * 
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR
from evaluation.metrics import retrieval  
import psutil
from functools import partial

OPTIMIZER_LIST = {
    "adam": Adam,
    "rmsprop": RMSprop,
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'rprop': Rprop,
    'asgd': ASGD,
    'lbfgs':LBFGS,   
}

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
   

class Trainer(pl.LightningModule):
  def __init__(self, args, lr=None, warmup_steps=None , **kwargs):
      super().__init__()

      self.args = args      
      self.train_data = None
      self.dev_data = None
      self.test_data = None        

      self.learning_rate = self.args.learning_rate if lr==None else lr  
      self.warmup_steps = self.args.warmup_steps if warmup_steps==None else warmup_steps

      self.dev_results = 0
      self.test_results = 0 
      self.log_output_file = open(  os.path.join( args.experiment_dir, 'log.txt' )  , 'a')         
      self.mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 
      self.param_groups = self.parameters()  
      self.kwargs = kwargs

  def init(self,):       
      if self.args.grad_check:
        self.apply(partial(self._set_gradient_checkpointing, value=True))        
      print("Gradient checkpointing = ", self.is_gradient_checkpointing)
      
  @property
  def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())

  def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


  def configure_optimizers(self):        
    if 'freeze_prefixes' in self.kwargs['trainee_kwargs']:               
        for name, param in self.model.named_parameters():            
            if any(prefix in name for prefix in self.kwargs['trainee_kwargs']['freeze_prefixes']):            
               param.requires_grad = False  
    
    def assign_multiple_lr( args, all_params_group , excluded_prefixes, learning_rates):        
        optimizer_grouped_parameters = [ {'params': [p for param_name, p in all_params_group if  not any(pref in param_name for pref in excluded_prefixes) ], 
                                    'weight_decay': 1e-4, 'lr': self.args.learning_rate} ]        
       
        for i,prefix in enumerate(excluded_prefixes):
           excluded_params = [ {'params': [p for param_name, p in all_params_group if  prefix in param_name ], 
                                    'weight_decay': 1e-4, 'lr': learning_rates[i]  } ]              
           optimizer_grouped_parameters.extend(excluded_params)

        return  optimizer_grouped_parameters         

    if self.args.clip_lr and self.args.dpr_lr:
        all_params = list(self.model.named_parameters())        
        all_params_keys = dict(self.model.named_parameters()).keys()
        if any( 'image_encoder' in param_name for param_name in all_params_keys ):
           all_params = assign_multiple_lr( self.args, all_params , ['image_encoder', 'question_model'], [self.args.clip_lr, self.args.dpr_lr]) 
        optimizer = torch.optim.AdamW(all_params, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon )    

    elif self.args.dpr_lr:
        all_params = list(self.model.named_parameters())        
        all_params_keys = dict(self.model.named_parameters()).keys()
        if any( 'question_model' in param_name for param_name in all_params_keys ):
           all_params = assign_multiple_lr( self.args, all_params , ['question_model'], [self.args.dpr_lr]) 
        optimizer = torch.optim.AdamW(all_params, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon )   
    
    elif self.args.clip_lr:
        all_params = list(self.model.named_parameters())        
        all_params_keys = dict(self.model.named_parameters()).keys()
        if any( 'image_encoder' in param_name for param_name in all_params_keys ):
           all_params = assign_multiple_lr( self.args, all_params , ['image_encoder'], [self.args.clip_lr]) 
        optimizer = torch.optim.AdamW(all_params, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon )   

    else:
        optimizer = torch.optim.AdamW(self.param_groups, lr=self.learning_rate, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon )

    train_len = self.trainer.datamodule.dp.train_len if self.trainer.datamodule.dp.train_len != None else len(self.trainer.datamodule.train_dataloader())    
    total_steps = int( ( train_len ) / self.args.train_batch_size / self.args.gradient_accumulation_steps) * self.args.num_train_epochs        
    
    if self.args.lr_decay : 
        if self.kwargs.get('answer_generator_kwargs', None):
          if  self.kwargs.get('answer_generator_kwargs', None)['class_name'] !=  "Blip2ForConditionalGeneration":  
              lr_scheduler =  LinearLR(optimizer, total_iters=10, start_factor=1, end_factor=0.00)               
              return [optimizer], [lr_scheduler]  
          
        return [optimizer]
    else:
        return [optimizer] #, [lr_scheduler]  
 
  def batched_cpu(self, batch):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
  
  def on_before_zero_grad(self, *args, **kwargs):
      if self.args.debug:
         print(f"\n[INFO] GPU Memory Allocated  on_before_zero_grad: {torch.cuda.memory_allocated() / 1e6:.2f} MB")  

  def on_before_optimizer_step(self, *args, **kwargs):
      if self.args.debug:
         print(f"[INFO] GPU Memory Allocated  on_before_optimizer_step: {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")  

  def on_before_backward(self, *args, **kwargs):
      if self.args.debug:
         print(f"[INFO] GPU Memory Allocated  on_before_backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")  

  def on_after_backward(self, *args, **kwargs):
      if self.args.debug:
         print(f"[INFO] GPU Memory Allocated  on_after_backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")  

  def on_train_batch_end(self, *args, **kwargs):       
      if self.args.debug:
         print(f"[INFO] GPU Memory Allocated  on_train_batch_end: {torch.cuda.memory_allocated() / 1e6:.2f} MB")  

  def on_validation_start(self,):
      if self.args.debug:
         print(f"[INFO] GPU Memory Allocated on validation start: {torch.cuda.memory_allocated() / 1e6:.2f} MB") 
         print(f"[INFO] Used RAM: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")  

  def on_train_start(self,):      
      print(f"\n[INFO] GPU Memory Allocated on train start: {torch.cuda.memory_allocated() / 1e6:.2f} MB")        
      print(f"[INFO] Used RAM: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")        

  def step(self, batch, batch_idx):
      raise NotImplementedError("Subclass and implement step.")
  
  def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)        
        self.log("train_loss", outputs['loss'])  
        return outputs
    
  def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.step(batch, batch_idx) 
        if 'batch_size' in outputs:
            self.log("val_loss", outputs['loss'],  batch_size=outputs['batch_size'])         
        else: 
            self.log("val_loss", outputs['loss'])      
        return self.batched_cpu(outputs)
    
  def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.step(batch, batch_idx)        
        if 'batch_size' in outputs:
            self.log("test_loss", outputs['loss'],  batch_size=outputs['batch_size'])         
        else: 
            self.log("test_loss", outputs['loss'])
        return self.batched_cpu(outputs)
    
  def eval_epoch_end(self, eval_outputs):
        warnings.warn("eval_epoch_end is not implemented.")
        return {}

  def training_epoch_end(self, outputs): 
      avg_loss = torch.stack( [out_dict['loss'] for out_dict in outputs]).mean()            
      self.log('avg_train_loss', avg_loss )
      self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

      self.log_output_file = open(  os.path.join( self.args.experiment_dir, 'log.txt' )  , 'a')
      self.log_output_file.write( 'avg_train_loss' + ' ' + str(avg_loss) + '\n')   
      self.log_output_file.write( 'lr' + ' ' + str(self.trainer.optimizers[0].param_groups[0]['lr']) + '\n')   
      self.log_output_file.close()    

      if 'eta' in outputs[0]:      
        avg_eta = torch.mean(torch.stack([out_dict['eta'] for out_dict in outputs]), dim=0)         
        print("\neta", self.model.eta)         

  def validation_epoch_end(self, outputs):
        """eval_epoch_end and log"""   
        metric_output = self.eval_epoch_end(outputs) 

        self.log_output_file = open(  os.path.join( self.args.experiment_dir, 'log.txt' )  , 'a')
        self.log_output_file.write( str(metric_output) + '\n')   
        self.log_output_file.close()
        
        if type(metric_output) == dict: 
           for k, v in metric_output.items():
               if f"eval_{k}" in self.kwargs["monitor_metric"].split(','):
                  self.log(f"eval_{k}", v)
        else:
            # Multiple valid dataloader
            for i in range(len(metric_output)):   
                metrics = metric_output[i]                
                # if i==1: #NOTE monitor only metrics of the second dataloader for model selection
                # iterate over the evaluation metrics of all validation dataloaders
                for k, v in metrics.items():
                    if f"eval_{k}_{i}" in self.kwargs["monitor_metric"].split(','):
                       self.log(f"eval_{k}_{i}", v)
            
  def test_epoch_end(self, outputs):
        """eval_epoch_end and log"""
        metric_output = self.eval_epoch_end(outputs) 

        self.log_output_file = open(  os.path.join( self.args.experiment_dir, 'log.txt' )  , 'a')
        self.log_output_file.write( '\n[TEST]' + '\n') 
        self.log_output_file.write( str(metric_output) + '\n')   
        self.log_output_file.close()
        
        if type(metric_output) == dict: 
           for k, v in metric_output.items():
               if f"eval_{k}" in self.kwargs["monitor_metric"].split(','):
                  self.log(f"eval_{k}", v)
        else:
            # Multiple valid dataloader
            for i in range(len(metric_output)):   
                metrics = metric_output[i]                
                # if i==1: #NOTE monitor only metrics of the second dataloader for model selection
                # iterate over the evaluation metrics of all validation dataloaders
                for k, v in metrics.items():
                    if f"eval_{k}_{i}" in self.kwargs["monitor_metric"].split(','):
                       self.log(f"eval_{k}_{i}", v)   

'''======================================================================================'''
def get_pretrained_model(args, **kwargs):
    def load_checkpoint(pretrained_model, args,**kwargs ):  
        if 'checkpoint' in kwargs: # in case of resuming training from chpt, splitted encoders might be missing          
            pretrained_model.load_state_dict(torch.load( kwargs['checkpoint'] ), strict=True)         
            print('loaded checkpoint:', kwargs['checkpoint'], '\n')  
              
        return pretrained_model
    
    Class = get_class(args, **kwargs)  

    # if issubclass(Class.__bases__[0], nn.Module):
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

class RA_Reader(Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)        

        self.args = args
        model_class = get_class(args, **kwargs['trainee_kwargs'])
        self.model = model_class(args, **kwargs)   

        self.loss_fct = torch.nn.NLLLoss(reduction='mean')
        self.init()        
        
    def step(self, inputs, batch_idx):
        entity_kb = None
        if hasattr(self.trainer.datamodule.dp, "entity_kb"): 
           entity_kb = self.trainer.datamodule.dp.entity_kb 

        kb = None
        if hasattr(self.trainer.datamodule.dp, "kb"): 
           kb = self.trainer.datamodule.dp.kb    

        outputs = self.model.compute_loss( inputs, self, kb=kb, entity_kb=entity_kb, dp=self.trainer.datamodule.dp)        
        return outputs
    
    def eval_generator_answers(self, eval_outputs):
        predictions = []
        all_answer_strings = []
        all_original_answer_strings = []
        questions = []
        for batch in eval_outputs:        
            predictions.extend( batch['predictions'])
            if batch['answers'] != None:
               all_answer_strings.extend(batch['answers'])
            else:
                all_answer_strings.extend([[ans] for ans in batch['original_answers']])

            questions.extend(batch['questions'])            

        if self.args.test or self.args.validate:
           root_log = Path( self.args.experiment_dir )
           prefix = 'val_' if self.args.validate else 'test_'
           prediction_file_name = prefix + 'predictions.json' 
           with open(root_log/prediction_file_name, 'wt') as file:
               json.dump(predictions, file)        
        
        answer_metrics = squad(predictions=predictions, references=all_answer_strings)
        return answer_metrics
    
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
        return self.trainer.datamodule.tokenizer.batch_decode(answers, skip_special_tokens=True)
    
    def pad_and_cat(self, arrays, padding_index=-100):        
        N, M, L = arrays[0].shape
        for array in arrays[1:]:
            n, m, l = array.shape
            assert m == M
            L = max(l, L)
            N += n
        # result = np.full_like(arrays[0], padding_index, shape=(N, M, L))
        result = torch.full((N,M,L), padding_index)

        N = 0
        for array in arrays:
            n, _, l = array.shape
            result[N:N+n, :, :l] = array
            N += n
        return result
    
    def eval_reader_answers(self, eval_outputs): 
        predictions = []
        all_answer_strings = []
        all_original_answer_strings = []
        questions = []
        for batch in eval_outputs:        
            predictions.extend( batch['predictions'])
            if batch['answers'] != None:
               all_answer_strings.extend(batch['answers'])
            else:
                all_answer_strings.extend([[ans] for ans in batch['original_answers']])

            questions.extend(batch['questions'])            

        if self.args.test or self.args.validate:
           root_log = Path( self.args.experiment_dir )
           prefix = 'val_' if self.args.validate else 'test_'
           prediction_file_name = prefix + 'predictions.json' 
           with open(root_log/prediction_file_name, 'wt') as file:
               json.dump(predictions, file)        

        answer_metrics = squad(predictions=predictions, references=all_answer_strings)
        return answer_metrics
    
    def eval_epoch_end(self, eval_outputs):
        eval_metrics = {}

        if self.kwargs.get('answer_generator_kwargs', None):
            if self.kwargs.get('answer_generator_kwargs', None).get('generator_only', None):            
               answer_metrics = self.eval_generator_answers(eval_outputs)
               eval_metrics.update( answer_metrics) 
               print('eval_metrics',eval_metrics)
               return eval_metrics                       
            # elif self.kwargs['answer_generator_kwargs'].get('rag_training', None) or self.kwargs['answer_generator_kwargs'].get('rag', None) :
            else:
                answer_metrics = self.eval_generator_answers(eval_outputs)
                eval_metrics.update( answer_metrics)            

        elif self.kwargs.get('reader_kwargs', None):       
            answer_metrics = self.eval_reader_answers(eval_outputs)
            eval_metrics.update( answer_metrics)            
        
        retrieval_metrics =  retrieval(eval_outputs, ignore_index=self.loss_fct.ignore_index)     
        eval_metrics.update( retrieval_metrics)            
        print('eval_metrics',eval_metrics)
        return eval_metrics

