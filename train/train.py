"""
@ Author: Omar
"""
import os
from models.trainer import * 
from data_processing.data_module import * 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from data_processing.loader import  *
from utils import utils
import pytorch_lightning as pl
import timeit


'''======================================================================================'''
def get_class(args, class_name, **kwargs):
    modules = dict(globals().items())    
    Class = modules[class_name]                       
    return Class 

def get_config_class(args, config_class_name, **kwargs):
    modules = dict(globals().items())    
    Class = modules[config_class_name]                       
    return Class       
'''======================================================================================''' 
def train_eval(args, **kwargs):
      start = timeit.default_timer()
      data_processor_class = get_class(args, **kwargs['data_module_kwargs']['data_processor'])
      dp = data_processor_class(args, kwargs)     
      dp.load_data() 

      args = dp.args # update args    
      
      if args.resume_from == None:
         # create model dir
         if not os.path.exists( os.path.join( *[args.main_dir , 'saved_models' ]) ):
            os.makedirs(os.path.join( *[args.main_dir , 'saved_models' ]))  

         ''' create experiment dir ''' 
         if args.experiment_name:
            args.output_dir = os.path.join( *[ 'saved_models' , args.experiment_name ])  
         else:            
            print("\nPlease set experiment_name arg !!!\n")
            exit()

         experiment_dir =  os.path.join( *[ args.main_dir , args.output_dir   ] )  
         experiment_dir = utils.create_experiment_dir(args, experiment_dir, **kwargs) 
         args.experiment_dir = experiment_dir         
      print( '[Experimesnt] ' + args.experiment_dir)  
      '''
      -----------------------------------------------------------------------------
      Set device
      -----------------------------------------------------------------------------
      '''

      device = utils.get_device()         
      if torch.cuda.is_available():
         if args.cpu:
            args.nbr_gpus = 0
            device = torch.device("cpu")
            args.device = "cpu"
         else:
              args.device = "cuda"
              if args.nbr_gpus == 0:
                 args.nbr_gpus = 1 
                 args.learning_rate = args.learning_rate *  args.nbr_gpus                       
      else:
         args.device = "cpu"
         args.nbr_gpus = 0

      if args.nbr_gpus > 1:
         acc = 'ddp'
      elif args.nbr_gpus == 1:
         acc = 'gpu'
      else:
         acc = 'cpu'
      
      '''
      -----------------------------------------------------------------------------
      Create models and trainer for different input seeds
      -----------------------------------------------------------------------------
      '''    
      data_module_class = get_class(args, **kwargs['data_module_kwargs'])
      data_module = data_module_class(args,dp, kwargs)           
      trainer_class = get_class(args, **kwargs['trainer_kwargs'])                
      model_trainer = trainer_class(args,**kwargs)
      
      
      '''
      -----------------------------------------------------------------------------
      Define callbacks
      -----------------------------------------------------------------------------
      '''        
      if args.tune_loss:   
         checkpoint_callback = ModelCheckpoint(monitor="avg_train_loss",dirpath= args.experiment_dir  , filename="model-{epoch:02d}-{train_loss:.4f}" + "_sd=" + str(args.seed) ,save_top_k=1, mode="min")    
      elif args.tune_valid_loss:   
         checkpoint_callback = ModelCheckpoint(monitor="avg_valid_loss",dirpath= args.experiment_dir  , filename="model-{epoch:02d}-{valid_loss:.4f}" + "_sd=" + str(args.seed) ,save_top_k=1, mode="min")       
      elif args.tune_dev:
         if "monitor_metric" in kwargs:
            monitor_vars = kwargs["monitor_metric"].split(',')
         else:         
            monitor_vars = ['eval_MRR@NM']     

         checkpoint_callbacks = []
         for monitor_var in monitor_vars:
             checkpoint_callback = ModelCheckpoint(monitor=monitor_var,dirpath= args.experiment_dir,
                                    filename="model-{epoch:02d}-{"+ monitor_var +":.4f}" + "_sd=" + str(args.seed) ,save_top_k=1, mode="max")   
             checkpoint_callbacks.append(checkpoint_callback)
      else:
            print('please select a tunnning option tune_loss | tune_dev ')
            exit(0) 
      all_callbacks = checkpoint_callbacks
      '''
      -----------------------------------------------------------------------------
      Init Lit models
      -----------------------------------------------------------------------------
      '''
      
      checkpoint_path = None
      if args.resume_from != None:  
         for file in os.listdir( args.resume_from):
            if file.endswith(".ckpt"):                     
               checkpoint_path = os.path.join(args.resume_from , file)
               break
      
      print('[INFO] args.nbr_gpus',args.nbr_gpus)    
      print("[INFO] args.learning_rate", args.learning_rate)  
      print('[INFO] args.dpr_lr', args.dpr_lr)      
      print('[INFO] args.clip_lr', args.clip_lr)      
      print('[INFO] args.train_batch_size',args.train_batch_size)         
      print('[INFO] args.eval_batch_size',args.eval_batch_size) 
      print('[INFO] args.num_train_epochs', args.num_train_epochs)

      VALIDATION_CHECK_INTERVAL = 1.0 # check every epoch
      if args.val_check_interval:
         VALIDATION_CHECK_INTERVAL = args.val_check_interval # check every N batches
      
      if args.trainer_checkpoint:
          if args.test:  
            print('\n[TEST Mode]\n')                 
            model_trainer = model_trainer.load_from_checkpoint(args.trainer_checkpoint, args=args, **kwargs )
            model_trainer.eval()    
            trainer = pl.Trainer(max_epochs=args.num_train_epochs, val_check_interval=VALIDATION_CHECK_INTERVAL, num_sanity_val_steps=args.sanity_val_steps, \
            callbacks=all_callbacks, reload_dataloaders_every_n_epochs=0, gpus=args.nbr_gpus, accelerator=acc, accumulate_grad_batches=args.grad_accum,            
            )          
            trainer.test( model_trainer, datamodule=data_module) 
            exit()

      trainer = pl.Trainer(max_epochs=args.num_train_epochs, val_check_interval=VALIDATION_CHECK_INTERVAL, num_sanity_val_steps=args.sanity_val_steps, \
            callbacks=all_callbacks, reload_dataloaders_every_n_epochs=0, gpus=args.nbr_gpus, accelerator=acc, accumulate_grad_batches=args.grad_accum,
            resume_from_checkpoint=checkpoint_path,
            )  
      '''num_sanity_val_steps(nbr of batches) is used to check the validation routine  without having to wait for the first validation check'''

      '''
      -----------------------------------------------------------------------------
      Train
      -----------------------------------------------------------------------------
      ''' 
      stop = timeit.default_timer()
      print('\n[INFO] Loading Time', (stop - start)/60 , ' mn', '\n')
      if args.test:  
        print('\n[TEST Mode]\n')
        trainer.test( model_trainer, datamodule=data_module)  
        exit()
      elif args.validate:
          print('\n[VAL Mode]\n')
          trainer.validate( model_trainer, datamodule=data_module)  
          exit()
      trainer.fit(model_trainer, data_module) 
      trainer.test( model_trainer, datamodule=data_module)   
      '''
      -----------------------------------------------------------------------------
      Dev results
      -----------------------------------------------------------------------------
      '''
      if model_trainer.local_rank == 0: 
         best_dev_score = trainer.checkpoint_callback.best_model_score.cpu().item()
         best_ckpt_path = trainer.checkpoint_callback.best_model_path
         print('best_ckpt_path = ', best_ckpt_path)
         print('best_model_score = ',  best_dev_score)         
         
      