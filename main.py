"""
@ Author: Omar
"""
import os
import argparse
from train import train
from utils.utils import build_kwargs
import timeit
import warnings
warnings.filterwarnings('ignore')
import json

'''
==============================================================================  
'''
parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length",default=256,type=int,help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

parser.add_argument("--question_max_seq_length",default=256,type=int,help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.") 

parser.add_argument("--grad_check",action="store_true",default=False,help=" pretrain on triviaqa")
parser.add_argument("--train_batch_size",default=4,type=int, help=" batch size for training.")
parser.add_argument("--eval_batch_size",type=int, required=True, help=" batch size for evaluation.") 
parser.add_argument("--viquae_dev_batch_size",type=int, default=1000, help=" batch size for evaluation.") 
parser.add_argument("--transformer_model_name", default="bert-base-uncased", type=str, help="")   

parser.add_argument('--gradient_accumulation_steps',type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
parser.add_argument("--lr_decay",action="store_true",default=False,help=" learning rate decay ") 

parser.add_argument("--adam_beta1",default=0.9,type=float,help="")
parser.add_argument("--adam_beta2",default=0.999,type=float,help="") 
parser.add_argument("--warmup_steps",default=4,type=int,help="")
parser.add_argument("--adam_epsilon",default=1e-8,type=float,help="")
parser.add_argument("--num_train_epochs", default=40, type=int,help="Total number of training epochs to perform.")

parser.add_argument("--learning_rate", default=2e-5,type=float,help="The initial learning rate for Adam.")
parser.add_argument("--dropout",default=0.5,type=float,help="dropout")
parser.add_argument("--bert_hidden_size",default=768,type=int, help=" bert hidden size ")    
parser.add_argument("--clip_lr",type=float,help=" clip learning rate for Adam.")
parser.add_argument("--dpr_lr",type=float,help=" clip learning rate for Adam.")
parser.add_argument("--sanity_run",action="store_true",default=False,help=" sanity_run ") 
parser.add_argument("--train_sanity_run",action="store_true",default=False,help=" train_sanity_run ") 
'''--- Misc ----'''
parser.add_argument("--output_dir",type=str,help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--main_dir",default='None',type=str,help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--experiment_dir",default='None',type=str,help="experiment_dir.")
parser.add_argument("--experiment_name", type=str, help=" experiment name")   
parser.add_argument("--search",  type=str, help="config file for search")

'''--- pt LIGHTNING ----'''
parser.add_argument("--tune_loss",action="store_true",default=False,help=" tune model on loss")
parser.add_argument("--tune_dev",action="store_true",default=True,help=" tune on dev set")
parser.add_argument("--tune_valid_loss",action="store_true",default=False,help=" tune on validation loss")
parser.add_argument("--sanity_val_steps", default=0, type=int, help="sanity check")
parser.add_argument("--cpu",action="store_true",default=False,help=" use cpu")
parser.add_argument("--keep_in_memory",action="store_true",default=False,help=" keep_in_memory")
parser.add_argument("--nbr_gpus", default=0, type=int, help="nbr gpus")
parser.add_argument("--N", default=0, type=int, help="nbr of node machines")
parser.add_argument("--nbr_workers", default=1, type=int, help="nbr workers")
parser.add_argument("--num_proc", default=1, type=int, help="num_proc for map function")
parser.add_argument("--local_cache", type=str, default=os.path.join(os.path.expanduser( '~' ),'my_transformers_cache') , help=" local cache for transfomers models and files")
parser.add_argument("--IMAGE_PATH", type=str,help=" Image dir")
parser.add_argument("--transformer_path", type=str, help=" absolute transformer path ")
parser.add_argument("--resume_from", type=str, help=" resume training from checkpoint")
parser.add_argument("--checkpoint", type=str, help=" load pretrained model")
parser.add_argument("--non_strict_load", action="store_true", help=" non stric load of pretrained models")
parser.add_argument("--trainer_checkpoint", type=str, help=" load pretrained model")
parser.add_argument("--config", type=str, help=" config kwargs")
parser.add_argument("--grad_accum",  default=1, type=int, help="accumulated grad batch size")

'''------- Transformer Tokenization Args -------'''

parser.add_argument("--cls_token", default="[CLS]", type=str, help="")
parser.add_argument("--sep_token", default="[SEP]", type=str, help="")
parser.add_argument("--pad_token", default="[PAD]", type=str, help="")
parser.add_argument("--cls_token_at_end",action="store_true",default=False,help="")
parser.add_argument("--mask_padding_with_zero",action="store_true",default=True,help="")
parser.add_argument("--pad_on_left",action="store_true",default=False,help="")

parser.add_argument("--sequence_a_segment_id", default=0, type=int, help="")
parser.add_argument("--pad_token_segment_id", default=0, type=int, help="")
parser.add_argument("--cls_token_segment_id", default=0, type=int, help="")
parser.add_argument("--pad_token_label_id", default=0, type=int, help="")
parser.add_argument("--pad_token_id", default=0, type=int, help="")
''' misc '''
parser.add_argument("--test",action="store_true",default=False,help=" run evaluation on test")
parser.add_argument("--validate",action="store_true",default=False,help=" run evaluation on val set")
parser.add_argument("--debug",action="store_true",default=False,help=" debug mode ")
parser.add_argument("--stop_debug",action="store_true",default=False,help=" stop_debug mode ")
parser.add_argument("--device", default=None, help="device")
parser.add_argument('--seed', type=int, default=32, help="random seed for initialization")
parser.add_argument("--val_check_interval",type=int, help=" check every N train batches ")
args = parser.parse_args()  

args.main_dir = os.path.dirname(os.path.realpath(__file__)) 
args.local_cache = os.path.join(os.path.expanduser( '../' ),'my_transformers_cache')  
if not os.path.exists( args.local_cache ):
  os.makedirs(os.path.join( args.local_cache ))  
  
if args.transformer_model_name == 'bert-base-uncased' :
   args.transformer_model_name = 'bert-base-uncased'
   args.cls_token = "[CLS]"; args.sep_token="[SEP]"; args.pad_token="[PAD]"; args.cls_token_at_end=False; args.sequence_a_segment_id=0; args.pad_token_segment_id=0 ; \
      args.cls_token_segment_id=0; args.pad_token_label_id=0;args.pad_token_id=0; args.mask_padding_with_zero=True; args.pad_on_left=False; 

if __name__=="__main__": 
  start = timeit.default_timer()     
   
  if args.resume_from != None: #XXX not tested yet
      loaded_checkpoint =  args.checkpoint
      resume_checkpoint  = args.resume_from

      new_num_train_epochs =  args.num_train_epochs    
      new_learning_rate =  args.learning_rate
      new_grad_check = args.grad_check         
      json_dict = json.load(open( os.path.join(args.resume_from,  "experiment_params.json" ), 'r' ))
      vars(args)['experiment_dir'] = args.resume_from

  kwargs = build_kwargs(args)
  train.train_eval(args, **kwargs) 
  stop = timeit.default_timer()
  print('Time: ', (stop - start)/60 , ' mn')
   

