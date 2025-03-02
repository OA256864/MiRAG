"""
@ Author: Omar
"""
import torch
from utils.utils import *

def get_eval_passages( item, dp, M, search_key='BM25'):
    """Keep the top-M passages retrieved by the IR"""
    indices = item[search_key+"_indices"][: M]
    scores = item[search_key+"_scores"][: M]
    return dp.kb.select(indices)['passage'], scores

def get_answer_position( batch, answers, answer_mask, max_n_answers, M):
    """Adapted from DPR"""
    start_positions, end_positions = torch.zeros_like(answer_mask), torch.zeros_like(answer_mask)
    for j, (input_ids, answer) in enumerate(zip(batch['input_ids'], answers)):
        L = input_ids.size(-1)
        answer_starts, answer_ends = [], []
        for a in answer:
            answer_len = a.size(0)
            enough = False
            for i in range(L-answer_len+1):
                if (a == input_ids[i: i+answer_len]).all():
                    start, end = i, i+answer_len-1
                    if start not in answer_starts and end not in answer_ends:
                        answer_starts.append(start)
                        answer_ends.append(end)
                        if len(answer_starts) >= max_n_answers:
                            enough = True
                            break
            if enough:
                break
        for i, (start, end) in enumerate(zip(answer_starts, answer_ends)):
            start_positions[j, i] = start
            end_positions[j, i] = end
            # un-mask answer
            answer_mask[j, i] = 1
    start_positions = start_positions.view(-1, M, max_n_answers)
    end_positions = end_positions.view(-1, M, max_n_answers)
    answer_mask = answer_mask.view(-1, M, max_n_answers)
    batch.update(dict(start_positions=start_positions, end_positions=end_positions, answer_mask=answer_mask))
    return batch

def prepare_inputs(args, data_dict, tokenizer, encoder_type, kwargs=None):
    max_seq_len = args.question_max_seq_length if encoder_type=='question' else args.max_seq_length   
    tokenized_text = tokenize(args, kwargs, tokenizer, data_dict['input'], args.max_seq_length)     
    return  tokenized_text

def tokenize(args, kwargs, tokenizer, input_text, max_len):
    if kwargs.get('tokenization_kwargs',None):    
       tokenized_text = tokenizer(input_text, return_tensors='pt', truncation=True, **kwargs['tokenization_kwargs'])  
    else:
        tokenized_text = tokenizer(input_text, return_tensors='pt', padding='longest', truncation=True, max_length=max_len )
    return tokenized_text

def pad_passage_items(items, invalid_indices):    
    for i, item in enumerate(items):                
        if i in invalid_indices:           
           items[i] = ''                   
    return items   

def build_multimodal_batch(args, items, tokenizer, data_processor, batch_index, imageFormatter, kwargs, training, all_kwargs=None):   
    M = 2
    batch_size = len(items)
    n_relevant_passages = 1
    n_irrelevant_passages = M-n_relevant_passages
    labels = []
    type_labels = []        
    all_relevant_indices = []
    all_irrelevant_indices = []
    mentions_start_pos, mentions_end_pos, mentions, linked_entities_ids = None, None, None, None
    wikidata_ids = []
    all_relevant_passages = []
    all_irrelevant_passages = []

    all_relevant_items = []
    all_irrelevant_items = []
    
    label_idx =0
    ignored_indices = []
    question_texts = []
    answers = []
    original_answers = []

    question_titles = []
    context_titles = [] 
    qids = []  
    question_types = [] 

    valid_question_indices = []
    invalid_question_indices = []
    for i, item in enumerate(items):  
        if kwargs.get('input_entity_key', None):
            question_titles.append( item[kwargs['input_entity_key']] )
        elif 'provenance' in item['output']:
            question_titles.append( item['output']['provenance'][0]['title'][0] )

        question_texts.append( item[kwargs['input_key']])
        qids.append(item['id'])
        original_answers.append( item['output']['original_answer'])
        answers.append( item['output']['answer'])        
        question_types.append(item.get('question_type', None))

    labels = [i for i in range(len(question_texts))] 
    all_question_items = items

    RAG = False
    WO_KB = False
    GEN_ONLY = False
    if all_kwargs.get('answer_generator_kwargs', None):
       RAG =  all_kwargs['answer_generator_kwargs'].get('rag_training', None) or all_kwargs['answer_generator_kwargs'].get('entity_prompt', None)
       WO_KB = all_kwargs['answer_generator_kwargs'].get('wo_kb', None)
       GEN_ONLY = all_kwargs['answer_generator_kwargs'].get('generator_only', None)
    '''
    ========================
    Passages
    ========================
    '''
    tokenized_text = None       
    if not(WO_KB):       
       if len(all_relevant_indices) > 0: 
          if all_kwargs['trainee_kwargs'].get('irrelevant_sampling', None): 
             all_passage_items = data_processor.kb.select(all_relevant_indices + all_irrelevant_indices)
          else:
              all_relevant_items = data_processor.kb.select(all_relevant_indices)
              all_passage_items = all_relevant_items  

          all_passages = pad_passage_items(all_passage_items[kwargs['passage_key']], invalid_question_indices)
          all_relevant_passages = all_passages[:len(question_texts)]
       
          if hasattr(data_processor, "entity_kb"):
             all_passage_image_items = data_processor.entity_kb[ all_passage_items['index']]  
             if kwargs.get('passage_entity_key', None):       
               context_titles = pad_passage_items(all_passage_image_items[kwargs['passage_entity_key']], invalid_question_indices)            
             # get dictionary of lists from list of dictionaries       
             all_passage_image_items = [dict(zip(all_passage_image_items,t)) for t in zip(*all_passage_image_items.values())]
          elif 'context_image' in all_question_items[0]:
              all_passage_image_items = [{'image':item['context_image']} for item in all_question_items]          
       else:
            if GEN_ONLY: 
                if training:
                  return None
                else:
                   print('no passage data')
                   exit()
    '''
    ========================
    Questions
    ========================
    '''
    question_inputs = tokenize(args, kwargs, tokenizer, question_texts , args.max_seq_length)      
    '''
    ========================
    Build batches
    ========================
    '''    
    labels = torch.tensor(labels)
    type_labels = torch.tensor(type_labels)      
    batch = {}
    passage_batch = {}
    
    if not(WO_KB) and GEN_ONLY:
       if tokenized_text == None:
           tokenized_text = tokenize(args, kwargs, tokenizer, all_passages , args.max_seq_length)
           passage_batch.update(tokenized_text)
    
    batch = dict(question_inputs=question_inputs,
                 passage_inputs=passage_batch, labels=labels, type_labels=type_labels, irrelevant_type_labels=None, answers=answers,
                   questions=question_texts, relevant_passages=all_relevant_passages, irrelevant_passages=all_irrelevant_passages
                   , original_answers=original_answers, context_titles=context_titles, question_titles=question_titles, qids=qids, question_types=question_types ) 

    if all_kwargs['trainee_kwargs'].get('use_image', None):
       if kwargs.get('image_processor_kwargs', None):    
          question_image_batch, question_images = imageFormatter.format_pixels(all_question_items, ignored_indices=ignored_indices, invalid_indices=[])  
          #NOTE: uncomment if need passage image
        #   batch.update(question_image_inputs=question_image_batch, passage_image_inputs=passage_image_batch)
          batch.update(question_image_inputs=question_image_batch)

    if all_kwargs.get('answer_generator_kwargs', None):
       if all_kwargs.get('answer_generator_kwargs', None)['class_name'] ==  "Blip2ForConditionalGeneration":
           batch.update(all_question_items=all_question_items)
    return batch 


