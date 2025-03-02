import torch
import os
import warnings
from PIL import Image, ImageFile
from transformers import  CLIPProcessor,CLIPFeatureExtractor,CLIPTokenizer
from transformers import Blip2Processor
from torch import nn

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

def load_image(args,file_name):    
    path = os.path.join(file_name)
    try:
        image = Image.open(path).convert('RGB')
    except Exception as e:
        # warnings.warn(f"Caught exception '{e}' with image '{path}'")
        return None    
    if image.width < 1 or image.height < 1:
        warnings.warn(f"Empty image '{path}'")
        return None
    return image

class ImageFormatter:
    """
    Helper to format image features (precomputed or pixels) in nice square Tensors expected by mm models.
    """
    def __init__(self, args, kwargs, precomputed=False ):
        super().__init__()
        self.precomputed = precomputed    
        self.args = args
        self.kwargs = kwargs

        if precomputed :
           True 
        else:            
            self.feature_extractor = get_pretrained_model(args,**kwargs['feature_extractor_kwargs'])             
            if kwargs['feature_extractor_kwargs']['class_name'] == 'Blip2Processor':
                self.feature_extractor = self.feature_extractor.image_processor

        self.IMAGE_PATH =  os.path.realpath("../datasets/viquae/viquae_images/images/")
    
    def format_pixels(self, items, image_key='image', ignored_indices=None, invalid_indices=[]):
        """Load images and convert to tensors while handling padded passages"""
        """ignored indices correspond to item indices whose text (passage) is empty"""
        images, indices = [], []
        ignored_indices_len = 0 
        to_be_ignored = []
        if ignored_indices:
           ignored_indices_len = len(ignored_indices)
           to_be_ignored = ignored_indices
        
        image_idx = 0
        image_paths = []
        for i, item in enumerate(items):
            # in case of padding passage
            if image_key not in item:
                print('image key is missing !!!')
                continue
            if i in invalid_indices:
               image_idx+=1
               continue 

            if i in  to_be_ignored:
               continue   
            
            img_path = os.path.join(self.IMAGE_PATH, item[image_key])
            image_paths.append(img_path)
            image = load_image(self.args,img_path)            
            # trouble during loading. user is already warned
            if image is None:
                image_idx+=1
                continue
            indices.append(image_idx)
            images.append(image)
            image_idx += 1        
      
        # corner-case: only padding images
        if not images:
            # size = self.feature_extractor.size
            if 'shortest_edge' in self.feature_extractor.size:
               size = self.feature_extractor.size['shortest_edge']
            else:
                size = self.feature_extractor.size['height']

            pixel_values = torch.zeros(len(items) - ignored_indices_len, 3, size, size)            
            print('only padding images !!')            
            return dict(pixel_values=pixel_values),images 
        
        # resize and pad actual images using feature_extractor
        # N. B. this is three time slower than load_image (in cumulated time)
        image_features = self.feature_extractor(images, return_tensors="pt")
        b, c, h, w = image_features['pixel_values'].shape   
        
        # opposite corner-case: no padding image, no need for all this trouble        
        if ignored_indices_len == 0 and  b == len(items):           
           return image_features, images
        
        # there are some padded images to handle
        pixel_values = torch.zeros(len(items) - ignored_indices_len, c, h, w)
        indices = torch.tensor(indices)
        pixel_values[indices] = image_features['pixel_values']
        output = dict(pixel_values=pixel_values)        
        return output, images


