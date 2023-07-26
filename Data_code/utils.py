import numpy as np
from numpy import array, exp
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re


###this file contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)




##### We mostly use softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
 
def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Here we convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)
def sigmoid(z):
      """Compute sigmoid values"""
      g = 1 / (1 + exp(-z))
      return g

##### This function is used to aggregate the attentions vectors. This has a lot of options refer to the parameters explanation for understanding each parameter.
def aggregate_attention(at_mask,row):
    """input: attention vectors from 2/3 annotators (at_mask), row(dataframe row), params(parameters_dict)
       function: aggregate attention from different annotators.
       output: aggregated attention vector"""
    
    
    #### If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    if(row['final_label'] in ['normal','non-toxic']):
        at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
    else:
        at_mask_fin=at_mask
        #### Else it will choose one of the options, where variance is added, mean is calculated, finally the vector is normalised.   
        at_mask_fin=int(5)*at_mask_fin
        at_mask_fin=np.mean(at_mask_fin,axis=0)
        at_mask_fin=softmax(at_mask_fin)

    return at_mask_fin

def custom_tokenize(sent,tokenizer,max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    try:

        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            #max_length = max_length,
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
                            ' ',                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,
                    
                       )
          ### decide what to later

    return encoded_sent

def returnMask(row,tokenizer):
    
    text_tokens=row['post_tokens']
    
    ##### a very rare corner case
    if(len(text_tokens)==0):
        text_tokens=['dummy']
        print("length of text ==0")
    #####
    
    mask_all= row['rationales']
    mask_all_temp=mask_all
    count_temp=0
    while(len(mask_all_temp)!=3):
        mask_all_temp.append([0]*len(text_tokens))
    
    word_mask_all=[]
    word_tokens_all=[]
    
    for mask in mask_all_temp:
        if(mask[0]==-1):
            mask=[0]*len(mask)
        
        
        list_pos=[]
        mask_pos=[]
        
        flag=0
        for i in range(0,len(mask)):
            if(i==0 and mask[i]==0):
                list_pos.append(0)
                mask_pos.append(0)
            
            
            
            
            if(flag==0 and mask[i]==1):
                mask_pos.append(1)
                list_pos.append(i)
                flag=1
                
            elif(flag==1 and mask[i]==0):
                flag=0
                mask_pos.append(0)
                list_pos.append(i)
        if(list_pos[-1]!=len(mask)):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i+1]])
        
        
        word_tokens=[101]
        word_mask=[0]

        
        for i in range(0,len(string_parts)):
            tokens=custom_tokenize(" ".join(string_parts[i]),tokenizer)
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks


        # if(params['bert_tokens']):
        ### always post truncation
        word_tokens=word_tokens[0:(int(128)-2)]
        word_mask=word_mask[0:(int(128)-2)]
        word_tokens.append(102)
        word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
        
#     for k in range(0,len(mask_all)):
#          if(mask_all[k][0]==-1):
#             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if(len(mask_all)==0):
        word_mask_all=[]
    else:    
        word_mask_all=word_mask_all[0:len(mask_all)]
    return word_tokens_all[0],word_mask_all




def preprocess_func(text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent

def returnMask_test(row,tokenizer):
    text_tokens=row['text']
    #print(text_tokens)
    ##### a very rare corner case
    if(len(text_tokens)==0):
        text_tokens=['dummy']
        print("length of text ==0")
    #####
    mask_all= row['annotation']
    mask_all_temp=mask_all
    
#     count_temp=0
#     while(len(mask_all_temp)!=3):
#         mask_all_temp.append([0]*len(text_tokens))
    
    word_mask_all=[]
    word_tokens_all=[]
    for list_spans in mask_all_temp:
        list_spans.sort(key=lambda x: x[0])
        
        list_pos=[]
        mask_pos=[]


        for i in range(len(list_spans)):
            element=list_spans[i]
            if(i==0):
                if(element[0]!=0):
                    list_pos.append(0)
                    mask_pos.append(0)
                    list_pos.append(element[0])
                    mask_pos.append(1)
                    list_pos.append(element[1])
                else:
                    list_pos.append(element[0])
                    mask_pos.append(1)
                    list_pos.append(element[1])
            elif(i==len(list_spans)-1):
                
                if(element[1]!=len(text_tokens)):
                    if(list_pos[-1]!=(element[0]-1)):
                        mask_pos.append(0)
                        list_pos.append(element[0])
                    mask_pos.append(1)
                    list_pos.append(element[1])
                    mask_pos.append(0)
                    list_pos.append(len(text_tokens))
                else:
                    mask_pos.append(0)
                    list_pos.append(element[0])
                    mask_pos.append(1)
                    list_pos.append(element[1])
            else:
                if(list_pos[-1]!=element[0]):
                    mask_pos.append(0)
                    list_pos.append(element[0])
                mask_pos.append(1)
                list_pos.append(element[1])
                
        if(list_pos[-1]!=len(text_tokens)):
            mask_pos.append(0)
            list_pos.append(len(text_tokens))
        
        
        string_parts=[]
        flag=0
        for i in range(len(list_pos)-1):
            start=list_pos[i]
            string_parts.append(text_tokens[start:list_pos[i+1]])
            flag=0

        word_tokens=[101]
        word_mask=[0]
        for i in range(0,len(string_parts)):
            temp_text=preprocess_func(string_parts[i])
            tokens=custom_tokenize(temp_text,tokenizer)
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks


        # if(params['bert_tokens']):
        ### always post truncation
        word_tokens=word_tokens[0:(int(128)-2)]
        word_mask=word_mask[0:(int(128)-2)]
        word_tokens.append(102)
        word_mask.append(0)
        
        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
        
#     for k in range(0,len(mask_all)):
#          if(mask_all[k][0]==-1):
#             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if(len(mask_all)==0):
        word_mask_all=[]
    else:    
        word_mask_all=word_mask_all[0:len(mask_all)]
    
    
    if(len(word_mask_all[1])>len(word_mask_all[0])):
        attention=word_mask_all[1][0:len(word_mask_all[0])]
    else:
        attention=word_mask_all[1]+[0]*(len(word_mask_all[0])-len(word_mask_all[1]))

    word_mask_all[1]=attention
    return word_tokens_all[0],word_mask_all




def returnMask_test_new(row,tokenizer):
    text_tokens=row['text']
    tokens=custom_tokenize(text_tokens,tokenizer)
    word_mask_all=[0]*len(tokens)
    return tokens,word_mask_all
