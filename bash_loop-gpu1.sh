#!/bin/bash
for i in $(seq 0 2 24)
    do
       python Auto_train_caller.py Params/all_params_bert_toxic_rationale_target.json $((i)) 0 
       python Auto_train_caller.py Params/all_params_bert_toxic.json $((i+1)) 1
    done

# python Auto_train_caller.py Params/all_params_bert_toxic_rationale.json 24 1
# python Auto_train_caller.py Params/all_params_bert_toxic_rationale.json 8 1
# python Auto_train_caller.py Params/all_params_bert_toxic_rationale_target.json 96 1
# python Auto_train_caller.py Params/all_params_bert_toxic_rationale_target.json 9 1
