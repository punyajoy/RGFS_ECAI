#!/bin/bash


for tp in 50
do
    for rs in {2021..2025}
    do
#         for m in 17
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Basile' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Waseem' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Davidson' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Olid' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Founta' $tp $rs $m $r
#             done
#         done
        for m in 1
        do
            for r in 0
            do
                python Auto_train_DA_caller_2_new_nomodel_crossdomain.py 'Basile' $tp $rs $m $r
                python Auto_train_DA_caller_2_new_nomodel_crossdomain.py 'Waseem' $tp $rs $m $r
                python Auto_train_DA_caller_2_new_nomodel_crossdomain.py 'Davidson' $tp $rs $m $r
                python Auto_train_DA_caller_2_new_nomodel_crossdomain.py 'Olid' $tp $rs $m $r
                python Auto_train_DA_caller_2_new_nomodel_crossdomain.py 'Founta' $tp $rs $m $r
            done
        done
#         for m in 16
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Basile' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Waseem' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Davidson' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Olid' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_nomodel_crossdomain_RAFTplusHX.py 'Founta' $tp $rs $m $r
#             done
#         done
    done
done


# for tp in 50
#     do
#         for rs in {2021..2025}
#             do
#                  for m in 1
#                     do
#                         for r in 0
#                         do
#                             python Auto_train_DA_caller_2_new_nomodel_crossdomain_HXtoOtherDatasets.py 'Basile' $tp $rs $m $r             
#                             python Auto_train_DA_caller_2_new_nomodel_crossdomain_HXtoOtherDatasets.py 'Waseem' $tp $rs $m $r
#                             python Auto_train_DA_caller_2_new_nomodel_crossdomain_HXtoOtherDatasets.py 'Davidson' $tp $rs $m $r
#                             python Auto_train_DA_caller_2_new_nomodel_crossdomain_HXtoOtherDatasets.py 'Olid' $tp $rs $m $r
#                             python Auto_train_DA_caller_2_new_nomodel_crossdomain_HXtoOtherDatasets.py 'Founta' $tp $rs $m $r
#                         done
#                     done
#             done
#     done