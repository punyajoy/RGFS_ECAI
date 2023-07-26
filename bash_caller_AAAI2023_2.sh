#!/bin/bash

for tp in 50
do
    for rs in {2021..2025}
    do
        for m in 16
        do
            for r in 0
            do
                python Domain_adaptation.py 'Davidson' $tp $rs $m $r
            done
        done
    done
done

# for tp in 50
# do
#     for rs in {2021..2025}
#     do
#         for m in 27
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Founta' $tp $rs $m $r
#             done
#         done
#     done
# done