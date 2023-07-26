#!/bin/bash

# for tp in 50
# do
#     for rs in {2022..2025}
#     do
#         for m in 28
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Waseem' $tp $rs $m $r
#             done
#         done
#     done
# done

# start_time=$(date +%s)
# for tp in 50
# do
#     for rs in 2021
#     do
#         for m in 16
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Waseem' $tp $rs $m $r
#             done
#         done
#     done
# done
# end_time=$(date +%s)
# elapsed=$(( end_time - start_time ))
# echo "Time Taken"
# echo $elapsed


python Auto_train_DA_caller_2_new_adversarial_attacks.py 'Basile' 50 2021 17 0

start_time=$(date +%s)
for tp in 50
do
    for rs in 2021
    do
        for m in 16 17
        do
            for r in 0
            do
#                 python Auto_train_DA_caller_2_new_adversarial_attacks.py 'Waseem' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_adversarial_attacks.py 'Olid' $tp $rs $m $r
#                 python Auto_train_DA_caller_2_new_adversarial_attacks.py 'Davidson' $tp $rs $m $r
                python Auto_train_DA_caller_2_new_adversarial_attacks.py 'Founta' $tp $rs $m $r
            done
        done
    done
done
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Time Taken"
echo $elapsed

# for tp in 50
# do
#     for rs in {2021..2025}
#     do
#         for m in 24
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Basile' $tp $rs $m $r
#             done
#         done
#     done
# done

# for tp in 50
# do
#     for rs in {2021..2025}
#     do
#         for m in 29
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Basile' $tp $rs $m $r
#             done
#         done
#     done
# done

# for tp in 50
# do
#     for rs in {2021..2025}
#     do
#         for m in 25
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Olid' $tp $rs $m $r
#             done
#         done
#     done
# done

# for tp in 50
# do
#     for rs in {2021..2025}
#     do
#         for m in 30
#         do
#             for r in 0
#             do
#                 python Auto_train_DA_caller_2_new.py 'Olid' $tp $rs $m $r
#             done
#         done
#     done
# done


