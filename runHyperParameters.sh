#!/usr/bin/env bash


for lr in  0.001 0.0001 0.00001 0.000005
do
    for model_type in eegnet shallow_eegnet deep_eegnet
    do
        python predictSeizureEEGNet.py with lr=$lr model_type=$model_type random_rearrange_each_batch=True
    done
done

for lr in  0.001 0.0001 0.00001 0.000005
do
    for model_type in eegnet shallow_eegnet deep_eegnet
    do
        python predictSeizureEEGNet.py with lr=$lr model_type=$model_type
    done
done
