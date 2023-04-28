#!/bin/bash

for alg in 'vdn'
do
    for space in 'abstract' 'discret'
    do
        for l in 0.0 1.0
        do
            for s in 1234 3456
            do
                python main.py --alg=$alg --action_space=$space --seed=$s --level=$l --no_graphics --dr_coef
            done
        done
    done
done

for alg in 'vdn'
do
    for space in 'abstract' 'discret'
    do
        for l in 0.0 1.0
        do
            for s in 1234 3456
            do
                python main.py --alg=$alg --action_space=$space --seed=$s --level=$l --no_graphics
            done
        done
    done
done

