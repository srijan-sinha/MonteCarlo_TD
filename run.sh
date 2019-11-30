#!/bin/bash

# To see what the code does run: python ./code/agent.py --help

mkdir -p graphs/Part1/monte_carlo/first_visit
mkdir -p graphs/Part1/monte_carlo/every_visit
mkdir -p graphs/Part1/td/1
mkdir -p graphs/Part1/td/3
mkdir -p graphs/Part1/td/5
mkdir -p graphs/Part1/td/10
mkdir -p graphs/Part1/td/20
mkdir -p graphs/Part1/td/100
mkdir -p graphs/Part1/td/1000
mkdir -p graphs/Part2
mkdir -p graphs/Part3
mkdir -p graphs/Part4

python ./code/agent.py -m -t -r -e -l
