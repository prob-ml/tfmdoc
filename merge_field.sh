#!/bin/bash

PATH='/home/liutianc/emr-data/merge/'


for year in $(seq 0 8)
do
    for user in $(seq 0 9)
    do
        echo "Start: 201"$year", usergroup: "$user

    	diag='diag_201'$year'_'$user'.csv'
        pharm='pharm_201'$year'_'$user'.csv'
        proc='proc_201'$year'_'$user'.csv'
        sort -t ',' -k 1n $PATH$diag -o $PATH$diag
        sort -t ',' -k 1n $PATH$pharm -o $PATH$pharm
        sort -t ',' -k 1n $PATH$proc -o $PATH$proc

        join -1 1 -2 1 $PATH$diag $PATH$pharm > $PATH'tmp.csv'
        join -1 1 -2 1 $$PATH'tmp.csv' $PATH$proc > $PATH'2010_'$user'.csv'

        echo "Finish: 201"$year", usergroup: "$user

    done
done
