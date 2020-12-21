#!/bin/bash

INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
OUTPATH='/home/'$USER'/emr-vte/'
OUTFILE=$OUTPATH'proc_cancer.csv'

# Write head
proc='proc_2010.csv'
sed "s/\\.0//g" $INPATH$proc | cut -d "," -f 1,3-6,9 | sed 's/ /_/g' | head -1 > $OUTFILE

for ((year = 0; year <= 8; year++));
do
	proc='proc_201'$year'.csv'
    echo "Start: "$proc
    for proc_cancer in `cat proc_cancer_icd9`
    do
    	sed "s/\\.0//g" $INPATH$proc | cut -d "," -f 1,3-6,9 | egrep $proc_cancer | sed 's/ /_/g' >> $OUTFILE
    done
done
