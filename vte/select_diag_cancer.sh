#!/bin/bash

INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
OUTPATH='/home/'$USER'/emr-vte/'
OUTFILE=$OUTPATH'diag_cancer.csv'

# Write head
diag='diag_2010.csv'
sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1,3-7,11 |sed 's/ /_/g' | head -1 > $OUTFILE

for ((year = 0; year <= 8; year++));
do
	diag='diag_201'$year'.csv'
    echo "Start: "$diag
    for diag_cancer in `cat diag_cancer_icd9`
    do
    	sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1,3-7,11 | egrep $diag_cancer |sed 's/ /_/g' >> $OUTFILE
    done
done
