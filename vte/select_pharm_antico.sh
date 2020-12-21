#!/bin/bash

INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
OUTPATH='/home/'$USER'/emr-vte/'
OUTFILE=$OUTPATH'pharm_antico.csv'

# Write head
pharm='pharm_2010.csv'
sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,8,15,20,26,27| sed 's/ /_/g' | sed "s/\"//g" | head -1 > $OUTFILE

for ((year = 0; year <= 8; year++));
do
	pharm='pharm_201'$year'.csv'
    echo "Start: "$pharm
    for pharm_drug in `cat pharm_antico`
    do
    	sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,8,15,20,26,27 | egrep -i $pharm_drug | sed 's/ /_/g' | sed "s/\"//g" >> $OUTFILE
    done
done
