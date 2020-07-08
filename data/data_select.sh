#!/bin/bash

INPATH=$1
OUTPATH=$2

# if [ ! -d $OUTPATH ]; then
#   mkdir $OUTPATH
# fi

# for ((year = 0; year <= 8; year++));
# do
#     # Clean diag data.
#     diag='diag_201'$year'.csv'
#     echo "Start: "$diag
#     sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1,3-7,11 | sed 's/ /_/g' | sort  -t ',' -k 1n -k 7 -k 4n > $OUTPATH$diag
    
#     head -3000 $OUTPATH$diag > $OUTPATH'diag_201'$year'_tmp.csv'

#     echo "Finish: "$diag
    
#     # Clean pharm data.
#     pharm='pharm_201'$year'.csv'
#     echo "Start: "$pharm
#     sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,8,15,20,26,27 | sed 's/ /_/g' | sed "s/\"//g" | sort -t ',' -k 1n -k 3 > $OUTPATH$pharm

#     head -3000 $OUTPATH$pharm > $OUTPATH'pharm_201'$year'_tmp.csv'

#     echo "Finish: "$pharm
    
#     # Clean procedure data
#     proc='proc_201'$year'.csv'
#     echo "Start: "$proc
#     sed "s/\\.0//g" $INPATH$proc | cut -d "," -f 1,3-6,9 | sed 's/ /_/g' | sort -t ',' -k 1n -k 6 -k 5n > $OUTPATH$proc
    
#     head -3000 $OUTPATH$proc > $OUTPATH'proc_201'$year'_tmp.csv'
    
#     echo "Finish: "$proc    
# done

# # Clean member data.
# member='member_detail.csv'
# echo "Start: "$member
# sed "s/\\.0//g" $INPATH$member | cut -d "," -f 1,8,13 > $OUTPATH$member
# echo "Finish: "$member  