#!/bin/bash

# echo $*
select='n'
field='n'
merge='n'
clean='n'


INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
# OUTPATH='/nfs/turbo/lsa-regier/emr-data/'
OUTPATH='/home/'$USER'/emr-data/'

para_num=$#

usage() {
    echo -e " -s 'select data: [y/n]' \n -f 'create user-date-field data [y/n]' \n -m 'create user-seq data [y/n]' \n -c 'clean tmp files [y/n]' \n" 1>&2
#     exit 1
}

if [[ ${para_num} -lt 1 ]]; then
    echo "None of expected manipulation is specified: "
    usage
fi

while getopts ":s:f:m:c:" opt
do
    case $opt in
        s)
        select=$OPTARG;;
        f)
        field=$OPTARG;;
        m)
        merge=$OPTARG;;
        c)
        clean=$OPTARG;;
        ?)
        error=$OPTARG
        echo "Unknown parameter: ${error}, expect: "
        usage
        exit 1;;
    esac
done

if [[ ${select} == 'y' ]]; then
    echo "Start: select column from original data. "
    ./data_select.sh $INPATH $OUTPATH 
    echo "Finish: select column from original data. "
fi

if [[ ${field} == 'y' ]]; then
    echo "Start: create user-date-field data. "
    # NOTE: after making sure everything is fine, remove --dev here.
    python3 ./data_field.py --create_field_seq --dev --merge_field --path $OUTPATH 1>&2
    echo "Finish: create user-date-field data. "
fi 

if [[ ${merge} == 'y' ]]; then
    echo "Start: create user-seq data. "
    ./data_merge.sh $OUTPATH
    python3 ./data_merge.py --path $OUTPATH 1>&2
    echo "Finish: create user-seq data. "
fi 

if [[ ${clean} == 'y' ]]; then
    echo "Start: clean tmp files. "
    for (( year = 0; year <= 8; year ++ ));
    do
        for (( group = 0; group <= 9; group++ ));
        do
            rm $OUTPATH'201'$year'_'$group'.csv'
            rm $OUTPATH'diag_201'$year'_'$group'.csv'
            rm $OUTPATH'pharm_201'$year'_'$group'.csv'
            rm $OUTPATH'proc_201'$year'_'$group'.csv'
        done
    done
    echo "Finish: clean tmp files. "
fi 

echo "Finish all..."
