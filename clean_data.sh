INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'

OUTPATH='/home/liutianc/emr-data/'

DIAG=$OUT_PATH'diag_201'

for i in $(seq 0 8)
do
	diag=$DIAG$i'.csv'
	echo $diag
	sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1-7,11 > $OUTPATH$diag
done