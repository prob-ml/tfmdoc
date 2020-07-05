INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
OUTPATH='/home/liutianc/emr-data/'

for i in $(seq 0 8)
do
    # Clean diag data.
	diag='diag_201'$i'.csv'
	echo "Start: "$diag
	sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1,3-7,11 > $OUTPATH$diag
	echo "Finish: "$diag
    
    # Clean pharm data.
    pharm='pharm_201'$i'.csv'
    echo "Start: "$pharm
    sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,8,15,20,26,27 > $OUTPATH$pharm
    echo "Finish: "$pharm
    
    # Clean procedure data
    proc='proc_201'$i'.csv'
    echo "Start: "$proc
    sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,3-6,9 > $OUTPATH$proc
    echo "Finish: "$proc    
done

# Clean member data.
member='member_detail.csv'
echo "Start: "$member
sed "s/\\.0//g" $INPATH$member | cut -d "," -f 1,13 > $OUTPATH$member
echo "Finish: "$member  