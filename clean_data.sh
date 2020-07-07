INPATH='/nfs/turbo/lsa-regier/OPTUMInsight_csv/'
OUTPATH='/home/liutianc/emr-data/'

for i in $(seq 0 8)
do
    # Clean diag data.
	diag='diag_201'$i'.csv'
	echo "Start: "$diag
	sed "s/\\.0//g" $INPATH$diag | cut -d "," -f 1,3-7,11 | sed "s/-//g" | sort  -t ',' -k 1n -k 7n -k 4n > $OUTPATH$diag
	echo "Finish: "$diag
    
    # Clean pharm data.
    pharm='pharm_201'$i'.csv'
    echo "Start: "$pharm
    sed "s/\\.0//g" $INPATH$pharm | cut -d "," -f 1,8,15,20,26,27 | sed 's/-//g' | sed "s/\"//g" | sort -t ',' -k 1n -k 3n -k 2 > $OUTPATH$pharm
    echo "Finish: "$pharm
    
    # Clean procedure data
    proc='proc_201'$i'.csv'
    echo "Start: "$proc
    sed "s/\\.0//g" $INPATH$proc | cut -d "," -f 1,3-6,9 | sed 's/-//g' | sort -t ',' -k 1n -k 6n -k 5n > $OUTPATH$proc
    echo "Finish: "$proc    
done

# Clean member data.
member='member_detail.csv'
echo "Start: "$member
sed "s/\\.0//g" $INPATH$member | cut -d "," -f 1,8,13 > $OUTPATH$member
echo "Finish: "$member  