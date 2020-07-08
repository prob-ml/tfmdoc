
FILEPATH=$1

for ((group = 0; group <= 9; group++));
do
    echo "Start merge: group: "$group
    
    sort -t ',' $FILEPATH'2010_'$group'.csv' $FILEPATH'2011_'$group'.csv' $FILEPATH'2012_'$group'.csv' $FILEPATH'2013_'$group'.csv' $FILEPATH'2014_'$group'.csv' $FILEPATH'2015_'$group'.csv' $FILEPATH'2016_'$group'.csv' $FILEPATH'2017_'$group'.csv' $FILEPATH'2018_'$group'.csv' > $FILEPATH'group_'$group'.csv'
    
    echo "Finish merge: group: "$group
done

