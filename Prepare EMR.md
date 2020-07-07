This document provides a pipeline of preparing EMR data .

### Step 1. Select useful columns and remove illegal characters.

See `clean_data.sh` for details.

There are three main steps: 

1. Select useful columns (labeled as blue) from `CDM_Data_Dictionary_72_ZIP5.xlsx`.
2. Remove all `'.0'` in all columns so that float IDs will be casted into `int`.
3. Replace all whitespace `' '` with  `'_'`
4. Sort data based on `Patid` $\rightarrow$  `Date` $\rightarrow$ `Position`. in increasing order.

### Step 2. Create diag/proc/pharm sequence.

See `create_field_seq()` in `prepare_emr.py` for details

Call diag/proc/pharm as "field".

1. Add additional information to each filed record, each subfield is separated by `_` and kept as `key:vlaue`.

   - `diag` record: `icd:9_loc:2_diag:V700`

   - `proc` record: `icd:9_proc:640`
   - `pharm` record: `TODO: add an example`

For each year:

1. Separate user into 10 groups based on the last number of `Patid`.
2. Create respective daily sequences: `(user, date, field_seq)`. The `field_seq` is ordered by `Position`.
3. Store users's daily field sequence in different groups in `field_year_group.csv`
4. Rename column names as `patid, date, diags/procs/drugs`.

### Step 3. Merge data from different fields in the same year.

See `merge_field()` in `prepare_emr.py` for details

1. Join three fields data in same year by based on (patid, date). 
2. Concatenate users' daily fields sequence based on `diags` $\rightarrow$ `proc` $\rightarrow$ `drugs`. as an extra column `seq`.



### Step 4. Append data from different year.

TODO



