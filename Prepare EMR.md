This document provides a pipeline of preparing EMR data .

### Step 1. Select useful columns and remove illegal characters.

See `clean_data.sh` for details.

There are four main steps: 

- Remove all `'.0'` so float IDs will be casted into `int`.

- Select useful columns (highlighted in blue) from `CDM_Data_Dictionary_72_ZIP5.xlsx`.

= Replace all whitespace `' '` with  `'_'`

- Sort data by `patid` - `date` - `position`(if available) in increasing orders.

### Step 2. Create diag/proc/pharm sequence.

See `create_field_seq()` in `prepare_emr.py` for details

Call diag/proc/pharm as "field".

Add additional information to each filed record, each subfield is separated by `_` and kept as `key:vlaue`.

   - `diag` record: `icd:9_loc:2_diag:V700`

   - `proc` record: `icd:9_proc:640`

   - `pharm` record: replace whitespace with `_`.

For each year:

- Separate user into 10 groups based on the last number of `Patid`.

- Create daily sequences: . The `field_seq` is ordered by `Position`.

= Store users's daily in-field sequence `(user, date, field_seq)`, where the order in `field_seq` is defined by `position`. Files are named `field_year_group.csv`

= Rename columns `patid, date, diags/procs/drugs`.

### Step 3. Merge data from different fields in the same year.

See `merge_field()` in `prepare_emr.py` for details

- Full outer join three fields data in same year on `patid` and `date`. 

- Concatenate users' daily fields sequence in `diags` - `proc` - `drugs` order and save as extra column `seq`.



### Step 4. Append data from different year.

TODO



