# Pipeline of preparing EMR data.

### Overview.

The `data_main.sh` is the main process controlilng the whole pipeline and the only file you need to modify. And here are some important points:

- ~Before start,~ Please specify where you want to save the output files with `OUTPATH` in `data_main.sh`. By default, it **automatically** directs to your directory: `/home/username/emr-data/`

- By default, we create tmp files for each original `diag_/proc_/pharm_201*` containing 3000 lines. These files are used to test if codes work as expected. If you want to run the program on whole dataset, just remove `--dev` in `data_main.sh` at
```
    python3 ./data_field.py --create_field_seq --dev --merge_field --path $OUTPATH 1>&2
```

- In order to run the whole pipeline, just use `data_main.sh` with four parameters (these parameters are explained below). And here are the name rule: `-s`: select, `-f`: field, `-m`: merge, `-c`: clean.

```
    ./data_main.sh -s y -f y -m y -c y
```
If you want to run part of the pipeline, you can change useless parameter values to `n` or just drop these paramters. For example, step 1 `-s` may take around 20 minutes to run, so you may want to just run it once, and then focus on the last three steps.


### Step 1. Select useful columns and remove illegal characters: -s

See `data_select.sh` for details.

There are four main steps: 

- Remove all `'.0'` so float IDs will be casted into `int`.

- Select useful columns (highlighted in blue) from `CDM_Data_Dictionary_72_ZIP5.xlsx`.

- Replace all whitespace `' '` with  `'_'`

- Sort data by `patid` - `date` - `position`(if available) in increasing orders.

### Step 2. Create diag/proc/pharm sequence and create user daily sequence: -f

See `create_field_seq()` in `data_field.py` for details

Call diag/proc/pharm as "field".

Add additional information to each filed record, each subfield is separated by `_` and kept as `key:vlaue`.

update: I remove the `_loc_` in `diag` so that using `'_'` to separate tokens makes no sense in practice. Since drug names contains `'_'` whereas cannot be separated.

   - `diag` record: `icd:9_diag:V700` 

   - `proc` record: `icd:9_proc:640`

   - `pharm` record: replace whitespace with `_`.

For each year:

- Separate user into 10 groups based on the last number of `Patid`.

- Create daily sequences: . The `field_seq` is ordered by `Position`.

- Store users's daily in-field sequence `(user, date, field_seq)`, where the order in `field_seq` is defined by `position`. Files are named `field_year_group.csv`

- Rename columns `patid, date, diags/procs/drugs`.

Merge data from different fields in the same year.

See `merge_field()` in `data_field.py` for details

- Full outer join three fields data in same year on `patid` and `date`. 

- Concatenate users' daily fields sequence in `diags` - `proc` - `drugs` order and save as extra column `seq`.



### Step 3. Merge data from different year and create user document: -m

See `data_merge.sh` for details: merge data from different years.

See `data_merge.py` for details: create user document, where each daily sequence is separated by `[SEP]`


### Step 4. (Optional) Remove useless files. -c

Recommendation: when your run the codes first time, you may keep these "useless" files(by dropping `-c` parameter) to see what are they.
