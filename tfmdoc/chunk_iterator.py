import logging

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

log = logging.getLogger(__name__)


def chunks_of_patients(files, columns):
    queues = [ParquetFile.iter_row_groups(f, columns=c) for f, c in zip(files, columns)]
    chunks = [next(q, None) for q in queues]

    total_completed = 0
    is_eof = np.zeros(len(files), dtype="bool")

    while not np.all(is_eof):

        last_patid = np.asarray([c.Patid.max().astype(float) for c in chunks])
        last_patid += is_eof * 1e20
        slowest_queue = last_patid.argmin()

        first_incomplete_patid = np.asarray([c.Patid.max() for c in chunks]).min()
        completed_patients = [c[c.Patid < first_incomplete_patid] for c in chunks]
        completed_patients = pd.concat(completed_patients)
        chunks = [c[c.Patid >= first_incomplete_patid] for c in chunks]

        yield completed_patients

        try:
            chunks[slowest_queue] = chunks[slowest_queue].append(
                next(queues[slowest_queue])
            )
        except StopIteration:
            is_eof[slowest_queue] = True
            n_incomplete = np.sum(is_eof)
            log.info(f"{n_incomplete} queues complete")

        try:
            newly_completed = completed_patients.Patid.unique().shape[0]
        except AttributeError:
            newly_completed = completed_patients.patid.unique().shape[0]
        total_completed += newly_completed

    yield pd.concat(chunks)
