from loguru import logger

from typing import Dict, List, Optional, Tuple
from retry import retry
from tqdm import tqdm
from .units import *


class QCArchiveFetch:
    def __init__(
        self,
        qca_server: str,
        dataset_and_local_db_names: Dict[str, str],
        dataset_type: str,
        dataset_specifications: List[str],
        local_cache_dir: Optional[str] = "./",
    ):
        import os

        self.qca_server = qca_server
        self.dataset_and_local_db_names = dataset_and_local_db_names
        self.local_cache_dir = local_cache_dir
        self.dataset_type = dataset_type
        self.dataset_specifications = dataset_specifications
        os.makedirs(self.local_cache_dir, exist_ok=True)

        self.qca_energy_unit = unit.hartree
        self.qca_distance_unit = unit.bohr
        self.datasets_downloaded = False

        # parse the dataset names and local database names from the dictionary
        self.local_database_names = []
        self.dataset_names = []
        self._entry_names = {}

        for (
            dataset_name,
            local_database_name,
        ) in self.dataset_and_local_db_names.items():
            self.local_database_names.append(local_database_name)
            self.dataset_names.append(dataset_name)
            self._entry_names[dataset_name] = []

        # initialize dataset status, setting state to False for
        # each dataset and specification
        self.dataset_status = {}
        for key in self.dataset_and_local_db_names.keys():
            temp_status = {}
            for specification in self.dataset_specifications:
                temp_status[specification] = False
            self.dataset_status[key] = temp_status

        # check if the datasets have already been fully downloaded
        self.check_db_status()

    """
    Main class for fetching datasets from a QCArchive server and saving results to local sqlite databases.
    
    Provides multi-threaded downloading of datasets for improved performance using concurrent.futures and   
    uses the retry package to handle intermittent connection errors.  Because records are saved to 
    local databases, the datasets can be downloaded in chunks or resumed if the download is interrupted.
    This additionally provides data persistence if the Python kernel is restarted, preventing the need to
    re-download the entire dataset again
    
    Parameters
    ----------
    qca_server : str, required
        URL of the QCArchive server to fetch the dataset from.
    dataset_names_local_db : Dict[str, str], required
        Name of the datasets to fetch from the QCArchive server and associated local database name.
        Each dataset will be saved to a separate local database as specified in the dict.
        Datasets need be of the same dataset type and have the same dataset specifications.
    dataset_type : str, required
        Type of dataset to fetch from the QCArchive server, e.g., "SinglePoint", "Optimization", etc.
    dataset_specifications : List[str], required
        List of specifications to fetch from the QCArchive server for each dataset_name
    local_cache_dir : str, optional, default='./'
        Location to save local databases.
    """

    def download(
        self,
        force_download=False,
        n_threads: Optional[int] = None,
        unit_testing_max_records: Optional[int] = None,
    ):
        """
        Download datasets from the QCArchive server and save to local databases.

        This uses concurrent.futures to download the datasets in parallel.

        Parameters
        ----------
        force_download : bool, optional, default=False
            If True, the dataset will be downloaded, even if it exists in the local database.
        n_threads : int, optional, default=None
            Number of threads to use for downloading the dataset.
            If None, n_threads will be set to (number_of_cores - 2).
        unit_testing_max_records : int, optional, default=None
            If set, datapoints [0:unit_testing_max_records] will be considered for download.
            This is used to limit the number of QCA datapoints downloaded during unit testing.

        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import os

        if self.datasets_downloaded and not force_download:
            logger.info("Dataset already downloaded. Skipping download.")
            return
        if force_download:
            logger.info("Forcing download of dataset.")
            # reset data_set status
            for key in self.dataset_and_local_db_names.keys():
                temp_status = {}
                for specification in self.dataset_specifications:
                    temp_status[specification] = False
                self.dataset_status[key] = temp_status

        if n_threads is None:
            number_of_cores = os.cpu_count()
            if len(self.dataset_specifications) < number_of_cores - 2:
                n_threads = len(self.dataset_specifications)
            else:
                n_threads = number_of_cores - 2

        with tqdm(total=0) as pbar:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = []
                for specification in self.dataset_specifications:
                    for (
                        dataset_name,
                        local_database_name,
                    ) in self.dataset_and_local_db_names.items():
                        # if the dataset isn't marked as complete, download it
                        if not self.dataset_status[dataset_name][specification]:
                            futures.append(
                                executor.submit(
                                    self._fetch_from_qcarchive,
                                    dataset_name,
                                    self.dataset_type,
                                    specification,
                                    local_database_name,
                                    self.local_cache_dir,
                                    force_download,
                                    unit_testing_max_records,
                                    pbar,
                                )
                            )
                for future in futures:
                    name, specification, spec_summary = future.result()
                    self.dataset_status[name][specification] = True
                    if len(spec_summary) != 0:
                        local_db = self.dataset_and_local_db_names[name]
                        local_json = local_db.split(".sqlite")[0]

                        summary_old = self._read_from_json(
                            self.local_cache_dir, f"{local_json}.json"
                        )
                        if len(summary_old) != 0:
                            for dataset_name in spec_summary.keys():
                                summary_old[dataset_name][specification] = spec_summary[
                                    dataset_name
                                ][specification]
                        else:
                            summary_old = spec_summary

                        self._write_to_json(
                            self.local_cache_dir,
                            f"{local_json}.json",
                            summary_old,
                        )

        self.check_db_status(verbose=True)

    def _json_datetime_serial(self, obj):
        """JSON serializer for datetime objects that are not serializable by default in json"""
        from datetime import date, datetime

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError("Type %s not serializable" % type(obj))

    def _write_to_json(
        self, output_file_path: str, output_file_name: str, data_output: dict
    ):
        """
        Writes a dictionary to a json file.

        Parameters
        ----------
        output_file_path : str, required
            Path to write the output file.
        output_file_name : str, required
            Name of the output file.
        data_output : dict, required
            Dictionary to write to the output file.
        """
        import json

        json_object = json.dumps(data_output, default=self._json_datetime_serial)
        with open(f"{output_file_path}/{output_file_name}", "w") as outfile:
            outfile.write(json_object)

    def _read_from_json(self, input_file_path: str, input_file_name: str) -> dict:
        """
        Reads a dictionary from a json file.

        Parameters
        ----------
        input_file_path : str, required
            Path to read the input file.
        input_file_name : str, required
            Name of the input file.

        Returns
        -------
        dict
            Dictionary read from the input json file.

        """
        import json
        import os

        json_file = f"{input_file_path}/{input_file_name}"

        if os.path.isfile(json_file):
            with open(json_file, "r") as infile:
                data = json.load(infile)

            return data
        else:
            return {}

    def _check_db_status(
        self, input_file_path: str, input_file_name: str, specification: str
    ) -> Tuple[bool, int, int, str]:
        """
        Reads json files to check status of local databases.

        Parameters
        ----------
        input_file_path : str, required
            Path to read the input file.
        input_file_name : str, required
            Name of the input json file.
        specification : str, required
            Name of the QCArchive specification to check.

        Returns
        -------
        Tuple[bool, int, int, str]
            Tuple containing the status of the local database (i.e., True if fully downloaded),
            the number of records in the local database, the number of records on the QCArchive server,
            and the timestamp of the last update.

        """
        temp_data = self._read_from_json(
            input_file_path,
            input_file_name,
        )
        if len(temp_data) != 0:
            for dataset_name in temp_data.keys():
                if (
                    temp_data[dataset_name][specification]["n_records_in_db"]
                    == temp_data[dataset_name][specification]["n_records_on_qca"]
                ):
                    return (
                        True,
                        temp_data[dataset_name][specification]["n_records_in_db"],
                        temp_data[dataset_name][specification]["n_records_on_qca"],
                        temp_data[dataset_name][specification]["timestamp"],
                    )
                else:
                    return (
                        False,
                        temp_data[dataset_name][specification]["n_records_in_db"],
                        temp_data[dataset_name][specification]["n_records_on_qca"],
                        temp_data[dataset_name][specification]["timestamp"],
                    )
        else:
            return (False, 0, 0, 0)

    def check_db_status(self, verbose=True):
        """
        Checks the status of the local databases to determine if they have been fully downloaded.

        Parameters
        ----------
        verbose : bool, optional, default=True
            If True, the status of each dataset will be printed to the screen.
            If False, only internal status of each dataset will be updated.

        """

        # first set to True, if any datasets are False, we will set this to False
        self.datasets_downloaded = True

        for dataset_name, local_database in self.dataset_and_local_db_names.items():
            local_json = local_database.split(".sqlite")[0]
            if verbose:
                print(f"Dataset: {dataset_name}")
            for specification in self.dataset_specifications:
                (
                    status,
                    n_records_in_db,
                    n_records_on_qca,
                    timestamp,
                ) = self._check_db_status(
                    self.local_cache_dir, f"{local_json}.json", specification
                )
                if status:
                    self.dataset_status[dataset_name][specification] = True
                    if verbose:
                        print(f" specification : {specification}")
                        print(
                            f"  Fully downloaded :: {n_records_in_db} of {n_records_on_qca} records.)"
                        )
                        print(f"  last updated: {timestamp}")
                else:
                    self.datasets_downloaded = False

                    self.dataset_status[dataset_name][specification] = False
                    if n_records_on_qca != 0:
                        if verbose:
                            print(f" specification : {specification}")

                            print(
                                f"  Partially downloaded :: {n_records_in_db} of {n_records_on_qca} records.)"
                            )
                            print(f"  last updated: {timestamp}")
                    else:
                        if verbose:
                            print(
                                f" specification : {specification}\n  Not downloaded."
                            )

    @retry(delay=1, jitter=1, backoff=2, tries=50, logger=logger, max_delay=10)
    def _fetch_from_qcarchive(
        self,
        dataset_name: str,
        dataset_type: str,
        specification_name: str,
        local_database_name: str,
        local_path_dir: str,
        force_download: bool,
        unit_testing_max_records: Optional[int] = None,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[str, str, dict]:
        """
        Fetches a dataset from the QCArchive server and saves it to a local database.

        Parameters
        ----------
        dataset_name : str, required
            Name of the dataset to fetch from the QCArchive server.
        dataset_type : str, required
            Type of dataset to fetch from the QCArchive server.
        specification_name : str, required
            Name of the specification to fetch from the QCArchive server.
        local_database_name : str, required
            Name of the local database to save the dataset to.
        local_path_dir : str, required
            Path to the local database to save the dataset to.
        force_download : bool, required
            If True, the dataset will be downloaded, even if it exists in the local database.
        unit_testing_max_records : int, optional, default=None
            If set, datapoints [0:unit_testing_max_records] will be considered for download.
            This is used to limit the number of QCA datapoints downloaded during unit testing.
        pbar : tqdm, optional, default=None
            If set, a progress bar will be displayed.
        """

        from sqlitedict import SqliteDict
        from qcportal import PortalClient

        client = PortalClient(self.qca_server)
        ds = client.get_dataset(dataset_type, dataset_name)

        entry_names = ds.entry_names
        self._entry_names[dataset_name] = entry_names

        if unit_testing_max_records is None:
            unit_testing_max_records = len(entry_names)

        with SqliteDict(
            f"{local_path_dir}/{local_database_name}",
            tablename=specification_name,
            autocommit=True,
        ) as local_database:
            db_keys = set(local_database.keys())
            to_fetch = []

            if force_download:
                for name in entry_names[0:unit_testing_max_records]:
                    to_fetch.append(name)
            else:
                for name in entry_names[0:unit_testing_max_records]:
                    if name not in db_keys:
                        to_fetch.append(name)

            if pbar is not None:
                pbar.total = len(to_fetch) + pbar.total
                pbar.refresh()

            if len(to_fetch) > 0:
                if specification_name == "entry":
                    logger.debug(
                        f"Fetching {len(to_fetch)} entries from {dataset_name}."
                    )
                    for entry in ds.iterate_entries(
                        to_fetch, force_refetch=force_download
                    ):
                        local_database[entry.name] = entry
                        if pbar is not None:
                            pbar.update(1)
                else:
                    logger.debug(
                        f"Fetching {len(to_fetch)} records from {dataset_name} for spec {specification_name}."
                    )

                    for record in ds.iterate_records(
                        to_fetch,
                        specification_names=[specification_name],
                        force_refetch=force_download,
                    ):
                        # iterate_records returns records a tuple, with [0] = name, [1] = specification, [2] = record
                        local_database[record[0]] = record[2]
                        if pbar is not None:
                            pbar.update(1)

                from datetime import datetime

                n_records = len(list(local_database.keys()))

                spec_summary = {
                    dataset_name: {
                        specification_name: {
                            "n_records_in_db": n_records,
                            "n_records_on_qca": len(entry_names),
                            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                        },
                    }
                }
                return (dataset_name, specification_name, spec_summary)

        return (dataset_name, specification_name, {})

    def _process_record_names(
        self, local_path_dir: str, filenames: List[str], dataset_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Returns a list of records_names for which there are no errors for each dataset_name.

        If multiple specifications are provided, the intersection of the names for each specification will be returned.


        Parameters
        ----------
        local_path_dir : str, required
            Path to the local database.
        filenames : List[str], required
            List of filenames for the local databases.
        dataset_names : List[str], required
            List of dataset names.
        specification_names : List[str], required
            Name of the specifications to process; "entry" will be ignored because it does not define a computation.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary where keys correspond to each dataset_name, and items are lists of record_names for each dataset.

        """
        from tqdm import tqdm
        from sqlitedict import SqliteDict

        # if entry is defined, we will ignore it because it does not define a computation
        specification_names.pop("entry")

        sorted_keys_dict = {}
        non_error_keys = []

        for filename, dataset_name in zip(filenames, dataset_names):
            non_error_local = []
            for specification_name in specification_names:
                non_error_keys_temp = []
                input_file_name = f"{local_path_dir}/{filename}"

                # identify the set of molecules that do not have errors
                with SqliteDict(
                    input_file_name, tablename=specification_name, autocommit=False
                ) as spice_db:
                    spec_keys = list(spice_db_spec2.keys())
                    for key in spec_keys:
                        if spice_db[key].status.value == "complete":
                            non_error_keys_temp.append(key)
                non_error_local.append(non_error_keys_temp)

            if len(non_error_local) == 1:
                non_error_keys[dataset_name] = non_error_local

            else:
                intersec = non_error_local[0]
                for i in range(len(non_error_local) - 1):
                    intersec = set(intersec).intersection(non_error_local[i + 1])
                non_error_keys[dataset_name] = intersec

        return non_error_keys

    def non_error_records(self) -> Dict[str, List[str]]:
        """
        Identify which records do not have errors for each dataset. Returns a dict of record_names for each dataset.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary where keys correspond to each dataset_name, and values are lists of record_names for each dataset.
        """

        # Internally we store dataset name and local db name in a dictionary
        # We need to separate them for processing

        self._non_error_records = self._process_record_names(
            self.local_cache_dir, self.local_database_names, self.dataset_names
        )
        return self._non_error_records

    def entry_names(self, dataset_name: str) -> List[str]:
        """
        Returns a list of entry_names for a given dataset_name.

        Parameters
        ----------
        dataset_name : str, required
            Name of the dataset to return entry_names.

        Returns
        -------
        List[str]
            List of entry_names for the dataset_name.
        """
        return self._entry_names[dataset_name]

    def entry_names(self) -> Dict[str, List[str]]:
        """
        Returns a dict containing a list of entry names (val) for each dataset_name (key).

        Returns
        -------
        Dict[str, List[str]]
            Dictionary where keys correspond to each dataset_name, and values are lists of entry_names for each dataset.


        """
        return self._entry_names

    def get_record(self, entry_name: str, dataset_name: str) -> dict:
        """
        Returns a dictionary containing the QCA  record for a given entry_name and dataset_name.

        Parameters
        ----------
        entry_name : str, required
            Name of the entry to return.
        dataset_name : str, required
            Name of the dataset to return the entry from.

        Returns
        -------
        dict
            Dictionary containing the QCA entry record, where each key corresponds to a specification.


        """
        from sqlitedict import SqliteDict

        temp_data = {}

        for specification in self.dataset_specifications:
            with SqliteDict(
                f"{self.local_cache_dir}/{self.dataset_and_local_db_names[dataset_name]}",
                tablename=specification,
                autocommit=True,
            ) as local_database:
                temp_data[specification] = local_database[entry_name]

        return temp_data
