import csv
import os

import pandas as pd

from src.arguments import init_train_argparse

def write_config_to_csv(args, csv_file_path) -> None:
    if not os.path.isfile(csv_file_path):
        df = pd.DataFrame.from_dict(data=vars(args), orient="index").T
        df.insert(0, "accuracy", 0)
        df.to_csv(csv_file_path)
        return 0
    else:
        df = pd.read_csv(csv_file_path, index_col=0, on_bad_lines='skip')
        if len(list(set(vars(args).keys()) - set(df.columns))) == 0:
            df = df.append(vars(args), ignore_index=True)
            df.to_csv(csv_file_path)
            return df.index[-1]
        else:
            parser = init_train_argparse()
            default_args, unknown = parser.parse_known_args()
            columns_to_add = list(set(vars(args).keys()) - set(df.columns))
            for column in columns_to_add:
                data = [getattr(default_args, column)] * len(df.index)
                df.insert(len(df.columns), column, data)
            df = df.append(vars(args), ignore_index=True)
            df.to_csv(csv_file_path)
            return df.index[-1]


def change_column_value_of_existing_row(column, value, csv_file_path, run_id):
    df = pd.read_csv(csv_file_path, index_col=0)
    df.loc[run_id, column] = value
    df.to_csv(csv_file_path)