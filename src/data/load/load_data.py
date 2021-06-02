from pathlib import Path

import pandas as pd
import glob

class DataLoader:
    def __init__(
            self,
            variable: str,
            input_dir: Path
    ):
        self.variable = variable
        self.input_dir = input_dir

    def data_load(self):
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        data = None
        for file in files:
            dataset = pd.read_csv(file)
            if data is None:
                data = dataset
            else:
                data = data.append(dataset)
        return data


if __name__ == '__main__':
    DataLoader('pm25', Path('../../../data/processed/')).data_load()
