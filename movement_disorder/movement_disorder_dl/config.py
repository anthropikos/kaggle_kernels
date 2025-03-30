from pathlib import Path


class BaseConfig():
    pass

class LaptopConfig(BaseConfig):
    DATA_DIR_PATH = (Path(__file__).parent / Path('../data/essential_tremor')).resolve()

class WorkstationConfig(BaseConfig):
    pass

class ClusterConfig(BaseConfig):
    pass