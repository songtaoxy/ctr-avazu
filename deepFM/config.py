
# set the path-to-files
TRAIN_FILE = "/data/barnett007/ctr-data/train.csv"
TEST_FILE = "/data/barnett007/ctr-data/test.csv"

SUB_DIR = "/output"


NUM_SPLITS = 5
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    "id","click","hour","C1","banner_pos"
    ,"site_id","site_domain","site_category"
    ,"app_id","app_domain","app_category"
    ,"device_id","device_ip","device_model","device_type","device_conn_type"
    ,"C14","C15","C16","C17","C18","C19","C20","C21"
    ,"size"
]

NUMERIC_COLS = [
    "hour1","day","weekday"
]

IGNORE_COLS = [
    "id", "target","site_id", "app_id", "hour", "C15", "C16"
]
