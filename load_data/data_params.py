"""
params about datasets.
"""
# you can select the dataset here
def get_dataInfo(dataname="weibo"):
    data_path = "./dataset/"
    unit_time = None
    observation_time = None
    prediction_time = None
    # the m smallest eigenvectors
    m = 32
    # observation time and prediction time
    if dataname == "weibo":
        observation_time = 3600 * 0.5
        prediction_time = [24 * 3600]
        unit_time = 3600
    elif dataname == "aps":
        observation_time = 365*3
        prediction_time = [365*20+5]
        unit_time = 365
    elif dataname == "twitter":
        observation_time = 3600*24*2
        prediction_time = [3600*24*32]
        unit_time = 3600*24
    elif dataname == "dblp":
        observation_time = 5
        prediction_time = [30]
        unit_time = 1
    return dataname, data_path, unit_time, observation_time, prediction_time, m