# Traffic information
FROM_DATE='2009-01-01' #'2009-01-01'
TO_DATE='2024-01-01'
AGGREGATION='max'
RESAMPLING='weekly'
LAMBDA_0 = 300
UPGRADE_PERIOD = 90
ALPHA_PERCENT = 5

# Datarate information
IS_CONSTANT_BITRATE = True
CONSTANT_BITRATE_GBPS = 100
DATARATE_LIST_GBPS = [50, 75, 100]
DATARATE_SELECTION_PROBABILITIES = [0.2, 0.05, 0.75]
