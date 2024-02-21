from generator_based_on_Seattle import *
from parameters import *
# generate traffic 

from pathlib import Path

def generate_for_load(LAMBDA = 100, ALPHAS_LIST = [2]):
	parent_folder: str = 'generated_traffic_data'
	load_folder_path: str = parent_folder + '/' + 'load' + '_' + str(LAMBDA)
	actual_traffic_folder_path: str = load_folder_path + '/' + 'actual_traffic'
	predicted_traffic_folder_path: str = load_folder_path + '/' + 'predicted_traffic'


	# Do not create patent folder 'generated_traffic_data' again
	Path(load_folder_path).mkdir(parents=True, exist_ok=True)
	Path(actual_traffic_folder_path).mkdir(parents=True, exist_ok=True)
	Path(predicted_traffic_folder_path).mkdir(parents=True, exist_ok=True)


	for alpha in ALPHAS_LIST:
		predicted_traffic_alphafolder_path: str = load_folder_path + '/' + 'predicted_traffic' + '/' + 'alpha' + str(alpha)
		Path(predicted_traffic_alphafolder_path).mkdir(parents=True, exist_ok=True)
		
	generate_traffic_based_on_seattle(
	    from_date=FROM_DATE, 
	    to_date=TO_DATE, 
	    first_erlang=LAMBDA, 
	    constant_bitrate=IS_CONSTANT_BITRATE, 
	    aggregation=AGGREGATION, 
	    resampling=RESAMPLING,
	    actual_traffic_folder_path=actual_traffic_folder_path)

	# predict three periods
	columns = ["current_global_time", "source_id", "destination_id", "datarate", "arrival_time", "departure_time"]

	for i in range(60):
		filepath: str = actual_traffic_folder_path + '/' + 'actual_traffic' + '_' + str(UPGRADE_PERIOD * i) + '_' + str(UPGRADE_PERIOD * (i + 1)) + '.txt'
		print(i)

		traffic_df = pd.read_csv(filepath, delimiter='\s\s+', engine='python', header=None, names=columns)
		
	
		for alpha in ALPHAS_LIST:
			predicted_traffic_alphafolder_path: str = load_folder_path + '/' + 'predicted_traffic' + '/' + 'alpha' + str(alpha)
			perdict_traffic_for_next_period( traffic_df,alpha =alpha,predicted_traffic_alphafolder_path=predicted_traffic_alphafolder_path,index=i)

for lamda_0 in [100,150,200,250,300,350]:
	print(lamda_0)
	generate_for_load(lamda_0,[1,3,5,10])
