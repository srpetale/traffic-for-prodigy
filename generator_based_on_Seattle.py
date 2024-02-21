import pandas as pd
import numpy as np
from traffic_generator import NetworkTrafficGenerator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
import math
from parameters import IS_CONSTANT_BITRATE, CONSTANT_BITRATE_GBPS, DATARATE_LIST_GBPS, DATARATE_SELECTION_PROBABILITIES, UPGRADE_PERIOD, LAMBDA_0, ALPHA_PERCENT

random.seed(2137)


def read_seattle_data(aggregation='avg'):
    """
    reads the files with Seattle data; 
    daily aggregation types: 'avg' and 'max'
    """
    if(aggregation == 'avg'):
        six_daily_traffic = pd.read_csv('agg_avg_daily.txt', sep='\t')
        six_daily_traffic = six_daily_traffic.rename(columns={'YYYY-MM-DD UTC': 'date', 'Average aggregate bits per second': 'bitrate'})
    elif(aggregation == 'max'):
        six_daily_traffic = pd.read_csv('agg_max_daily.txt', sep='\t')
        six_daily_traffic = six_daily_traffic.rename(columns={'YYYY-MM-DD UTC': 'date', 'Maximum aggregate bits per second': 'bitrate'})
    else:
        print('unknown aggregation')
        return pd.DataFrame([])
    
    six_daily_traffic = six_daily_traffic.drop(columns='Epoch Seconds')
    six_daily_traffic = six_daily_traffic.set_index('date')
    six_daily_traffic.index = pd.to_datetime(six_daily_traffic.index)
    six_daily_traffic = six_daily_traffic.asfreq('d')
    return six_daily_traffic


def seattle_percentage_from_date(aggregation='avg', from_date=0, to_date=0, resampling='no', first_erlang=100):
    """
    draws Seatlle traffic as a percentage of traffic from specified date formatted as 'yyyy-mm-dd' (or from the beginning if 0);
    daily aggregation types: 'avg' and 'max'
    """
    seattle_avg = read_seattle_data(aggregation)
    if(resampling == 'weekly'):
        seattle_avg = seattle_avg.resample('7d').mean()
    if(resampling == 'monthly'):
        seattle_avg = seattle_avg.resample('M').mean()    
    if(to_date == 0):
        to_date = len(seattle_avg)
    seattle_list = seattle_avg[from_date:to_date]['bitrate']
    first_value = seattle_list.iloc[0]
    percentages = (seattle_list / first_value) * first_erlang
    return percentages  


def generate_traffic_based_on_seattle(from_date=0, to_date=0, aggregation = 'max', resampling='weekly', number_of_nodes=14, mean_holding_time=1.0, constant_bitrate=IS_CONSTANT_BITRATE, first_erlang=LAMBDA_0, period_length=UPGRADE_PERIOD,actual_traffic_folder_path=''):
    source_ids = []
    destination_ids = []
    datarates = []
    arrival_times = []
    departure_times = []
    loads_in_erlang = seattle_percentage_from_date(aggregation=aggregation, from_date=from_date, to_date=to_date, resampling=resampling, first_erlang=first_erlang)

    if(resampling == 'weekly'):
        number_of_connections: int = loads_in_erlang*7
    elif(resampling == 'monthly'):
        number_of_connections: int = loads_in_erlang*30    
    else:
        print('unknown resampling')    
        return pd.DataFrame([])
    
    if IS_CONSTANT_BITRATE:
        list_of_datarates=[CONSTANT_BITRATE_GBPS]
    else:
        list_of_datarates=DATARATE_LIST_GBPS
    
    last_arrival = 0.0

    period = 1
    for load in range(len(loads_in_erlang)):
        traffic_generator = NetworkTrafficGenerator(number_of_nodes=number_of_nodes,
        list_of_datarates=list_of_datarates,
        lambda_in_poisson_distribution=loads_in_erlang.iloc[load],
        mean_holding_time=mean_holding_time, 
        current_time=last_arrival)

        connection_id = 0
        while connection_id < number_of_connections.iloc[load]:
            source, destination, datarate, arrival_time, holding_time = traffic_generator.get_connection()

            source_ids.append(source)
            destination_ids.append(destination)
            datarates.append(datarate)
            arrival_times.append(arrival_time)
            departure_times.append(arrival_time+holding_time)
            last_arrival = arrival_time
            connection_id += 1

            if(last_arrival >= period*period_length):
                generated_traffic = pd.DataFrame({'current_global_time':arrival_times, 'source_id':source_ids,'destination_id':destination_ids,'datarate':datarates,'arrival_time':arrival_times,'departure_time':departure_times})
                filepath: str = actual_traffic_folder_path + '/' + 'actual_traffic' + '_' + str(UPGRADE_PERIOD * (period-1)) + '_' + str(UPGRADE_PERIOD * period) + '.csv'

                #export DataFrame to text file
                with open(filepath, 'w') as f:
                    df_string = generated_traffic.to_string(header=False, index=True)
                    f.write(df_string)
                source_ids = [] 
                destination_ids = [] 
                datarates = [] 
                arrival_times = [] 
                departure_times = [] 
                generated_traffic = []
                period += 1

    
    # generated_traffic = pd.DataFrame({'current_global_time':arrival_times, 'source_id':source_ids,'destination_id':destination_ids,'datarate':datarates,'arrival_time':arrival_times,'departure_time':departure_times})
    # return generated_traffic


def divide_generated_traffic_into_periods(generated_traffic, period_length=90):
    """
    period length is in days
    """
    periods = []
    for i in range(0,math.ceil(max(generated_traffic['current_global_time'])),period_length):
        periods.append(generated_traffic.loc[(generated_traffic['current_global_time'] >= i) & (generated_traffic['current_global_time'] <= i+period_length)])

    return periods


def perdict_traffic_for_next_period(traffic_from_previous_period, alpha=ALPHA_PERCENT, constant_bitrate=True, number_of_nodes=15, period_length = 90,predicted_traffic_alphafolder_path='',index=0):
    def divide_into_windows(series, window_size):
        windows = []
        targets = []
        length = len(series)
            
        for i in range(length - window_size):
            window = series[i:i+window_size]
            target = series[i+window_size]
            windows.append(window)
            targets.append(target)
            
        return windows, targets
    
    increase_ratio = 1 + (alpha/100)

    period_start = traffic_from_previous_period['current_global_time'].min()
    first_week_count = len(traffic_from_previous_period['current_global_time'].loc[traffic_from_previous_period['current_global_time'] < period_start + 7])
    last_week_count = len(traffic_from_previous_period['current_global_time'].loc[(traffic_from_previous_period['current_global_time'] > (period_start + period_length - 7)) & (traffic_from_previous_period['current_global_time'] < (period_start + period_length))])
    add_connections = math.ceil(len(traffic_from_previous_period)*(abs(first_week_count-last_week_count)/first_week_count)*0.05)

    current_traffic = traffic_from_previous_period
    train_start = math.floor(min(traffic_from_previous_period['current_global_time']))
    train_end = train_start+period_length
    
    sequences = []
    sources = []
    destinations = []
    for source in range(number_of_nodes):
        for destination in range(number_of_nodes):
            if(source != destination):
                sources.append(source)
                destinations.append(destination)
                sequences.append(current_traffic[(current_traffic['source_id']==source) & (current_traffic['destination_id']==destination)].reset_index(drop=True))

    dfs = []

    for node_pair in range(len(sequences)):
        series = sequences[node_pair]['arrival_time']
        window_size = 3
        windows, targets = divide_into_windows(series, window_size)



        X = np.array(windows)
        y = np.array(targets)
        
        if(len(targets) > window_size):
            model = LinearRegression()
            model.fit(X, y)

            series = sequences[node_pair]['arrival_time']
            num_predictions = len(series) - window_size
            next_windows = []
            newest_prediction = 0
            i=0
            while newest_prediction < train_end+period_length:
                newest_prediction = model.predict(X[-1:])
                if(newest_prediction < train_end):
                    newest_prediction = train_end + abs(np.random.normal(0, 1))
                next_window = np.append(X[-1, 1:], newest_prediction)
                next_windows.append(next_window)
                X = np.vstack([X, next_window])
                if(i > int(num_predictions/100*increase_ratio)):
                    break
                i += 1

            predictions = []
            for window, target in zip(next_windows, model.predict(next_windows)):
                if(target < train_end):
                    target = train_end + abs(np.random.normal(0, 1))
                predictions.append(target)

            mean = 0  
            std_dev = 1  
            predictions_with_noise = [prediction + abs(np.random.normal(mean, std_dev)) for prediction in predictions]

            num_samples = 0
            
            if(len(predictions_with_noise) < num_predictions + add_connections):
                num_samples = num_predictions + add_connections - len(predictions_with_noise)
            
            if(num_samples < math.ceil(num_samples*increase_ratio)):
                num_samples += math.ceil(num_samples*increase_ratio) - len(predictions_with_noise)

            new_elements = []
            for _ in range(num_samples):
                new_element = random.uniform(train_end,train_end+period_length) # this creates additional connections nicely spread throughout the whole period
                # new_element = random.uniform(train_end,train_end+7) # creates new connections within the first week to create a momentary spike
                new_elements.append(new_element)

            predictions_with_noise.extend(new_elements)
            predictions_with_noise.sort()

            new_df = pd.DataFrame(predictions_with_noise)
            new_df.columns = ['current_global_time']
            new_df['source_id'] = sources[node_pair]
            new_df['destination_id'] = destinations[node_pair]
            if(constant_bitrate):
                new_df['datarate'] = CONSTANT_BITRATE_GBPS
            else:
                new_df['datarate'] = np.array([np.random.choice(DATARATE_LIST_GBPS, 1, p=DATARATE_SELECTION_PROBABILITIES) for _ in range(len(new_df))]).flatten()    
            new_df['arrival_time'] = new_df['current_global_time']
            new_df['departure_time'] = new_df['arrival_time']+random.expovariate(0.75)      
            dfs.append(new_df)      

    final_ml = pd.concat(dfs)
    final_ml = final_ml.sort_values(by=['current_global_time']).reset_index(drop=True)
    final_ml = final_ml.drop(final_ml[final_ml.current_global_time > train_end+period_length].index)        
    filepath: str = predicted_traffic_alphafolder_path + '/' + 'predicted_traffic' + '_' + str(index + UPGRADE_PERIOD) + '_' + str(index + UPGRADE_PERIOD*2) + '.csv'

                #export DataFrame to text file
    with open(filepath, 'w') as f:
        df_string = final_ml.to_string(header=False, index=True)
        f.write(df_string) 

def perdict_traffic(generated_traffic, alpha=1, period_length=90, constant_bitrate=True, number_of_nodes=15):
    
    def divide_into_windows(series, window_size):
        windows = []
        targets = []
        length = len(series)
            
        for i in range(length - window_size):
            window = series[i:i+window_size]
            target = series[i+window_size]
            windows.append(window)
            targets.append(target)
            
        return windows, targets
    
    predictions_for_periods = []
    increase_ratio = 1 + (alpha/100)
    
    for train_start in range(0,math.ceil(max(generated_traffic['current_global_time'])),period_length):
        train_end = train_start+period_length
        current_traffic = generated_traffic.copy()
        current_traffic = current_traffic.loc[(current_traffic['current_global_time'] >= train_start) & (current_traffic['current_global_time'] <= train_end)]

        current_traffic['current_global_time'] = current_traffic['current_global_time']-train_start
        current_traffic['arrival_time'] = current_traffic['arrival_time']-train_start
        current_traffic['departure_time'] = current_traffic['departure_time']-train_start

        sequences = []
        sources = []
        destinations = []
        for source in range(number_of_nodes):
            for destination in range(number_of_nodes):
                if(source != destination):
                    sources.append(source)
                    destinations.append(destination)
                    sequences.append(current_traffic[(current_traffic['source_id']==source) & (current_traffic['destination_id']==destination)].reset_index(drop=True))

        dfs = []

        for node_pair in range(len(sequences)):
            series = sequences[node_pair]['arrival_time']
            window_size = 3
            windows, targets = divide_into_windows(series, window_size)

            X = np.array(windows)
            y = np.array(targets)
            
            if(len(targets) > window_size):
                model = LinearRegression()
                model.fit(X, y)

                series = sequences[node_pair]['arrival_time']
                num_predictions = len(series) - window_size
                next_windows = []
                newest_prediction = 0
                i=0
                while newest_prediction < train_end+period_length:
                    newest_prediction = model.predict(X[-1:])
                    if(newest_prediction < train_end):
                        newest_prediction = train_end + abs(np.random.normal(0, 1))
                    next_window = np.append(X[-1, 1:], newest_prediction)
                    next_windows.append(next_window)
                    X = np.vstack([X, next_window])
                    if(i > int(num_predictions/100*increase_ratio)):
                        break
                    i += 1

                predictions = []
                for window, target in zip(next_windows, model.predict(next_windows)):
                    if(target < train_end):
                        target = train_end + abs(np.random.normal(0, 1))
                    predictions.append(target)

                mean = 0  
                std_dev = 1  
                predictions_with_noise = [prediction + abs(np.random.normal(mean, std_dev)) for prediction in predictions]

                num_samples = 0
                
                if(len(predictions_with_noise) < math.ceil(num_predictions*increase_ratio)):
                    num_samples = math.ceil(num_predictions*increase_ratio) - len(predictions_with_noise)

                new_elements = []
                for _ in range(num_samples):
                    new_element = random.uniform(train_end,train_end+period_length) # this creates additional connections nicely spread throughout the whole period
                    # new_element = random.uniform(train_end,train_end+7) # creates new connections within the first week to create a momentary spike
                    new_elements.append(new_element)

                predictions_with_noise.extend(new_elements)
                predictions_with_noise.sort()

                new_df = pd.DataFrame(predictions_with_noise)
                new_df.columns = ['current_global_time']
                new_df['source_id'] = sources[node_pair]
                new_df['destination_id'] = destinations[node_pair]
                if(constant_bitrate):
                    new_df['datarate'] = CONSTANT_BITRATE_GBPS
                else:
                    new_df['datarate'] = np.array([np.random.choice(DATARATE_LIST_GBPS, 1, p=DATARATE_SELECTION_PROBABILITIES) for _ in range(len(new_df))]).flatten()    
                new_df['arrival_time'] = new_df['current_global_time']
                new_df['departure_time'] = new_df['arrival_time']+random.expovariate(0.75)
                dfs.append(new_df)

        final_ml = pd.concat(dfs)
        final_ml = final_ml.sort_values(by=['current_global_time']).reset_index(drop=True)
        final_ml = final_ml.drop(final_ml[final_ml.current_global_time > train_end+period_length].index)  
        predictions_for_periods.append(final_ml)              
    return predictions_for_periods


def draw_generated_traffic(df_to_draw, figsize=(15, 5)):
    if(df_to_draw.empty):
        print('nothing to plot')
    else:    
        test = df_to_draw[['datarate','arrival_time','departure_time']]
        values = list(test.itertuples(index=False, name=None))

        sorted_values = sorted(values, key=lambda x: (x[1], x[2]))
        time_points = []
        sums = []
        values = []
        current_sum = 0
        for value, start_time, end_time in sorted_values:
            time_points.append(start_time)
            values.append(value)
            time_points.append(end_time)
            values.append(-value)

        time_points, values = zip(*sorted(zip(time_points, values)))

        for i in range(len(time_points)):
            current_sum +=values[i]
            sums.append(current_sum)

        plt.figure(figsize=figsize)
        plt.plot(time_points, sums, ds='steps-post')
        plt.xlabel('day')
        plt.ylabel('bitrate')
        plt.title('generated traffic over time')
        plt.tight_layout()
        plt.show()


def draw_erlangs_from_Seattle(aggregation='avg', from_date=0, to_date=0, first_erlang=100, ylog=False, resampling='no', figsize=(15,5), savefig=False):
    """
    draws Seatlle traffic as a percentage of traffic from specified date formatted as 'yyyy-mm-dd' (or from the beginning if 0);
    daily aggregation types: 'avg' and 'max'
    """
    seattle_avg = read_seattle_data(aggregation)
    if(resampling == 'weekly'):
        seattle_avg = seattle_avg.resample('7d').mean()
    if(resampling == 'monthly'):
        seattle_avg = seattle_avg.resample('M').mean()    
    plt.figure(figsize=figsize)
    if(to_date == 0):
        to_date = len(seattle_avg)
    seattle_list = seattle_avg[from_date:to_date]['bitrate']
    first_value = seattle_list.iloc[0]
    percentages = (seattle_list / first_value) * first_erlang
    plt.plot(percentages)
    plt.title('Erlangs based on Seattle traffic (daily {}, {} resampling) relative to {}'.format(aggregation, resampling, from_date))
    plt.ylabel('erlang')
    plt.xlabel('date')
    plt.ylim(bottom=0)
    if(ylog):
        plt.yscale('log')
    plt.tight_layout()
    if(savefig):
        plt.savefig('seattle_percentage_from_date_{}_{}.jpeg'.format(from_date,aggregation), dpi=600)
    plt.show() 
