import sys
import os
import argparse
import csv
import pandas as pd
import random
import copy
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

'''
Sample Run Code:
python parseResults.py --output-path /Users/robert/Desktop/outputs --experiment-name rebate1

python parseResults.py --output-path D:\MyProfile\Desktop\outputs --experiment-name ReliefMultiset2
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist")

    #Save summary file
    with open(output_path + "/" + experiment_name + '/summary_results.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results = os.listdir(output_path + "/" + experiment_name + '/rawoutputs')
        writer.writerow(['Dataset','Algorithm','Runtime (s)','Ranked Features','Ranked Scores'])
        for result in results:
            full_path = output_path + "/" + experiment_name + '/rawoutputs/' + result
            with open(full_path) as file:
                ranked_features = []
                ranked_scores = []
                for line in file:
                    if 'SCORES' in line:
                        pass
                    elif 'Analysis' in line:
                        algorithm = line.split(' ')[0]
                    elif 'Run Time' in line:
                        runtime = float(line.split(' ')[-1])
                    else:
                        ranked_features.append(line.split('\t')[0])
                        ranked_scores.append(line.split('\t')[1])
                ranked_features = ','.join(map(str,ranked_features))
                ranked_scores = ','.join(map(str, ranked_scores))
            writer.writerow([result.split(algorithm+'_')[1],algorithm,runtime,ranked_features,ranked_scores])
    file.close()

    #Read summary file to create merged dictionary
    data = pd.read_csv(output_path + "/" + experiment_name + '/summary_results.csv').values
    merged_dict = {}
    for experiment in data:
        feature_data = experiment[3].split(',')
        if experiment[0][:-7] in merged_dict: #-7 removes .txt and dataset number, merging replicates under the same key
            if experiment[1] in merged_dict[experiment[0][:-7]]:
                merged_dict[experiment[0][:-7]][experiment[1]].append(feature_data)
            else:
                merged_dict[experiment[0][:-7]][experiment[1]] = [feature_data]
        else:
            merged_dict[experiment[0][:-7]] = {experiment[1]:[feature_data]}

    #Add random shuffle to merged dictionary
    for merged_dataset in merged_dict:
        algos = list(merged_dict[merged_dataset].keys())
        features = copy.deepcopy(merged_dict[merged_dataset][algos[0]][0]) #Arbitrary first list of ranked features

        shuffle_list = []
        num_replicates = len(merged_dict[merged_dataset][algos[0]])
        for i in range(num_replicates):
            random.shuffle(features)
            shuffle_list.append(copy.deepcopy(features))
        merged_dict[merged_dataset]['random_shuffle'] = shuffle_list

    #Convert ranked features into heatmap rows
    for merged_dataset in merged_dict:
        algos = list(merged_dict[merged_dataset].keys())
        num_noisy_features = None  # Expand scope of variable
        for algo in merged_dict[merged_dataset]:
            for i in range(len(merged_dict[merged_dataset][algo])):
                merged_dict[merged_dataset][algo][i],num_noisy_features = getPercentile(merged_dict[merged_dataset][algo][i])
            merged_dict[merged_dataset][algo] = getHeatmapRow(merged_dict[merged_dataset][algo],num_noisy_features)

        #Generate Heatmap
        algos.remove('random_shuffle')
        algos.sort()
        algos.insert(0,'random_shuffle')

       

        df_array = []
        for algo in algos:
            df_array.append(merged_dict[merged_dataset][algo])
        df_array = np.array(df_array)

        cols = []
        cols.append('Optimal')
        for c in range(1,num_noisy_features+1):
            cols.append(str(round(c/num_noisy_features*100))+'%')
        df = pd.DataFrame(df_array,columns=cols,index=algos)

        #Replaces with correct capitalizations
        new_index = {'multisurf':'MultiSURF','relieff_100nn':'ReliefF_100NN','multisurf_abs':'MultiSURF_abs','relieff_100nn_abs':'ReliefF_100NN_abs'}
        df.rename(index=new_index,inplace=True)

        custom_cmap = sb.color_palette('Oranges', n_colors=1000)[:800] + sb.color_palette('Blues', n_colors=1000)[800:]
        ax = sb.heatmap(data=df, vmin=0, vmax=1, cmap=custom_cmap)


        #This following code removes all the xtick labels and replaces it with regularly spaced ones
        #Known bug that this causes some spacing issues on datasets with very few features (only known source of error is 6-bit multiplexer)
        new_labels=['Optimal','10%','20%','30%','40%','50%','60%','70%','80%','90%','']
        old_ticks = ax.get_xticks()
        new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), len(new_labels))
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(new_labels,fontsize=8)
        ax.tick_params(axis='both', which='both', length=0)
        
        #add title here if want
        #plt.title('Title here')
        plt.xlabel('Predictive features in top % of ranked features')

        #rotate xtick and ytick so that it is horizontal
        plt.yticks(rotation='horizontal')
        plt.xticks(rotation='horizontal')
        
        plt.savefig(output_path + "/" + experiment_name + '/' + merged_dataset + '.pdf', bbox_inches = 'tight')
        plt.savefig(output_path + "/" + experiment_name + '/' + merged_dataset + '.png', bbox_inches='tight')
        plt.close('all')

def getPercentile(ranked_features):
    num_features = len(ranked_features)
    predictive_features = []
    final_predictive_index = 0
    for feature_index in range(len(ranked_features)):
        if ranked_features[feature_index][0] == 'M' or ranked_features[feature_index][0] == 'A':
            predictive_features.append(ranked_features[feature_index])
            final_predictive_index = feature_index

    ideal_final_predictive_index = len(predictive_features) - 1
    noisy_features_that_snuck_in = final_predictive_index - ideal_final_predictive_index
    num_noisy_features = num_features - len(predictive_features)

    return noisy_features_that_snuck_in/num_noisy_features,num_noisy_features

def getHeatmapRow(percentiles,num_noisy_features):
    heatmapRow = []
    for i in range(num_noisy_features+1):
        num_less_or_equal_to = 0
        for p in percentiles:
            if p <= i/num_noisy_features:
                num_less_or_equal_to += 1
        heatmapRow.append(num_less_or_equal_to/len(percentiles))
    return np.array(heatmapRow,dtype=float)

if __name__ == '__main__':
    sys.exit(main(sys.argv))