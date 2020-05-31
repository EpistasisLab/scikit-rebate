from skrebate.vls import VLS
from skrebate.iter import Iter
from skrebate.turf import TURF
from skrebate.relieff import ReliefF
from skrebate import SURF
import random
import pandas as pd

######CONTROL PANEL#####################################################################################################
test_VLS = False
test_TURF = False
test_TURFfit = False
test_Iterfit = True
test_VLSfit = False

#Test VLS Subset Method#################################################################################################
def test_subsets(num_indices,num_subsets,size_subsets):
    v = VLS(ReliefF(), num_subsets, size_subsets)
    s = v.make_subsets(list(range(num_indices)), num_subsets, size_subsets)
    return check_subsets(list(range(num_indices)),num_subsets,size_subsets,s)

def check_subsets(possible_indices,num_feature_subset,subset_size,subsets):
    #Check num subsets
    if len(subsets) != num_feature_subset:
        return 'num wrong'

    #Check len subsets
    for subset in subsets:
        if len(subset) != subset_size:
            return 'len wrong'

    #Check no repeated in each subset
    for subset in subsets:
        uniques = []
        for s in subset:
            if s in uniques:
                return 'unique wrong'
            uniques.append(s)

    #Check all values in subset
    for subset in subsets:
        for s in subset:
            if s in possible_indices:
                possible_indices.remove(s)
    if len(possible_indices) != 0:
        return 'incomplete wrong'

    return 'checks passed'

if test_VLS:
    print(test_subsets(200,40,5))
    print(test_subsets(200,5,40))
    print(test_subsets(100,40,5))
    print(test_subsets(100,200,5))
    print(test_subsets(100,200,5))
    for i in range(100):
        while True:
            num_features = random.randint(5,1000)
            size_subset = int(num_features*random.uniform(0,1))+1
            num_subsets = (int(num_features/size_subset)+1)*random.randint(1,100)
            if num_subsets*size_subset >= num_features and size_subset <= num_features:
                break
        print(test_subsets(num_features,num_subsets,size_subset))
    print('VLS Subset Creation Test Complete\n')

#Test TURF Features per Iteration#######################################################################################
def test_features_per_iteration(num_features,pct,num_scores_to_return):
    v = TURF(ReliefF(),pct,num_scores_to_return)
    f = v.get_features_per_iteration(num_features,pct,num_scores_to_return)
    print(f)

if test_TURF:
    test_features_per_iteration(10,4,3)
    test_features_per_iteration(10,1,10)
    test_features_per_iteration(10,0.9,10)
    test_features_per_iteration(10,0.9,2)
    test_features_per_iteration(10,0.2,3)
    test_features_per_iteration(10, 0.99, 3)
    test_features_per_iteration(10,2, 2)
    test_features_per_iteration(20, 0.8, 8)
    test_features_per_iteration(20, 0.8, 12)
    test_features_per_iteration(20, 0.5, 12)

#Test VLS Fit###########################################################################################################
if test_VLSfit:
    data = pd.read_csv('data/Multiplexer20Modified.csv', sep=',')
    data_features = data.drop('Class', axis=1).values
    data_phenotypes = data['Class'].values

    t = VLS(ReliefF(n_jobs=-1), num_feature_subset=4, size_feature_subset=6)
    t.fit(data_features, data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

    t = VLS(SURF(n_jobs=-1), num_feature_subset=4, size_feature_subset=6)
    t.fit(data_features, data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

#Test TURF Fit##########################################################################################################
if test_TURFfit:
    data = pd.read_csv('data/Multiplexer20Modified.csv', sep=',')
    data_features = data.drop('Class', axis=1).values
    data_phenotypes = data['Class'].values

    t = TURF(ReliefF(n_jobs=-1),pct=0.8,num_scores_to_return=8)
    t.fit(data_features,data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

    t = TURF(SURF(n_jobs=-1),pct=0.5,num_scores_to_return=12)
    t.fit(data_features,data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

    t = TURF(VLS(ReliefF(n_jobs=-1),num_feature_subset=4,size_feature_subset=6),pct=0.5,num_scores_to_return=12)
    t.fit(data_features,data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

#Test Iter Fit##########################################################################################################
if test_Iterfit:
    data = pd.read_csv('data/Multiplexer20Modified.csv', sep=',')
    data_features = data.drop('Class', axis=1).values
    data_phenotypes = data['Class'].values

    t = Iter(ReliefF(n_jobs=-1), max_iter=3)
    t.fit(data_features, data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

    t = Iter(SURF(n_jobs=-1), max_iter=3)
    t.fit(data_features, data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)

    t = Iter(VLS(ReliefF(n_jobs=-1), num_feature_subset=4, size_feature_subset=6), max_iter=2)
    t.fit(data_features, data_phenotypes)
    print(t.feature_importances_)
    print(t.top_features_)