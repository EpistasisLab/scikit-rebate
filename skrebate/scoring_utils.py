import numpy as np

def get_row_missing(xc, xd, cdiffs, index, cindices, dindices):
    row = np.empty(0, dtype=np.double)
    cinst1 = xc[index]
    dinst1 = xd[index]
    can = cindices[index]
    dan = dindices[index]
    for j in range(index):
        dist = 0
        dinst2 = xd[j]
        cinst2 = xc[j]

        # continuous
        cbn = cindices[j]
        idx = np.unique(np.append(can, cbn))   # create unique list
        c1 = np.delete(cinst1, idx)       # remove elements by idx
        c2 = np.delete(cinst2, idx)
        cdf = np.delete(cdiffs, idx)

        # discrete
        dbn = dindices[j]
        idx = np.unique(np.append(dan, dbn))
        d1 = np.delete(dinst1, idx)
        d2 = np.delete(dinst2, idx)

        # discrete first
        dist += len(d1[d1 != d2])

        # now continuous
        dist += np.sum(np.absolute(np.subtract(c1, c2)) / cdf)

        row = np.append(row, dist)
    return row


def compute_score(attr, NN, feature, inst, nan_entries, headers, class_type, X, y, labels_std):
    """Evaluates feature scores according to the ReliefF algorithm"""

    fname = headers[feature]
    ftype = attr[fname][0]  # feature type
    ctype = class_type # class type
    diff_hit = diff_miss = 0.0
    count_hit = count_miss = 0.0
    mmdiff = 1
    diff = 0

    if nan_entries[inst][feature]:
        return 0.

    xinstfeature = X[inst][feature]

    #--------------------------------------------------------------------------
    if ctype == 'discrete':
        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:
                continue

            xNNifeature = X[NN[i]][feature]
            absvalue = abs(xinstfeature - xNNifeature) / mmdiff

            if y[inst] == y[NN[i]]:   # HIT
                count_hit += 1
                if xinstfeature != xNNifeature:
                    if ftype == 'continuous':
                        diff_hit -= absvalue
                    else: # discrete
                        diff_hit -= 1
            else: # MISS
                count_miss += 1
                if xinstfeature != xNNifeature:
                    if ftype == 'continuous':
                        diff_miss += absvalue
                    else: # discrete
                        diff_miss += 1

        hit_proportion = count_hit / float(len(NN))
        miss_proportion = count_miss / float(len(NN))
        diff = diff_hit * miss_proportion + diff_miss * hit_proportion
    #--------------------------------------------------------------------------
    else: # CONTINUOUS endpoint
        mmdiff = attr[fname][3]
        same_class_bound = labels_std

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:
                continue

            xNNifeature = X[NN[i]][feature]
            absvalue = abs(xinstfeature - xNNifeature) / mmdiff

            if abs(y[inst] - y[NN[i]]) < same_class_bound: # HIT
                count_hit += 1
                if xinstfeature != xNNifeature:
                    if ftype == 'continuous':
                        diff_hit -= absvalue
                    else: # discrete
                        diff_hit -= 1
            else: # MISS
                count_miss += 1
                if xinstfeature != xNNifeature:
                    if ftype == 'continuous':
                        diff_miss += absvalue
                    else: # discrete
                        diff_miss += 1

        hit_proportion = count_hit / float(len(NN))
        miss_proportion = count_miss / float(len(NN))
        diff = diff_hit * miss_proportion + diff_miss * hit_proportion

    return diff

def compute_scores(inst, attr, nan_entries, num_attributes, NN, headers, class_type, X, y, labels_std):
    scores = np.zeros(num_attributes)
    #NN = self._find_neighbors(inst)
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, NN, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std)
    return scores
