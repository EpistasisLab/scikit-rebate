def parse_output(output_filename):
    start_parsing = False
    filename = ''
    algorithm = ''
    scoring_time = ''
    variables_ranked = []
    variables_scores = []
    
    with open(output_filename) as in_file:
        for line in in_file:
            if 'iter' in line:
                algorithm += 'iter'
            elif 'vls' in line:
                algorithm += 'vls'
            elif 'turf' in line:
                algorithm += 'turf'
            elif 'Completed scoring' in line:
                    scoring_time = line.split('Completed scoring in')[1].split('seconds')[0].strip()
            elif line.count('\t') == 1:
                variables_ranked.append(line.split('\t')[0].strip())
                variables_scores.append(line.split('\t')[1].strip())
    return filename, algorithm, scoring_time, ','.join(variables_ranked), ','.join(variables_scores)



'''
if __name__=="__main__":
    import pandas as pd
    import numpy as np
    
    import sys
    import os

    
    output = sys.argv[1]    
#     myHome = os.environ.get('HOME')
#     outputPath = myHome +'/idata/output/'
#     dataPath = myHome +'/idata/datasets/'   
  
    main(output)
    '''