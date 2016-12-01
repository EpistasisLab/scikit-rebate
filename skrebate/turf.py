# Mon Sep 19 09:54:45 EDT 2016
import sys
import time as tm
import datetime as dt
###############################################################################
def runTurf(header, x, y, attr, var, pct, iterations, algorithm, options):
    from operator import itemgetter
    import Common as cmn
    import numpy as np

    lost = dict()
    start = tm.time()
    save_x = x
    V = options['verbose']
    if(V): print('Under TURF Control...')

    #--------------------------------------------------------------------------
    def adjust_variables(var, attr):
        c = d = 0
        for key in attr:
            if attr[key][0] == 'continuous':
                c += 1
            else:
                d += 1

        var['dpct'] = (float(d) / (d + c) * 100, d)
        var['cpct'] = (float(c) / (d + c) * 100, c)
    #--------------------------------------------------------------------------
    def create_newdata(header, x):
        dlist = []
        cnt = 0
        tmp = 0
        hlist = []

        if(V):
            print('Reducing attributes by ' + str(options['turfpct']) + '%')
            sys.stdout.flush()

        for a in table:
            if(cnt >= keepcnt):
                lost[a[0]] = iteration + 1
                hlist.append(a[0])     # append lost attribe names to hlist
                i = header.index(a[0])
                dlist.append(i)
            cnt += 1

        header = np.delete(header,dlist).tolist() #remove orphans from header
        x = np.delete(x,dlist,axis=1) #remove orphaned attributes from data
        #x = np.ascontiguousarray(x, dtype=np.double)

        if(V):
            print('Getting new variables, attributes and distance array')
            sys.stdout.flush()

        var = cmn.getVariables(header, x, y, options)
        attr = cmn.getAttributeInfo(header, x, var, options)

        if(V):
            print("---------------  Parameters  ---------------")
            print("datatype:   " + var['dataType'])
            print("attributes: " + str(var['NumAttributes']))

            if(var['dataType'] == 'mixed'):
                print("    continuous: " + str(var['cpct'][1]))
                print("    discrete:   " + str(var['dpct'][1]))
            if(var['mdcnt'] > 0):
                print("missing:    " + str(var['mdcnt']))
            print("--------------------------------------------")
            sys.stdout.flush()

        begin = tm.time()

        return header, x, attr, var, lost
    #--------------------------------------------------------------------------
    fullscores = dict()
    print("Total Iterations: " + str(iterations))
    for iteration in range(iterations):
        numattr = var['NumAttributes']
        if(V):
            print ("============================================")
            print ("Iteration:  " + str(iteration+1))
            print ("Attributes: " + str(numattr))
            sys.stdout.flush()

        table = []

        #Scores = fun(header,x,y,attr,var,distArray,options)
        if(algorithm == 'relieff'):
            import relieff
            R = relieff.ReliefF(pname=options['phenotypename'], 
                                missing=options['missingdata'], 
                                verbose=options['verbose'], 
                                n_neighbors=options['neighbors'], 
                                dlimit=options['discretelimit'], 
                                hdr=header)
            R.fit(x,y)
            Scores = R.Scores

        elif(algorithm == 'multisurf'):
            import multisurf
            M = multisurf.MultiSURF(pname=options['phenotypename'], 
                                    missing=options['missingdata'], 
                                    verbose=options['verbose'], 
                                    dlimit=options['discretelimit'], 
                                    hdr=header)
            M.fit(x,y)
            Scores = M.Scores

        elif(algorithm == 'surf'):
            import surf
            S = surf.SURF(pname=options['phenotypename'], 
                          missing=options['missingdata'], 
                          verbose=options['verbose'], 
                          dlimit=options['discretelimit'], 
                          hdr=header)
            S.fit(x,y)
            Scores = S.Scores
    
        elif(algorithm == 'surfstar'):
            import surfstar
            S = surfstar.SURFstar(pname=options['phenotypename'], 
                                  missing=options['missingdata'], 
                                  verbose=options['verbose'], 
                                  dlimit=options['discretelimit'], 
                                  hdr=header)
            S.fit(x,y)
            Scores = S.Scores

        if(V):
            print('Building scores table...')
            sys.stdout.flush()

        for j in range(var['NumAttributes']):
            table.append([header[j], Scores[j]])
            fullscores[header[j]] = (Scores[j])

        table = sorted(table,key=itemgetter(1), reverse=True)

        if(iteration + 1 < iterations):
            keepcnt = int(numattr - numattr * pct)
            header,x,attr,var,lost = create_newdata(header, x)

    if(V):
        print('Turf finished! Overall time: ' + str(tm.time() - start))
        sys.stdout.flush()
    return Scores,save_x,var,fullscores,lost,table
###############################################################################
