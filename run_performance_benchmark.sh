python -m cProfile -o perf_data.pstats performance_tests.py
gprof2dot -f pstats perf_data.pstats | dot -Tpng -o perfgraph.png
