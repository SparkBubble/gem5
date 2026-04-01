# ./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --num-cpus=4 --num-dirs=4 --network=garnet --topology=Mesh_XY --mesh-rows=2 --sim-cycles=10000000 --synthetic=uniform_random --injectionrate=0.04


/usr/bin/python3 util/sweep_garnet_injection.py \
    --gem5-bin ./build/NULL/gem5.opt \
    --config configs/example/garnet_synth_traffic.py \
    --extractor util/extract_garnet_stats.py \
    --num-cpus 16 \
    --num-dirs 16 \
    --network garnet \
    --topology Mesh_XY \
    --mesh-rows 4 \
    --sim-cycles 10000000 \
    --synthetic transpose \
    --min-rate 0 \
    --max-rate 1 \
    --coarse-step 0.01 \
    --max-fine-intervals 0 \
    --workdir m5out/sweep_AGEBASEDRR_0p01_transpose \
    --sa2-policy age-based-rr \
    --force-rerun
