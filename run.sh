./cp.sh

scons build/NULL/gem5.opt -j8

/usr/bin/python3 myutil/sweep_garnet_injection.py \
    --gem5-bin ./build/NULL/gem5.opt \
    --config configs/example/garnet_synth_traffic.py \
    --extractor myutil/extract_garnet_stats.py \
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
    --workdir m5out/sweep_HYBRIDRRAGE_0p01_transpose \
    --sa2-policy hybrid-rr-age \
    --force-rerun

/usr/bin/python3 myutil/sweep_garnet_injection.py \
    --gem5-bin ./build/NULL/gem5.opt \
    --config configs/example/garnet_synth_traffic.py \
    --extractor myutil/extract_garnet_stats.py \
    --num-cpus 16 \
    --num-dirs 16 \
    --network garnet \
    --topology Mesh_XY \
    --mesh-rows 4 \
    --sim-cycles 10000000 \
    --synthetic uniform_random \
    --min-rate 0 \
    --max-rate 1 \
    --coarse-step 0.01 \
    --max-fine-intervals 0 \
    --workdir m5out/sweep_HYBRIDRRAGE_0p01_uniform \
    --sa2-policy hybrid-rr-age \
    --force-rerun

/usr/bin/python3 myutil/sweep_garnet_injection.py \
    --gem5-bin ./build/NULL/gem5.opt \
    --config configs/example/garnet_synth_traffic.py \
    --extractor myutil/extract_garnet_stats.py \
    --num-cpus 16 \
    --num-dirs 16 \
    --network garnet \
    --topology Mesh_XY \
    --mesh-rows 4 \
    --sim-cycles 10000000 \
    --synthetic bit_complement \
    --min-rate 0 \
    --max-rate 1 \
    --coarse-step 0.01 \
    --max-fine-intervals 0 \
    --workdir m5out/sweep_HYBRIDRRAGE_0p01_complement \
    --sa2-policy hybrid-rr-age \
    --force-rerun
