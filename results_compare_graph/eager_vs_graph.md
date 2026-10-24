| 방식 | batch | eager tok/s | graph tok/s | graph/eager | eager speedup | graph speedup |
|---|---|---|---|---|---|---|
| AR | 1 | 22.7 | 64.6 | 2.85 | 1.00 | 1.00 |
| AR | 8 | 178.5 | 377.6 | 2.12 | 1.00 | 1.00 |
| AR | 64 | 1225.3 | 1122.1 | 0.92 | 1.00 | 1.00 |
| AR | 128 | 1344.6 | 1209.2 | 0.90 | 1.00 | 1.00 |
| DFlash-b16 (full state) | 1 | 106.2 | 262.7 | 2.47 | 4.68 | 4.07 |
| DFlash-b16 (full state) | 8 | 791.6 | 900.1 | 1.14 | 4.43 | 2.38 |
| DFlash-b16 (full state) | 64 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (full state) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (mode-AL snapshot + recompute) | 1 | 100.2 | 263.9 | 2.63 | 4.42 | 4.09 |
| DFlash-b16 (mode-AL snapshot + recompute) | 8 | 756.1 | 1005.3 | 1.33 | 4.24 | 2.66 |
| DFlash-b16 (mode-AL snapshot + recompute) | 64 | 1154.4 | 1122.2 | 0.97 | 0.94 | 1.00 |
| DFlash-b16 (mode-AL snapshot + recompute) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (SpecLA, recompute) | 1 | 103.7 | 269.1 | 2.60 | 4.57 | 4.17 |
| DFlash-b16 (SpecLA, recompute) | 8 | 776.2 | 1047.4 | 1.35 | 4.35 | 2.77 |
| DFlash-b16 (SpecLA, recompute) | 64 | 1200.6 | 1158.6 | 0.97 | 0.98 | 1.03 |
| DFlash-b16 (SpecLA, recompute) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, top-2) | 1 | 100.1 | 262.2 | 2.62 | 4.41 | 4.06 |
| DFlash-b16 (conf-K, top-2) | 8 | 735.9 | 977.5 | 1.33 | 4.12 | 2.59 |
| DFlash-b16 (conf-K, top-2) | 64 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, top-2) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, top-4) | 1 | 101.0 | 261.4 | 2.59 | 4.45 | 4.05 |
| DFlash-b16 (conf-K, top-4) | 8 | 741.3 | 967.3 | 1.31 | 4.15 | 2.56 |
| DFlash-b16 (conf-K, top-4) | 64 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, top-4) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.3) | 1 | 100.4 | 263.4 | 2.62 | 4.43 | 4.08 |
| DFlash-b16 (conf-K, θ=0.3) | 8 | 744.2 | 988.4 | 1.33 | 4.17 | 2.62 |
| DFlash-b16 (conf-K, θ=0.3) | 64 | 1134.5 | OOM | OOM | 0.93 | OOM |
| DFlash-b16 (conf-K, θ=0.3) | 128 | OOM | OOM | OOM | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.7) | 1 | 99.5 | 263.0 | 2.64 | 4.39 | 4.07 |
| DFlash-b16 (conf-K, θ=0.7) | 8 | 738.3 | 982.7 | 1.33 | 4.14 | 2.60 |
| DFlash-b16 (conf-K, θ=0.7) | 64 | 1122.8 | OOM | OOM | 0.92 | OOM |
| DFlash-b16 (conf-K, θ=0.7) | 128 | OOM | OOM | OOM | OOM | OOM |
