| 방식 | GPU | batch 1 | batch 8 | batch 64 | batch 128 |
|---|---|---|---|---|---|
| DFlash-b16 (full state) | 0 | 4.68× (106) | 4.43× (792) | OOM | OOM |
| DFlash-b16 (mode-AL snapshot + recompute) | 0 | 4.42× (100) | 4.24× (756) | 0.94× (1154) | OOM |
| DFlash-b16 (SpecLA, recompute) | 0 | 4.57× (104) | 4.35× (776) | 0.98× (1201) | OOM |
| DFlash-b16 (mode-AL snapshot + recompute) | 1 | 4.44× (103) | 4.14× (772) | 0.95× (1172) | OOM |
| DFlash-b16 (SpecLA, recompute) | 1 | 4.49× (104) | 4.15× (773) | 0.98× (1219) | OOM |
| DFlash-b16 (conf-K, top-1) | 1 | 4.38× (101) | 3.99× (744) | 0.92× (1135) | OOM |
| DFlash-b16 (conf-K, top-2) | 0 | 4.41× (100) | 4.12× (736) | OOM | OOM |
| DFlash-b16 (conf-K, top-4) | 0 | 4.45× (101) | 4.15× (741) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.1) | 1 | 4.38× (101) | 4.01× (747) | 0.94× (1165) | OOM |
| DFlash-b16 (conf-K, θ=0.3) | 0 | 4.43× (100) | 4.17× (744) | 0.93× (1134) | OOM |
| DFlash-b16 (conf-K, θ=0.5) | 1 | 4.39× (102) | 4.02× (749) | 0.93× (1145) | OOM |
| DFlash-b16 (conf-K, θ=0.7) | 0 | 4.39× (99) | 4.14× (738) | 0.92× (1123) | OOM |
| DFlash-b16 (conf-K, θ=0.9) | 1 | 4.39× (102) | 4.04× (753) | 0.91× (1126) | OOM |
