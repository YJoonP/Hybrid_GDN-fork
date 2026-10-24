| 방식 | GPU | batch 1 | batch 8 | batch 64 | batch 128 |
|---|---|---|---|---|---|
| DFlash-b16 (full state) | 0 | 4.07× (263) | 2.38× (900) | OOM | OOM |
| DFlash-b16 (mode-AL snapshot + recompute) | 0 | 4.09× (264) | 2.66× (1005) | 1.00× (1122) | OOM |
| DFlash-b16 (SpecLA, recompute) | 0 | 4.17× (269) | 2.77× (1047) | 1.03× (1159) | OOM |
| DFlash-b16 (oracle AL, 1 snapshot) | 0 | 4.11× (266) | 2.68× (1013) | 1.00× (1122) | OOM |
| DFlash-b16 (conf-K, top-1) | 0 | 4.08× (264) | 2.60× (982) | OOM | OOM |
| DFlash-b16 (conf-K, top-2) | 0 | 4.06× (262) | 2.59× (978) | OOM | OOM |
| DFlash-b16 (conf-K, top-4) | 0 | 4.05× (261) | 2.56× (967) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.1) | 0 | 4.07× (263) | 2.62× (991) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.3) | 0 | 4.08× (263) | 2.62× (988) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.5) | 0 | 4.08× (263) | 2.61× (987) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.7) | 0 | 4.07× (263) | 2.60× (983) | OOM | OOM |
| DFlash-b16 (conf-K, θ=0.9) | 0 | 4.07× (263) | 2.59× (978) | OOM | OOM |
| DFlash-b16 (conf-K B head, top-1) | 0 | 4.12× (266) | 2.65× (999) | OOM | OOM |
| DFlash-b16 (conf-K B head, top-2) | 0 | 4.07× (263) | 2.63× (994) | OOM | OOM |
| DFlash-b16 (conf-K B head, top-4) | 0 | 4.06× (262) | 2.61× (985) | OOM | OOM |
| DFlash-b16 (conf-K B head, θ=0.1) | 0 | 4.08× (264) | 2.65× (1002) | OOM | OOM |
| DFlash-b16 (conf-K B head, θ=0.3) | 0 | 4.09× (264) | 2.66× (1003) | OOM | OOM |
| DFlash-b16 (conf-K B head, θ=0.5) | 0 | 4.08× (264) | 2.65× (1000) | OOM | OOM |
| DFlash-b16 (conf-K B head, θ=0.7) | 0 | 4.09× (264) | 2.65× (999) | OOM | OOM |
| DFlash-b16 (conf-K B head, θ=0.9) | 0 | 4.09× (264) | 2.63× (993) | OOM | OOM |
