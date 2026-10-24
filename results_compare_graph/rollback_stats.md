| 방식 | GPU | batch | slot/시퀀스 | hit율 (복사만) | S_0에서 재계산 비율 | 스텝당 재계산 토큰 |
|---|---|---|---|---|---|---|
| DFlash-b16 (conf-K, top-1) | 0 | 1 | 2 | 44.3% | 13.9% | 2.04 |
| DFlash-b16 (conf-K, top-1) | 0 | 8 | 2 | 44.0% | 13.9% | 2.03 |
| DFlash-b16 (conf-K, top-2) | 0 | 1 | 3 | 64.3% | 5.9% | 1.12 |
| DFlash-b16 (conf-K, top-2) | 0 | 8 | 3 | 64.1% | 5.9% | 1.13 |
| DFlash-b16 (conf-K, top-4) | 0 | 1 | 5 | 83.8% | 2.0% | 0.39 |
| DFlash-b16 (conf-K, top-4) | 0 | 8 | 5 | 83.9% | 1.8% | 0.39 |
| DFlash-b16 (conf-K, θ=0.1) | 0 | 1 | 2 | 8.1% | 91.6% | 4.98 |
| DFlash-b16 (conf-K, θ=0.1) | 0 | 8 | 2 | 8.5% | 91.4% | 4.95 |
| DFlash-b16 (conf-K, θ=0.3) | 0 | 1 | 2 | 20.1% | 75.0% | 4.05 |
| DFlash-b16 (conf-K, θ=0.3) | 0 | 8 | 2 | 20.6% | 73.9% | 3.98 |
| DFlash-b16 (conf-K, θ=0.5) | 0 | 1 | 2 | 36.1% | 42.8% | 2.73 |
| DFlash-b16 (conf-K, θ=0.5) | 0 | 8 | 2 | 36.5% | 42.5% | 2.70 |
| DFlash-b16 (conf-K, θ=0.7) | 0 | 1 | 2 | 42.9% | 15.3% | 2.04 |
| DFlash-b16 (conf-K, θ=0.7) | 0 | 8 | 2 | 42.9% | 15.4% | 2.05 |
| DFlash-b16 (conf-K, θ=0.9) | 0 | 1 | 2 | 36.2% | 3.9% | 2.47 |
| DFlash-b16 (conf-K, θ=0.9) | 0 | 8 | 2 | 36.3% | 3.8% | 2.46 |
| DFlash-b16 (conf-K B head, top-1) | 0 | 1 | 2 | 30.4% | 31.1% | 2.56 |
| DFlash-b16 (conf-K B head, top-1) | 0 | 8 | 2 | 31.1% | 30.5% | 2.53 |
| DFlash-b16 (conf-K B head, top-2) | 0 | 1 | 3 | 49.8% | 16.2% | 1.53 |
| DFlash-b16 (conf-K B head, top-2) | 0 | 8 | 3 | 49.9% | 16.0% | 1.52 |
| DFlash-b16 (conf-K B head, top-4) | 0 | 1 | 5 | 72.6% | 6.2% | 0.65 |
| DFlash-b16 (conf-K B head, top-4) | 0 | 8 | 5 | 73.0% | 5.9% | 0.64 |
| DFlash-b16 (conf-K B head, θ=0.1) | 0 | 1 | 2 | 8.2% | 91.5% | 4.97 |
| DFlash-b16 (conf-K B head, θ=0.1) | 0 | 8 | 2 | 8.8% | 90.8% | 4.91 |
| DFlash-b16 (conf-K B head, θ=0.3) | 0 | 1 | 2 | 15.7% | 78.7% | 4.18 |
| DFlash-b16 (conf-K B head, θ=0.3) | 0 | 8 | 2 | 16.6% | 77.9% | 4.13 |
| DFlash-b16 (conf-K B head, θ=0.5) | 0 | 1 | 2 | 25.0% | 57.9% | 3.24 |
| DFlash-b16 (conf-K B head, θ=0.5) | 0 | 8 | 2 | 25.5% | 57.3% | 3.21 |
| DFlash-b16 (conf-K B head, θ=0.7) | 0 | 1 | 2 | 30.2% | 34.4% | 2.60 |
| DFlash-b16 (conf-K B head, θ=0.7) | 0 | 8 | 2 | 30.4% | 33.7% | 2.57 |
| DFlash-b16 (conf-K B head, θ=0.9) | 0 | 1 | 2 | 29.6% | 9.4% | 2.69 |
| DFlash-b16 (conf-K B head, θ=0.9) | 0 | 8 | 2 | 29.5% | 9.2% | 2.69 |
| DFlash-b16 (oracle AL, 1 snapshot) | 0 | 1 | 1 | 100.0% | 0.0% | 0.00 |
| DFlash-b16 (oracle AL, 1 snapshot) | 0 | 8 | 1 | 100.0% | 0.0% | 0.00 |
| DFlash-b16 (oracle AL, 1 snapshot) | 0 | 64 | 1 | 100.0% | 0.0% | 0.00 |
| DFlash-b16 (mode-AL snapshot + recompute) | 0 | 1 | 1 | – | – | m=2 (스텝별 m 분포 {'5': 7, '3': 11, '2': 4884, '1': 333, '6': 6}) |
| DFlash-b16 (mode-AL snapshot + recompute) | 0 | 8 | 1 | – | – | m=2 (스텝별 m 분포 {'8': 1, '6': 3, '2': 641, '1': 23}) |
| DFlash-b16 (mode-AL snapshot + recompute) | 0 | 64 | 1 | – | – | m=2 (스텝별 m 분포 {'8': 1, '6': 2, '2': 93}) |
