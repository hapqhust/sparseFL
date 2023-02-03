# EXPERIMENT PLAN
### Baseline Model
- FedAVG
- FedProx
- FedFV
- FedFA
- Scafford

### Proposal
- Random matching
- Cluster matching

### Plan
> MNIST Dataset

| Data partition         | Number of clients   |
| ---------------------- | ------------------- |
| Cluster Sparse         | 100, 200, 500, 1000 |
| Sparse                 | 100, 200, 500, 1000 |
| 50% Sparse - 50% Dense | 100, 200            |
| 70% Sparse - 30% Dense | 100, 200            |
| 30% Sparse - 70% Dense | 100, 200            |

> Note: Run all algorithms in `baseline model` and `proposal` with the number of clients is `200` to compare these algorithms. With other num_clients only run algorithms in `proposal`.