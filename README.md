# Dynamic Graph Learning for Accurate Point-of-Interest Visit Forecasting

Arash Hajisafi, Haowen Lin, Sina Shaham, Haoji Hu, Maria Despoina Siampou, Yao-Yi Chiang, and Cyrus Shahabi.  
"Learning dynamic graphs from all contextual information for accurate point-of-interest visit forecasting."  
In *Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems*, pp. 1–12. 2023.  
[Link to paper](https://dl.acm.org/doi/abs/10.1145/3589132.3625567)

---

This repository hosts the implementation of our Busyness Graph Neural Network (BysGNN) framework for predicting visits to Points-of-Interest (POIs) by leveraging a comprehensive set of contextual signals, as introduced in the above paper. By capturing the complex interplay between spatial, temporal, semantic, and taxonomic contexts, BysGNN model significantly improves the accuracy of POI visit forecasting.

> **Important Note:**  
> The data used in this project is from the SafeGraph (private) dataset. If you have access to this dataset, you can use the BysGNN model as described in this repository by changing the arguments in `main.py` to point to the correct dataset directories.
> 
> For those who want to see an implementation of BysGNN on an open dataset, please check the implementation in the [NeuroGNN repository](https://github.com/USC-InfoLab/NeuroGNN/).

## Directory Structure

- `baselines`: Contains baseline models and examples.
- `busyness_graph`: Contains the main implementation of the Busyness Graph Neural Network (BysGNN).
- `busyness_graph_ablation`: Contains ablation study experiments.
- `busyness_graph_adj_interpretation`: Contains adjacency matrix interpretation experiments.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{hajisafi2023learning,
  title={Learning dynamic graphs from all contextual information for accurate point-of-interest visit forecasting},
  author={Hajisafi, Arash and Lin, Haowen and Shaham, Sina and Hu, Haoji and Siampou, Maria Despoina and Chiang, Yao-Yi and Shahabi, Cyrus},
  booktitle={Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems},
  pages={1--12},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
