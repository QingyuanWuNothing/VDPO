# VDPO: Variational Delayed Policy Optimization

## guidelines
### 1. requirement
    conda create -n VDPO python=3.10
    conda activate VDPO
    pip install -r requirement.yaml
    pip install gymnasium[mujoco]
### 2. run the VDPO
    python3 VDPO.py --env=Ant-v4 --delay=5

## Citation
```
@article{wu2024variational,
  title={Variational Delayed Policy Optimization},
  author={Wu, Qingyuan and Zhan, Simon Sinong and Wang, Yixuan and Wang, Yuhui and Lin, Chung-Wei and Lv, Chen and Zhu, Qi and Huang, Chao},
  journal={arXiv preprint arXiv:2405.14226},
  year={2024}
}
```

## Acknowledgement
1. CleanRL: https://github.com/vwxyzjn/cleanrl
2. SAC: https://github.com/haarnoja/sac
3. AD-RL: https://github.com/QingyuanWuNothing/AD-RL