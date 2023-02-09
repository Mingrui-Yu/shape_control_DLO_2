# Global Model Learning for Large Deformation Control of Elastic Deformable Linear Objects: An Efficient and Adaptive Approach

The paper is accepted by IEEE Transactions on Robotics (IEEE T-RO).

[[arXiv](https://arxiv.org/abs/2205.04004)] [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9888782/)] [[Code](https://github.com/Mingrui-Yu/shape_control_DLO_2)]



## Video

<p align="center">
<iframe width="800" height="450" src="https://www.youtube.com/embed/Gh5ncipo2SA" title="Global Model Learning for Large Deformation Control of Elastic Deformable Linear Objects" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>




## Abstract

Robotic manipulation of deformable linear objects (DLOs) has broad application prospects in many fields. However, a key issue is to obtain the exact deformation models (i.e., how robot motion affects DLO deformation), which are hard to theoretically calculate and vary among different DLOs. Thus, shape control of DLOs is challenging, especially for large deformation control which requires global and more accurate models. In this paper, we propose a coupled offline and online data-driven method for efficiently learning a global deformation model, allowing for both accurate modeling through offline learning and further updating for new DLOs via online adaptation. Specifically, the model approximated by a neural network is first trained offline on random data, then seamlessly migrated to the online phase, and further updated online during actual manipulation. Several strategies are introduced to improve the model's efficiency and generalization ability. We propose a convex-optimization-based controller and analyze the system's stability using the Lyapunov method. Detailed simulations and real-world experiments demonstrate that our method can efficiently and precisely estimate the deformation model, and achieve large deformation control of untrained DLOs in 2D and 3D dual-arm manipulation tasks better than the existing methods. It accomplishes all 24 tasks with different desired shapes on different DLOs in the real world, using only simulation data for the offline learning.

## Citation

Please cite our paper if you find it helpful :)
```
@ARTICLE{yu2022global,
  author={Yu, Mingrui and Lv, Kangchen and Zhong, Hanzhong and Song, Shiji and Li, Xiang},
  journal={IEEE Transactions on Robotics}, 
  title={Global Model Learning for Large Deformation Control of Elastic Deformable Linear Objects: An Efficient and Adaptive Approach}, 
  year={2023},
  volume={39},
  number={1},
  pages={417-436},
  doi={10.1109/TRO.2022.3200546}}
```





