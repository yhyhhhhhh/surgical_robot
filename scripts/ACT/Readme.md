### 之前的版本
最初的版本的内窥镜位置和相机参数是
```python
self.scene.ecm.init_state.pos = (0.0, 0.11, 0.2)
self.scene.ecm.init_state.rot = (0.9659,-0.2588,0,0)
self.scene.camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ECM/ecm_end_link/camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.9, focus_distance=300.0, horizontal_aperture=30, clipping_range=(0.1, 2000.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0.0 ,0 ), convention="ros"),
)
```
采用action chunk size 为30，对应的模型为/media/yhy/PSSD/DVRK_DATA/orbit_surgical/state_machine/ckpt/Re_block_30_ckpt

之后调整了相机参数以及内窥镜位置，目的是加入深度图像，但是没有模型
```python
self.scene.ecm.init_state.pos = (0.0, 0.15, 0.1)
self.scene.ecm.init_state.rot = (0.866,-0.5,0,0)
Set Camera
self.scene.camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ECM/ecm_end_link/camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=20, focus_distance=300.0, horizontal_aperture=30, clipping_range=(0.1, 2000.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0, 0.0 ,0.707 ), convention="ros"),
)
```
### 3.12日改动
- 加入了对深度图像做处理，将两张图输入到模型中，对应的模型是
/media/yhy/PSSD/DVRK_DATA/orbit_surgical/state_machine/ckpt/depthcam_50
实验现象：
采用时间集成确实对运动的平滑性有一定的提升，但是为什么可以平滑运动这个原理还没搞清楚
当仅采用内窥镜图像的时候，从图像中判断不出前后关系的现象更加明显了，但是夹爪可以夹取
当仅采用深度图像的时候，从图像中判断前后关系，也就是夹爪可以更好地以正确的姿势放置到block上，但是夹爪不能夹取
同时采用时却两个缺点都有了，有时候会到block之后，有时候会放置到正确的位置，但是夹爪不能夹取，或者夹取的时机不对
下一步试试如果调换深度图像和内窥镜图像的输入顺序，看看效果，先输入深度图像，再输入内窥镜图像
还有问题是当query_frequency比较小的时候会出现往复运动的情况，这个问题不太清楚是什么原因
ACT算法中图像的输入究竟代表了什么
论文中还提到了high_frequency的pid控制，这个的效果还没探索
然后的问题是现在采用的控制是关节控制，他采用的关节底层控制这个还不知道，我们可不可以换成直接设定关节位置？