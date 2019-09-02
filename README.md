# mlod.tf
Tensorflow code for [Missing Labels in Object Detection](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Weakly%20Supervised%20Learning%20for%20Real-World%20Computer%20Vision%20Applications/Xu_Missing_Labels_in_Object_Detection_CVPRW_2019_paper.pdf)

 - [Poster](http://frostsally.com/xu/asset/Frost2019MissingLabelPoster.pdf)

 - [ZhiHu](https://zhuanlan.zhihu.com/p/72859378)

**This repository is currently not complete. Please open an issue if you want any script related to the paper.**

This repository is based on the Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). To set up the environment, please follow the instruction in this [repo](https://github.com/endernewton/tf-faster-rcnn).

### update: download the Teacher Model prediction

In `./lib/datasets/pascal_voc.py`, pseudo boxes are loaded through the function "_load_annotation", and there should be a file with path: `./data/pseudo/`. Please use the following command to dowload the peuso label.
``` 
cd data/pseudo
wget https://reserach-project.s3.eu-west-2.amazonaws.com/mlod/W2F.mat 
cp W2F.mat voc_2007_trainval_gt.mat
```

After that, please refer the batch file `test_train.batch` to see the way to update the pseudo labels.


### Installation
1. Clone the repository
  ```Shell
  git clone https://https://github.com/Frostinassiky/mlod.tf.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```
  
### Dataset

To convert PASCAL VOC or COCO dataset to a missing label dataset, please go to this [repo](https://github.com/Frostinassiky/Bi-Level-Converter)

### Train/Test your own model
  
  Check the batch file.
  
### Citation

If you find this implementation helpful, please consider citing:

    @inproceedings{xu2019missing,
      title={Missing Labels in Object Detection},
      author={\textbf{Xu, Mengmeng} and Bai, Yancheng and Ghanem, Bernard},
      booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      year={2019}
    }
    
For convenience, here is the faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
