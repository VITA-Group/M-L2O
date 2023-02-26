# M-L2O: Towards Generalizable Learning-to-Optimize by Test-Time Fast Self-Adaptation

The official implementation of ICLR 2023 paper [M-L2O: Towards Generalizable Learning-to-Optimize by Test-Time Fast Self-Adaptation](https://openreview.net/forum?id=s7oOe6cNRT8).

## Abstract

Learning to Optimize (L2O) has drawn increasing attention as it often remarkably
accelerates the optimization procedure of complex tasks by overfitting specific
task types, leading to enhanced performance compared to analytical optimizers.
Generally, L2O develops a parameterized optimization method (i.e., optimizer)
by learning from solving sample problems. This data-driven procedure yields L2O
that can efficiently solve problems similar to those seen in training, that is, drawn
from the same "task distribution". However, such learned optimizers often struggle
when new test problems come with a substantial deviation from the training task
distribution. This paper investigates a potential solution to this open challenge,
by meta-training an L2O optimizer that can perform fast test-time self-adaptation
to an out-of-distribution task, in only a few steps. We theoretically characterize
the generalization of L2O, and further show that our proposed framework (termed
as M-L2O) provably facilitates rapid task adaptation by locating well-adapted
initial points for the optimizer weight. Empirical observations on several classic
tasks like LASSO, Quadratic and Rosenbrock demonstrate that M-L2O converges
significantly faster than vanilla L2O with only 5 steps of adaptation, echoing our
theoretical results.

## Usage

### Enviroment

```bash
conda env create -f environment.yml
```

### Meta-Training

#### Pretrained for Transfer Learning

```bash
cd Meta_Train
python train_rnnprop.py --save_path normal_pretrained --problem lasso_train_mix --if_cl=True
```

#### M-L2O

```bash
cd Meta_Train
python train_rnnprop.py --save_path ml2o_pretrained --if_cl=True
```

### (Meta-)Testing (Rosenbrock)

Firstly, run the following commands to generate a few fixed optimizees.

```bash
cd Meta_Train
for i in $(seq 1 10); do CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop.py --output_path optimizee_${i}_ros --seed ${i} --problem rosenbrock; done
```

#### Vanilla L2O

```bash
# Use optimizee_0_ros for example. 
cd Meta_Adapt
python train_rnnprop.py --save_path vanilla_rosenbrock --optimizee_path ../MAML_Train/optimizee_0_ros/initial_optimizee_params.pickle --num_epochs 10 --seed 1 --problem rosenbrock
cd ../Meta_Train
python evaluate_rnnprop.py --output_path vanilla_rosenbrock --optimizee_path optimizee_1_ros/initial_optimizee_params.pickle --path ../MAML_Adapt/vanilla_rosenbrock/rp.l2l-final --problem rosenbrock --seed 1 
```

#### Direct Transfer

```bash
cd Meta_Train
python evaluate_rnnprop.py --output_path direct_transfer_rosenbrock --optimizee_path optimizee_0_ros/initial_optimizee_params.pickle --path ../MAML_Adapt/normal_pretrained/rp.l2l-0 --problem rosenbrock --seed 1
```

#### Transfer Learning

```bash
cd MAML_Adapt/
python train_rnnprop.py --save_path transfer_learning_rosenbrock --optimizee_path ../MAML_Train/optimizee_0_ros/initial_optimizee_params.pickle --num_epochs 10 --seed 1 --optimizer_path normal_pretrained/rp.l2l-0 --problem rosenbrock 
cd ../MAML_Train/
python evaluate_rnnprop.py --output_path transfer_learning_rosenbrock --optimizee_path optimizee_0_ros/initial_optimizee_params.pickle --path ../MAML_Adapt/transfer_learning_rosenbrock/rp.l2l-0 --problem rosenbrock --seed 1
```

#### M-L2O

```bash
cd MAML_Adapt/
python train_rnnprop.py --save_path ml2o_rosenbrock --optimizee_path ../MAML_Train/optimizee_0_ros/initial_optimizee_params.pickle --num_epochs 10 --seed 1 --optimizer_path ml2o_pretrained/rp.l2l-0 --problem rosenbrock 
cd ../MAML_Train/
python evaluate_rnnprop.py --output_path ml2o_rosenbrock --optimizee_path optimizee_0_ros/initial_optimizee_params.pickle --path ../MAML_Adapt/ml2o_rosenbrock/rp.l2l-0 --problem rosenbrock --seed 1
```