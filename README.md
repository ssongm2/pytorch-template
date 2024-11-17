# PyTorch Template Project [Study]
PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* 딥러닝 프로젝트에 적용하기 위한 폴더 구조 정의
* CONFIG.json file -> 파라미터 조정의 편리성 위한 파일
* 더욱 편리한 파라미터 조정 위해 커스터마이징 가능한 command line options
* checkpoint 저장/다시 시작 기능
* 빠른 개발을 위한 추상 클래스 제공
  *BaseTrainer: 체크포인트 저장/다시 시작, 학습 과정 logging 등 수행
  *BaseDataLoader: 배치 생성, 데이터 셔플, 검증 데이터 분할 수행
  *BaseModel: 기본 모델 요약 제공

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - 학습 시작하는 코드
  ├── test.py - 모델 평가하는 코드
  │
  ├── config.json - 학습 설정 옵션(데이터 경로, 하이퍼파람리터 등)
  ├── parse_config.py - config.json 파일과 cli 옵션 처리하는 코드
  │
  ├── new_project.py - 새 프로젝트 초기화하는 코드
  │
  ├── base/ - 추상 클래스 (base)
  │   ├── base_data_loader.py - 1. 데이터 로더
  │   ├── base_model.py - 2. 모델
  │   └── base_trainer.py - 3. 학습기
  │
  ├── data_loader/ - 추상 클래스 (데이터 로드 관련)
  │   └── data_loaders.py - 1. 데이터 로드 정의
  │
  ├── data/ - 입력 데이터 저장 경로
  │
  ├── model/ - 추상 클래스 (모델)
  │   ├── model.py - 1. 모델 정의
  │   ├── metric.py - 2. 평가 지표 정의
  │   └── loss.py - 3. 손실 함수 정의
  │
  ├── saved/ - 추상 클래스 (저장)
  │   ├── models/ - 1. 학습된 모델
  │   └── log/ - 2. 학습 과정 담은 로그 경로 (tensorboard, logging output)
  │
  ├── trainer/ - 추상 클래스 (학습)
  │   └── trainer.py - 1. 학습기
  │
  ├── logger/ - 추상 클래스 (로그, tensorboard 시각화, logging)
  │   ├── visualization.py - 1. 시각화
  │   ├── logger.py - 2. 로그
  │   └── logger_config.json - 3. 로그 설정 파일
  │  
  └── utils/ - 추상 클래스 (유틸리티)
      ├── util.py - 1. 유틸리티 함수
      └── ...
  ```

## Usage
현재 repo는 MNIST 예시로 사용한 template
코드 실행은 'python train.py -c config.json' 명령어 입력


### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // 학습 세션 이름
  "n_gpu": 1,                   // 학습에 사용할 gpu num
  
  "arch": {
    "type": "MnistModel",       // 학습 모델 아키텍처 이름
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // 데이터 로더 선택
    "args":{
      "data_dir": "data/",             // 데이터셋 경로
      "batch_size": 64,                // 배치 사이즈
      "shuffle": true,                 // 학습 데이터 셔플(validation split 전)
      "validation_split": 0.1          // validation set 비율
      "num_workers": 2,                // 데이터 로딩에 사용할 cpu num(2개면 병렬)
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // 가중치 감쇠 (option) -> 가중치 지나치게 커지는 것 방지하는 정규화, 과적합 방지
      "amsgrad": true		       // Adam의 변형인 AMSGrade 활성화 -> 학습 안정성 높임
    }
  },
  "loss": "nll_loss",                  // loss -> negative log lkelihood loss: 주로 이진 분류에서 사용
  "metrics": [
    "accuracy", "top_k_acc"            // 평가할 metric 리스트
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate 스케줄러 -> StepLR: 일정 step size마다 lr 감소하는 방법, 가중치 업데이트 작게 하면 모델이 안정적으로 최적화 가능
    "args":{			       
      "step_size": 50,                 // 50마다 lr 감소
      "gamma": 0.1		       // lr 감소 비율 -> 0.1: 기존 lr의 10%로 줄임(0.1 -> 0.01 -> 0.001)
    }
  },
  "trainer": {
    "epochs": 100,                     // 학습 에폭 num
    "save_dir": "saved/",              // 체크포인트 저장 경로는 save_dir/models/name
    "save_freq": 1,                    // 체크포인트 저장 주기 -> 1: 1에폭마다 체크포인트 save
    "verbosity": 2,                    // 로그 출력 수준 -> 0: quiet(최종 결과만), 1: 에폭마다, 2: 상세 출력(배치단위 loss, lr 변화 등)
  
    "monitor": "min val_loss"          // 모델 성능 평가할 metric, 최적화 방향 지정 -> off: 비활성화, min val_loss: val_loss가 작아지는 방향으로 최적화
    "early_stop": 10	               // early stopping 대기 에폭 수 -> 0: 비활성화, 10: 10 에폭동안 개선 없으면 종료
  
    "tensorboard": true,               // tensorboard 시각화 유무
  }
}
```

Add addional configurations if you need.

### Using config files
config file에서 학습 설정 지정 후, 아래 명령어 수행

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
이전에 저장한 checkpoint에서 학습 재개하려면, 아래 명령어 수행

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
config file에서 n_gpu 값으로 multi-GPU 활성화 가능
- 사용 가능한 GPU보다 작은 수 설정하면, 첫번째부터 n개의 GPU 사용
- 특정 GPU 선택하려면, CUDA_VISIBLE_DEVICES 설정 사용

  ```
  python train.py --device 2,3 -c config.json //GPU 장치 순서의 인덱스 번호로 2번과 3번 사용
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support pytorch native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
