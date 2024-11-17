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
코드 실행은 `python train.py -c config.json` 명령어 입력


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
template file로 새로운 프로젝트 디렉토리 만드려면, `new_project.py` 사용, 아래 명령어 수행
 
 ```
'python new_project.py ../NewProject' 
 ```
실행하면, NewProject라는 이름의 새 프로젝트 폴더 생성
불필요한 파일(캐시, git files, readme file)은 제외하고 template만 복사

### Custom CLI options
- config.json 파일을 수정하는 것이 일반적으로 안전하고 쉽지만, 자주 바뀌는 값은 command line에서 수정하는 것이 편리
- 현재 template은 JSON을 디폴트로 사용하지만, CLI flags에서 custom option 설정해서 사용자가 원하는 것으로 바꿀 수 있음 (CLI flags: 터미널에서 명령어 옵션으로 값 전달하는 방법)

  ```python
  # CustomArgs 객체를 flags, type, target 속성으로 정의
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # 여기 추가된 옵션은 CLI flags에서 수정 가능
  ]
  ```

- `target`은 터미널에서 전달된 CLI flag를 덮어쓰기 위해 config dict의 특정 값에 접근할때 사용하는 key
- lr option은 ('optimizer', 'args', 'lr'), 그 이유는 `config['optimizer']['args']['lr']`가 learning rate를 가리키기 때문

실행 예시: config에서 주어진 설정 기반으로 학습하지만, batch size만 수정해서 256으로 변경
```
train.py -c config.json --bs 256
```

### Data Loader
* **Writing your own data loader**

1. **상속 `BaseDataLoader`** 

    `BaseDataLoader`는 torch.utils.data.DataLoader를 상속받은 클래스

    `BaseDataLoader`가 처리하는 것:
    * 다음 batch 생성
    * 데이터 shuffling
    * validation daata loader 생성, 아래 함수 호출
      ```python
      'BaseDataLoader.split_validation()'
      ```

* **DataLoader Usage**

  `BaseDataLoader`는 iterator이므로, 배치 돌려면 아래처럼 사용:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  MNIST data의 로딩 예시는 `data_loader/data_loaders.py`

### Trainer
* **Writing your own trainer**

1. **상속 `BaseTrainer`**

    `BaseTrainer`가 처리하는 것:
    * 학습 과정 logging
    * Checkpoint 저장
    * Checkpoint 재개
    * 현재 가장 좋은 성능 저장 위해 monitoring, early stopping 지원
      * config의 `monitor`가 `max val_accuracy`면, validation accuracy가 최고 성능으로 업데이트 될때마다 **model_best.pth**로 checkpoint 저장
      * config의 `early_stop`이 설정되어 있으면, 지정된 에폭 동안 성능 개선 없을 때 학습 자동 종료, 비활성화하려면 `early_stop`을 `0`으로 지정하거나, config에서 해당 옵션 삭제

3. **추상 methods 구현**

    `trainer/train.py`에서 구현
    (필수)학습 과정 정의 `_train_epoch()`
    (옵션)검증 과정 정의 `_valid_epoch()`

* **Example**

  MNIST training 예시는 `trainer/trainer.py`

* **Iteration-based training**

  `Trainer.__init__`은 옵션으로 `len_epoch` 받을 수 있음
  - `len_epoch`: 각 에폭에서 처리할 배치(step) 개수 제어, 이를 통해 에폭 길이 고정하거나 조정

### Model
* **Writing your own model**

1. **상속 `BaseModel`**

    `BaseModel`이 처리하는 것:
    * `torch.nn.Module`을 상속받은 클래스
    * `__str__`: 모델 출력 시, 학습 가능한 파라미터 개수 표시하도록 'print'함수 동작 수정.

2. **추상 method 구현**

    순전파 과정은 `forward()` 메서드 구현

* **Example**

  LeNet 예시는 `model/model.py`

### Loss
손실 함수는 `model/loss.py`에서 구현 가능, config file의 `loss` 항목에 해당 손실 함수 이름 지정하여 사용 가능

### Metrics
Metric 함수는 `model/metric.py`에 위치

multiple metric 모니터링 하려면, config file에서 아래처럼 수정: 
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
학습 중에 추가 로그 정보 원하면, _train_epoch()에서 로깅 데이터를 `log` 딕셔너리에 병합, 아래 참고:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s} #추가 로그 정보
  log.update(additional_log)
  return log
  ```

### Testing
`test.py`로 실행해서 테스트 가능, `--resume` 옵션 사용해 지정된 checkpoint 경로를 전달

### Validation data
data loader에서 validation data 분할하려면, 
- `BaseDataLoader.split_validation()` 호출
- 그럼, config file에서 지정한 validation size에 따라 validation data loader 반환
- `validation_split` 옵션: ratio (0.0 ~ 1.0), sample num (0 ~ total num)

**Note**: the `split_validation()`은 원본 데이터 로더를 수정함

**Note**: config file에서 `"validation_split"`이 '0'으로 설정하면, split_validation은 None 반환

### Checkpoints
config files에서 학습 세션 이름 지정 가능:
  ```json
  "name": "MNIST_LeNet",
  ```

- checkpoints는 `save_dir/name/timestamp/checkpoint_epoch_n`에 저장, timestamp: 저장 시각(mmdd_HHMMSS -> 월,일,시,분,초)
- config file의 copy도 같은 폴더에 저장

**Note**: checkpoints 포함 정보:
  ```python
  {
    'arch': arch, #모델 아키텍처
    'epoch': epoch, #에폭 num
    'state_dict': self.model.state_dict(), #모델 가중치
    'optimizer': self.optimizer.state_dict(), #optimizer 상태
    'monitor_best': self.mnt_best, #현재 최고 monitoring 값
    'config': self.config #config file 설정 정보
  }
  ```

### Tensorboard Visualization
Tensorboard 시각화 지원 (`torch.utils.tensorboard` 또는 [TensorboardX](https://github.com/lanpa/tensorboardX) 사용)

1. **Install**

    - pytorch 1.1 이상이면, tensorboard 설치 by `pip install tensorboard>=1.14.0`.
    - pytorch 1.1 미만이면, 가이드 따라 설치 [TensorboardX](https://github.com/lanpa/tensorboardX).

3. **Run training** 

    config file에서 tensorboard 옵션 활성화

    ```
     "tensorboard" : true
    ```

4. **Open Tensorboard server** 

    1. project의 root 디렉토리에서 다음 명령어 실행:
   	`tensorboard --logdir saved/log/`
    2. 서버 열리면, 브라우저에서 `http://localhost:6006` 접속해서 tensorboard 확인
       
- 디폴트로 config file에 지정된 loss, metrics, input images, parameter의 히스토그램이 tensorboard에 기록
- 추가 시각화 원하면, 'trainer._train_epoch'에서 다음과 같은 method 활용 가능
	- `add_scalar('tag', data)`: 스칼라 값 기록
 	- `add_image('tag', image)`: 이미지 데이터 기록
- add_something()은 template에서 제공, 이는 `tensorboardX.SummaryWriter`과 `torch.utils.tensorboard.SummaryWriter`모듈의 wrapper로 작동.

**Note**: 현재 step을 지정할 필요 없음, `logger/visualization.py`에 정의된 'WriterTensorboard' 클래스가 step 자동으로 지정

## Contribution
- 함수 추가 및 활용 자유롭게 가능하며 PEP8 coding style 따름
- Code는 commit하기 전에 [Flake8](http://flake8.pycqa.org/en/latest/) 검사 패스해야 함

## TODOs

- [ ] Multiple optimizers (다중 옵티마이저)
- [ ] Support more tensorboard functions
- [x] Using fixed random seed (고정된 랜덤 시드 사용)
- [x] Support pytorch native tensorboard (tnative tensorboard 지원)
- [x] `tensorboardX` logger support 
- [x] Configurable logging layout, checkpoint naming 
- [x] Iteration-based training (instead of epoch-based) (epoch기반 대신 iteration 기반 학습)
  - epoch: 전체 데이터셋을 n바퀴
  	- (예시) epoch 10이다 = 모델이 전체 데이터셋을 10번 학습한다
  	- (참고) 데이터셋 크기 100, 배치 크기 10이면 1 epoch당 10번 반복(=epoch당 정해지는 반복횟수 step)해서 전체 데이터셋을 1바퀴 돈다.
 	 - 이때 epoch이 10이면 이 과정을 10번 하는 것이다.
  - iter: 단순히 n번 반복, 데이터셋 크기와 관계 없음
  	- (예시) iter 10이다 = 모델이 10번 반복 학습한다
  	- (참고) 데이터셋 크기 작으면 그냥 다시 섞어서 계속 학습, 데이터셋 크기 크면 전체 데이터 다 학습 안해도 종료
- [x] Adding command line option for fine-tuning (fine-tunng 위한 cmd line 옵션 추가)

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
