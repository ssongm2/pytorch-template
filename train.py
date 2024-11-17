import argparse #config 설정 코드에서 바꾸지 않고 cmd창에서 받기 위해 사용
import collections #다양한 데이터 타입 활용 가능, 여기서는 namedtuple 사용(일반 튜플과 달리 각 요소에 이름으로 접근 가능)
import torch
import numpy as np
import data_loader.data_loaders as module_data #data_loader 폴더의 data_loaders.py 파일
import model.loss as module_loss #model 폴더의 loss.py 파일
import model.metric as module_metric #model 폴더의 metric.py 파일
import model.model as module_arch #model 폴더의 model.py 파일
from parse_config import ConfigParser #config.json 설정 파일과 cls(cmd창에서 받아와서 옵션 수정)처리하는 클래스
from trainer import Trainer #학습 클래스
from utils import prepare_device #gpu 설정 위한 함수


#랜덤 시드 고정
SEED = 123
torch.manual_seed(SEED) #Pytorch 랜덤 시드 고정(난수 (ex. 가중치 초기화) 유지)
torch.backends.cudnn.deterministic = True #cuda에서 고정된 시드
torch.backends.cudnn.benchmark = False #CuDNN의 벤치마크 비활성화(활성화하면 속도 빠르지만, 난수 생성에 영향 줘서 재현성 저하)
np.random.seed(SEED) #Numpy 랜덤 시드 고정

def main(config):
    
    logger = config.get_logger('train') #학습 로깅 위한 logger 설정

    data_loader = config.init_obj('data_loader', module_data) #config 파일로 학습 data_loader 객체 설정
    valid_data_loader = data_loader.split_validation() #검증 data_loader도 설정

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch) #config 파일에서 정의된 모델 아키텍처 로드
    logger.info(model) #logger로 출력하여 모델 구조 확인

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu']) #멀티 GPU 설정
    model = model.to(device)
    if len(device_ids) > 1: #1개 이상이면, 
        model = torch.nn.DataParallel(model, device_ids=device_ids) #torch.nn.DataParallel로 병렬 처리하도록 설정

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']) #getattr로 config에서 설정한 손실함수 로드
    metrics = [getattr(module_metric, met) for met in config['metrics']] #getattr로 config에서 설정한 metrics 리스트 로드

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) #학습 가능한 파라미터만 선택
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) #옵티마이저에 전달 
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) #스케줄러 설정

    #학습과정 관리 클래스
    trainer = Trainer(model, criterion, metrics, optimizer, #criterion: 손실함수, config: 파라미터 설정파일, data_loader: 학습에 쓸 데이터로 미니배치 단위
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

#argparse로 cmd창에서 파라미터 수정 가능
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template') #객체 생성
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)') #config 파일 경로 지정
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)') #checkpoint 경로 지정
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)') #GPU 사용 가능 여부 지정

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
