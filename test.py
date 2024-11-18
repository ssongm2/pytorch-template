import argparse #학습에 필요한 세팅값 코드 내부 대신 명령줄로 수정하는 모듈
import torch #Pytorch
from tqdm import tqdm #진행 정도 bar
import data_loader.data_loaders as module_data #data_loader 폴더의 data_loaders.py 파일 로드
import model.loss as module_loss #model 폴더의 loss.py 파일 로드
import model.metric as module_metric #model 폴더의 metric.py 파일 로드
import model.model as module_arch #model 폴더의 model.py 파일 로드
from parse_config import ConfigParser #configParser 클래스 사용해서 JSON 설정 파일 및 cls 옵션 처리


def main(config):
    logger = config.get_logger('test') #테스트 로그 남기기 위해 logger 설정

    #getattar 함수로 config file에서 지정한 type 항목으로 data_loader 인스턴스 생성
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0, #테스트라서 validation_split 없음
        training=False, #테스트라서 학습 끔
        num_workers=2 #병렬 처리 worker 프로세스 수, 0이면 싱글로 병렬x, 1 이상이면 그 숫자만큼 병렬처리, 일반적으로 cpu 코어 수의 절반 사용
    )

    #모델 구조 생성
    model = config.init_obj('arch', module_arch) #config json파일에서 모델 아키텍처 로드
    logger.info(model) #모델 정보 출력

    #loss, metric 처리
    loss_fn = getattr(module_loss, config['loss']) #config file에서 정의한 loss 함수 로드
    metric_fns = [getattr(module_metric, met) for met in config['metrics']] #config file에서 정의한 metrics list에서 평가지표 하나씩 로드

    logger.info('Loading checkpoint: {} ...'.format(config.resume))  #config file에서 이전 학습 상태 체크포인트 경로
    checkpoint = torch.load(config.resume) #지정된 체크포인트 파일 로드
    state_dict = checkpoint['state_dict'] #체크포인트 파일에서 모델의 state dict 추출, state dict는 각 레이어의 가중치, bias등 현재 학습 상태의 파라미터 갖는 딕셔너리
    if config['n_gpu'] > 1: #gpu 1개 이상이면
        model = torch.nn.DataParallel(model) #병렬 처리
    model.load_state_dict(state_dict) #체크포인트 파일에서 불러온 state_dict를 모델에 적용

    #모델 테스트 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #가능하면 gpu, 안되면 cpu
    model = model.to(device) #모델에 gpu 설정
    model.eval() #평가 모드 설정

    total_loss = 0.0 #총 loss 초기화
    total_metrics = torch.zeros(len(metric_fns)) #각 metric들을 초기화 (각 metric별로 0 지정)

    with torch.no_grad(): #평가니까 gradient 계산 비활성화
        for i, (data, target) in enumerate(tqdm(data_loader)): #tqdm으로 진행 상태 표시
            data, target = data.to(device), target.to(device) #데이터를 device(gpu or cpu)로 이동
            output = model(data) #모델에 입력값 넣어 예측값 생성

            #
            # save sample images, or do something with output here
            #

            #테스트 데이터에 대한 loss, metric 계산
            loss = loss_fn(output, target) #예측과 실제값 손실 꼐산
            batch_size = data.shape[0] #현재 배치 크기
            total_loss += loss.item() * batch_size #평균 손실 계산 위해서, 배치 손실을 더해 total loss 만들어줌
            for i, metric in enumerate(metric_fns): #각각의 metric 계산 위해 하나씩 돌면서
                total_metrics[i] += metric(output, target) * batch_size #평가지표 * 배치 크기의 합계로 전체 metrics 계산

    n_samples = len(data_loader.sampler) #전체 샘플 수
    log = {'loss': total_loss / n_samples} #평균 loss 계산해서 log 딕셔너리 생성
    log.update({ #log 딕셔너리에 평균 metric 지표 추가
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    }) #total_metrics[i].item() / n_samples는 해당 metric의 평균값, metric_fns는 평가지표 리스트니까 하나씩 돌면서 저장하는 것
    logger.info(log) #log 딕셔너리 내용 콘솔에 출력


if __name__ == '__main__':  #test.py가 직접 실행될때만 동작, 다른 코드에서 모듈로 가져오면 내부 실행 X
    args = argparse.ArgumentParser(description='PyTorch Template') #argparse는 코드 내부 수정없이 명령어로 config 파일 옵션값 수정 위해 사용, 객체 생성
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)') #add_argument는 수정 가능한 함수, -c, --config로 config file 경로 수정 가능
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)') #-r, --resume로 checkpoint 경로 수정 가능
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)') #-d, --device로 사용할 GPU 인덱스 수정 가능

    config = ConfigParser.from_args(args) #configParser는 명렬줄에서 전달된 옵션 args 읽고, json파일과 결합해 최종 config 생성
    main(config) #main에 최종 config 전달
