import numpy as np
import torch
from torchvision.utils import make_grid #이미지 텐서를 grid로 배열해 시각화
from base import BaseTrainer #학습 loop 관리 위한 기본 클래스
from utils import inf_loop, MetricTracker #무한 loop 생성 위한 헬퍼 함수, metric 관리하는 클래스


class Trainer(BaseTrainer): #학습을 관리하는 역할하는 클래스, BaseTrainer 상속받아 학습, 검증 진행
    """
    Trainer class
    """
    #초기화: 모델, 손실, metric, 옵티마이저, 설정값, device, 데이터 로더, val_데이터 로더, 학습률 스케줄러
    #len_epoch=None은 epoch 기반 학습, len_epoch=지정값 있으면 iter 기반 학습
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        #기본 설정
        self.config = config
        self.device = device
        self.data_loader = data_loader

        #epoch 기반 학습이면,
        if len_epoch is None:
            self.len_epoch = len(self.data_loader) #데이터 로더 크기 사용
        #iter 기반 학습이면,
        else:
            self.data_loader = inf_loop(data_loader) #무한 데이터 로더로 변환(이유: 만약 데이터셋 작으면 반복 학습 중 데이터 부족할 수 있음 이걸 방지하기 위해)
            self.len_epoch = len_epoch #지정된 횟수만큼 학습 반복 횟수 제한
        #검증 데이터로더             
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None #검증 데이터 로더가 제공됐는지 확인해서 수행 여부 결정(True: 검증 수행, False: 검증 수행X)

        #학습률 스케줄러
        self.lr_scheduler = lr_scheduler
        #log step 설정: 학습 진행 중 로그 출력할 간격 설정
        self.log_step = int(np.sqrt(data_loader.batch_size)) #배치 크기의 제곱근으로 설정

        #MetricTracker로 학습, 검증에서 loss, metric 기록
        #metric 이름은 함수 이름으로 추출하고, writer는 Tensorboard 시각화 도구로 연결해서 출력 가능
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    #epoch당 train 과정 정의하는 함수
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        #epoch: 정수형, 현재 학습 중인 에폭
        #반환값: 해당 에폭에서의 평균 loss와 metric
        
        """
        self.model.train() #학습 모드 전환
        self.train_metrics.reset() #metric 초기화
        for batch_idx, (data, target) in enumerate(self.data_loader): #data loader에서 배치 가져와 학습
            data, target = data.to(self.device), target.to(self.device) #데이터 device로 전달

            #딥러닝 학습 과정
            self.optimizer.zero_grad() #기존 gradient 초기화
            output = self.model(data) #output 예측
            loss = self.criterion(output, target) #예측과 실제값 손실 계산
            loss.backward() #gradient 역전파
            self.optimizer.step() #가중치 업데이트

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx) 
                #set_step: 현재 학습 단계(에폭과 관계없이 전체 학습 과정에서의 진행) 설정해서 시각화에 사용
                #epoch-1: 이전에 학습한 epoch의 idx, len_epoch: 한 epoch의 배치수(데이터로더 100, 배치크기 10이면 len_epoch=10), batch_idx: 현재 epoch의 batch index
                #(epoch-1)*len_epoch: 이전 에폭까지의 배치 수, batch_idx: 현재 에폭에서의 학습 배치 정도
                #즉, 현재까지 계산된 총 배치 수
            self.train_metrics.update('loss', loss.item()) #loss metric 업데이트(loss.item: 현재 배치의 loss)
            for met in self.metric_ftns: #metric list 중에 하나씩 돌면서
                self.train_metrics.update(met.__name__, met(output, target)) #예측과 실제값으로 평가 및 업데이트 

            if batch_idx % self.log_step == 0: #현재 배치 idx가 log_step의 배수일 때 log 출력
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format( #디버깅 레벨로 로그 기록
                    epoch,
                    self._progress(batch_idx),
                    loss.item())) #디버깅 레벨: 현재 에폭, 진행상황, 현재 배치 손실
                    #DEBUG 레벨: 상세 값들 출력해서 세부 동작 과정을 확인할 수 있음. (배치별 손실, 학습률, 파라미터 변화 등)
                    #(참고) INFO 레벨: 간단한 요약 정보 확인 가능. 일반적 로그. (에폭, 손실, 최정 정확도 등)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True)) #학습 데이터를 시각화 도구(ex. Tensorboard)에 기록, 한 행에 8개씩 데이터 넣음

            if batch_idx == self.len_epoch: #한 에폭 다 돌면 멈춤
                break
        log = self.train_metrics.result() #한 에폭 끝나면 학습 결과 log 객체 생성

        if self.do_validation: #do_validation 켜지면,
            val_log = self._valid_epoch(epoch) #검증 단계 실행
            log.update(**{'val_'+k : v for k, v in val_log.items()}) #검증 결과를 로그에 추가, 이때 val_을 붙여 학습 로그와 구분

        if self.lr_scheduler is not None: #학습률 스케줄러 값 지정받으면,
            self.lr_scheduler.step() #에폭 종료하고 학습률 업데이트
        return log #로그 반환

    #epoch당 valid 과정 정의하는 함수
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        epoch: 정수형, 현재 학습 에폭
        반환: 검증과 관련된 로그 정보
        """
        self.model.eval() #평가 모드 전환
        self.valid_metrics.reset() #평가 metric 초기화
        with torch.no_grad(): #검증이니까 gradinet 업데이트 없음
            for batch_idx, (data, target) in enumerate(self.valid_data_loader): #valid 데이터 로더에서 데이터 하나씩 불러와서
                data, target = data.to(self.device), target.to(self.device) #device로 데이터 이동

                output = self.model(data) #검증 데이터로 예측 생성
                loss = self.criterion(output, target) #예측값과 실제값 손실 계산

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid') #현재까지 처리된 배치 개수
                self.valid_metrics.update('loss', loss.item()) #loss 값 valid metric에 업데이트
                for met in self.metric_ftns: #metric list에서 하나씩 metric 가져와서
                    self.valid_metrics.update(met.__name__, met(output, target)) #실제, 예측으로 계산해서 업데이트
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True)) #현재 배치 입력 데이터를 시각화 도구(Tensorboard)에 기록, make_grid는 배치 데이터로 한 행에 8개 데이터

        # 모델 파라미터의 히스토그램을 tensorboard에 추가
        for name, p in self.model.named_parameters(): #파라미터값, 파라미터 이름을 함께 로드
            self.writer.add_histogram(name, p, bins='auto') 
                #tensor를 histogram으로 기록해 TensorBoard에 표시, name: 파라미터 이름, p: 파라미터 텐서 bins='auto': 구간 개수 자동으로 설정
        return self.valid_metrics.result() #검증의 모든 metric 최종 결과 반환

    #현재 진행 상태를 계산
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]' #양식
        if hasattr(self.data_loader, 'n_samples'): #data_loader 객체에 n_samples 속성이 있다면,
            current = batch_idx * self.data_loader.batch_size #지금까지 처리된 데이터 샘플 개수(배치 idx * batch_Size)
            total = self.data_loader.n_samples #전체 샘플 개수
        else: #n_samples 속성 없다면,
            current = batch_idx #현재 배치 idx
            total = self.len_epoch #한 epoch에서 처리할 전체 배치 개수 (len_epoch)
        return base.format(current, total, 100.0 * current / total) (현재진행, 총 데이터 개수, 몇% 진행)
