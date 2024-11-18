from torchvision import datasets, transforms 
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader): #BaseDataLoader 상속받아서 MNIST 데이터셋에 맞게 확장한 클래스
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data_dir: MNIST 데이터셋 저장할 경로, batch_size: 한번에 로드할 샘플 개수, num_workers: 병렬 프로세스 수(기본 1), training: 학습이면 True, 테스트면 False
        trsfm = transforms.Compose([ #여러 변환 순차적으로 적용
            transforms.ToTensor(), #텐서로 변환 (0~1 사이로 정규화)
            transforms.Normalize((0.1307,), (0.3081,)) #평균 0.1307, 표준편차 0.3081로 데이터 정규화
        ])
        self.data_dir = data_dir #데이터셋 저장할 경로
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm) #torchvision에서 제공하는 MNIST 데이터셋 로드
        #data_dir: 데이터셋 경로, train: 학습이면 True, download: 데이터셋 경로에 없으면 다운로드, transform: 데이터변환 적용함
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        #super().__init__(): 부모 클래스(BaseDataLoader) 생성자 호출, BaseDataLoader에서 정의된 data load 관련 기본 설정 초기화
        #데이터셋, 사용자 설정 값을 부모 클래스에 전달

#<요약>
#1. MNIST 데이터셋 로드
#2. BaseDataLoader 초기화
#3. 데이터 로더 생성
#이렇게 해서 BaseLoader의 기본 기능을 재사용하여 MNIST에 맞게 수정 가능!
