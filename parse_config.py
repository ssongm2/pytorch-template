import os #파일 경로, 시스템 관련 작업
import logging #로그 기록
from pathlib import Path #경로 처리
from functools import reduce, partial #함수형 프로그래밍 도구, reduce: 여러 값 누적하여 처리, partial: 함수 인자의 일부를 디폴트값 고정해서 새로운 함수 생성
from operator import getitem #딕셔너리, 리스트에서 특정 항목 로드
from datetime import datetime #날짜, 시간 처리
from logger import setup_logging #로깅 설정 초기화
from utils import read_json, write_json #json 읽기, 쓰기


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        # config 파일 파싱해서 하이퍼파라미터, 모듈 초기화, 체크포인트, 로깅 등 처리하는 클래스
        :config json file 파싱하는 클래스. 학습 위한 하이퍼파라미터 철리, 모듈 초기화, 체크포인트 저장, 로깅 모듈 처리
        :param config: 기타 설정, 하이퍼파라미터 담긴 딕셔너리. config.json file의 내용물.
        :param resume: String형, 체크포인트 경로
        :param modification: 키-값 딕셔너리, config dict의 특정 값 수정
        :param run_id: 학습 과정 unique 식별자. 체크포인트 저장, 학습 로그 저장에 사용. 디폴트는 timestamp 형태
        """
        # config file 로드 및 추가 수정
        self._config = _update_config(config, modification) #config file 로드 및 수정사항 업데이트
        self.resume = resume #클래스 인스턴스에 resume 저장, 클래스에서 체크포인트 경로 사용할 수 있도록 함

        # 학습된 모델 및 로그가 저장될 경로 지정
        save_dir = Path(self.config['trainer']['save_dir']) #trainer가 저장될 곳은 save_dir에 담긴 경로

        exper_name = self.config['name'] #config file에서 실험 이름 로드
        if run_id is None:  #run_id가 전달된게 없다면 (run_id의 디폴트는 timestamp)
            run_id = datetime.now().strftime(r'%m%d_%H%M%S') #현재 시간이 run_id
        self._save_dir = save_dir / 'models' / exper_name / run_id #저장 경로는 save_dir/ 'models' / 실험이름 / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id #로그 경로는 save_dir / 'log' / 실험이름 / run_id

        # 체크포인트와 로그 저장을 위한 dir 생성
        exist_ok = run_id == '' 
            #run_id가 빈 문자열이면 exist_ok=True. 즉, 기존 dir 재사용. 내부 파일들을 덮어씀.
            #run_id가 빈 문자열 아니면 exist_ok=False. 즉, 새로운 dir 사용. 기존 run_id로 dir 있으면 중복으로 FileExistError 발생해서 덮어쓰기 방지.
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok) #체크포인트 dir 저장할 생성
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok) #로그(loss, metric) dir 저장할 생성

        # 체크포인트 dir를 config file에 업데이트
        write_json(self.config, self.save_dir / 'config.json')

        # 로깅 관련 설정 모듈
        setup_logging(self.log_dir) #self.log_dir로 dir사용해 로그 파일 기록하도록 설정
        self.log_levels = { #log_levels은 로깅 레벨
            0: logging.WARNING, #0이면 WARNING만 기록
            1: logging.INFO, #1이면 info, warning 기록
            2: logging.DEBUG #2면 디버깅 포함해 모두 기록
        }

    @classmethod #클래스 메서드 데코레이터: 클래스 전체에 속하는 메서드 정의, 클래스 자체를 인자로 받음(=클래스 인스턴스 없어도 호출 가능)
    def from_args(cls, args, options=''): 
            #cls 인자: classmethod라 첫번째로 클래스 객체 cls를 자동으로 받는 것. 이걸로 클래스 속성이나 메서드에 접근 가능.
            #args, options????????
        """
        cli 인자 기반으로 클래스 초기화. train, test에서 사용.
        """
        #커맨드 옵션 추가
        for opt in options: #options: 커맨드로 입력받을 custom option list, namedtuple 타입으로 flags, type, target으로 구성.
            args.add_argument(*opt.flags, default=None, type=opt.type) 
                #add_argument로 커맨드에서 입력받을 옵션 추가. flags는 옵션 이름 리스트(ex. --lr), 입력 못받으면 디폴트는 None, type은 각 flag의 타입(ex. int, float)
        #커맨드 옵션 파싱
        if not isinstance(args, tuple): #args가 tuple인지 확인(= args 객체가 파싱된 상태인지 확인)하고 아니라면,
            args = args.parse_args() #커맨드로 받은 값 파싱해서 객체로 변환

        if args.device is not None: #커맨드로 받은 device 환경 있으면,
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device #지정받은 GPU만 사용
        if args.resume is not None: #커맨드로 받은 checkpoint 경로 있으면,
            resume = Path(args.resume) #경로를 객체로 변환
            cfg_fname = resume.parent / 'config.json' #해당 checkpoint파일 저장된 dir에서 config 파일 찾음
        else: #커맨드에서 checkpoint 경로 지정 없으면,
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example.
            assert args.config is not None, msg_no_cfg #커맨드에서 config 지정 없었으면, 에러 출력
            resume = None #checkpoint 경로 없음
            cfg_fname = Path(args.config) #커맨드에서 지정된 config 파일 경로 객체로 변환
        
        config = read_json(cfg_fname) #변환한 file 읽기
        if args.config and resume: #커맨드에서 config 파일 지정하고, checkpoint 재개하는 경우
            # fine-tuning 위해 업데이트
            config.update(read_json(args.config))
                #새로운 config file 내용을 기존 config에 업데이트

        # cli 옵션 딕셔너리로 변환
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
            # 수정사항 정리. opt.target은 config에서 변경할 값의 위치 나타냄, _get_opt_name(opt.flags)는 커맨드로 받음
        return cls(config, resume, modification)
            #configParser 클래스 사용해서 config, 새로 시작한 checkpoint(fine-tuning), 수정 사항 전달

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
