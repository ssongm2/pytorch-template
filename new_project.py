import sys
from pathlib import Path
from shutil import copytree, ignore_patterns

#template file로 새로운 pytorch project 초기화하는 sciprt
# 새로운 프로젝트 생성하려면 다음을 실행 `python3 new_project.py ../MyNewProject`
# MyNewProject가 만들어짐

current_dir = Path() #현재 dir
assert (current_dir / 'new_project.py').is_file(), 'Script should be executed in the pytorch-template directory' #현재 dir에 new_project.py가 존재하는지 확인
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject' #sys.argv로 프로젝트 이름 할당 됐는지 확인, 인자 없으면 에러 출력

project_name = Path(sys.argv[1]) #sys.argv의 1번 인덱스가 project_name
target_dir = current_dir / project_name #dir 설정

ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"] #copy 대상 중 제외할 file과 dir 정의
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore)) #현재 dir의 모든 file을 target_dir로 복사, 이때 ignore은 제외
print('New project initialized at', target_dir.absolute().resolve()) #새 project인 target_dir의 경로 출력
