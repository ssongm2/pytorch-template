import sys
from pathlib import Path
from shutil import copytree, ignore_patterns

#template file로 새로운 pytorch project 초기화하는 sciprt
# 새로운 프로젝트 생성하려면 다음을 실행 `python3 new_project.py ../MyNewProject`
# MyNewProject가 만들어짐

current_dir = Path()
assert (current_dir / 'new_project.py').is_file(), 'Script should be executed in the pytorch-template directory'
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject'

project_name = Path(sys.argv[1])
target_dir = current_dir / project_name

ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"]
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
print('New project initialized at', target_dir.absolute().resolve())
