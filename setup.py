from setuptools import find_packages,setup
from typing import List 

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements.extend(f.readlines())
        requirements = [req.strip() for req in requirements]
        requirements = [req for req in requirements if req != HYPEN_E_DOT]
    return requirements


setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Aravind',
    author_email='abhi.aripaka2002@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)

