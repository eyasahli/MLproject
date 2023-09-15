from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines() #readlines includes the \n
        requirements=[req.replace("\n","") for req in requirements]  #cleaning the lines
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup (
    name='mlproject',
    version='0.0.1',
    author='Eya',
    author_email='eyasahli32@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)