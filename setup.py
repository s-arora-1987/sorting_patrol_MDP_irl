from distutils.core import setup 
from catkin_pkg.python_setup import generate_distutils_setup 

d = generate_distutils_setup( 
    packages=['sorting_patrol_MDP_irl'], 
    package_dir={'': 'sortingMDP'} 
) 

setup(**d) 
