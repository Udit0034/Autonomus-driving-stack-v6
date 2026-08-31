import os
from setuptools import setup
from setuptools.command.install_scripts import install_scripts as _install_scripts

package_name = 'autonomy_planning'

class install_scripts(_install_scripts):
    def run(self):
        self.install_dir = os.path.join(os.path.dirname(self.install_dir), 'lib', package_name)
        os.makedirs(self.install_dir, exist_ok=True)
        super().run()

setup(
    name=package_name,
    version='0.0.0',
    packages=['autonomy_planning'],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/empty.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Planning domain package for AV6',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'mission_planner_node = autonomy_planning.mission_planner_node:main',
            'vpg_node = autonomy_planning.vpg_node:main',
        ],
    },
    cmdclass={'install_scripts': install_scripts},
)
