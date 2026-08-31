import os
from setuptools import setup
from setuptools.command.install_scripts import install_scripts as _install_scripts

package_name = 'autonomy_perception'

class install_scripts(_install_scripts):
    def run(self):
        self.install_dir = os.path.join(os.path.dirname(self.install_dir), 'lib', package_name)
        os.makedirs(self.install_dir, exist_ok=True)
        super().run()

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
    description='Perception domain package split for AV6',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'carla_node = autonomy_perception.carla_node_native:main',
            'dashboard_node = autonomy_perception.dashboard_node:main',
            'doppler_node = autonomy_perception.doppler_node:main',
            'engine_builder_node = autonomy_perception.engine_builder_node:main',
            'infrence_node = autonomy_perception.infrence_node:main',
            'traffic_light_fusion_node = autonomy_perception.traffic_light_fusion_node:main',
            'traffic_sign_node = autonomy_perception.traffic_sign_node:main',
        ],
    },
    cmdclass={'install_scripts': install_scripts},
)
