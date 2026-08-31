from setuptools import setup

package_name = 'autonomy_stack'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/main.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/eval_debug.rviz', 'rviz/eval_inference.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Master launch package for the AV6 autonomy stack',
    license='TODO: License declaration',
    entry_points={},
)
