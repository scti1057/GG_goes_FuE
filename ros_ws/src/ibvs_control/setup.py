from setuptools import find_packages, setup

package_name = 'ibvs_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duckie5',
    maintainer_email='duckie.town@web.de',
    description='IBVS control nodes for twist command generation',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ibvs_twist_controller = ibvs_control.ibvs_twist_controller_node:main',
        ],
    },
)
