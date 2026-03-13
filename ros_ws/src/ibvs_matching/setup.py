from setuptools import find_packages, setup

package_name = 'ibvs_matching'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'matches_viz = ibvs_matching.matches_viz_node:main',

            'descriptor_matcher = ibvs_matching.descriptor_matcher_node:main',
            'local_rescue_debug = ibvs_matching.local_rescue_debug_node:main',

        ],
    },
)
