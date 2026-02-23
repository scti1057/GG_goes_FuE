
from setuptools import setup

setup(
    name='irp_ur5e_support',
    version='0.0.1',
    packages=['kovis_moduls', 'pih_moduls'],
    package_dir={'': 'scripts'},
    install_requires=['setuptools'],
    zip_safe=True,
          entry_points={
            'console_scripts': [
                'ap_movePose = ap_movePose:main',
                'ap_moveTwist = ap_moveTwist:main',
                'ap_moveJointsPos = ap_moveJointsPos:main',
                'robot_pose_publisher = robot_pose_publisher:main',
                'admittanz_regler = admittanz_regler:main',
                'admittanz_regler_publisher = admittanz_regler_publisher:main',
                'admittance_force_controller_test = admittance_force_controller_test:main',
            ],
        },
)

