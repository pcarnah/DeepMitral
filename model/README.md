Pretrained network on 40 3D TEE volumes as described in:

P. Carnahan, J. Moore, D. Bainbridge, M. Eskandari, E. C. S. Chen and T. M. Peters. “DeepMitral: Fully Automatic 3D Echocardiography Segmentation for Patient Specific Mitral Valve Modelling” in Proceedings of MICCAI 2021


checkpoint_final.pt can be used for deepmitral training, validate or segment.

Training will continue from this checkpoint, validate and segment will use the checkpoint to laod network parameters.