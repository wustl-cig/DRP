# [A Restoration Network as an Implicit Prior]

Image denoisers have been shown to be powerful priors for solving inverse problems in imaging.  In this work, we introduce a generalization of these methodsthat allows any image restoration network to be used as an implicit prior. The pro-posed method uses priors specified by deep neural networks pre-trained as general restoration operators.  The method provides a principled approach for adapting state-of-the-art restoration models for other inverse problems. Our theoreticalresult analyzes its convergence to a stationary point of a global functional associated with the restoration operator.  Numerical results show that the method usinga super-resolution prior achieves state-of-the-art performance both quantitativelyand qualitatively. Overall, this work offers a step forward for solving inverse prob-lems by enabling the use of powerful pre-trained restoration models as priors.


## Requirements

The environment is

```
python=3.7.11, numpy=1.21.5, scipy==1.7.3, pytorch=1.10.2=py3.7_cuda11.3_cudnn8.2.0_0
```

## Models

The pre-trained models can be downloaded from [Google drive](https://drive.google.com/drive/folders/1_FWxwMotwD-U1z3IhqsA_AkT_URiUzza?usp=sharing).
Once downloaded, place them into `./model_zoo/swinir`.
## How to run the code

deblur:

```
python deblur_experments/DRP_deblur_k1.py
```

sisr:

```
python sisr_experments/DRP_sisr_2x_k1.py
```

denoise:

```
python denoising_experiments/DRP_denoising_2x.py 
```
