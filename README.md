# Wavebender GAN:
### Deep architecture for high-quality and controllable speech synthesis through interpretable features and exchangeable neural synthesizers
##### [Gustavo Teodoro Döhler Beck][gustavo_profile], [Ulme Wennberg][ulme_profile], [Zofia Malisz][zofia_profile], [Gustav Eje Henter][gustav_profile]
---

[paper_link]: https://arxiv.org/abs/2108.13320
[gustav_profile]: https://people.kth.se/~ghe/
[gustavo_profile]: https://www.linkedin.com/in/gustavotbeck/
[ulme_profile]: https://www.kth.se/profile/ulme
[zofia_profile]: https://www.kth.se/profile/malisz
[demo_page]: 
[pretrained_model_link]: 
[ljspeech_link]: https://keithito.com/LJ-Speech-Dataset/
[github_link]: https://github.com/gustavo-beck/wavebender-gan
[github_new_issue_link]: https://github.com/gustavo-beck/wavebender-gan/issues/new
[tacotron2_link]: https://github.com/NVIDIA/tacotron2
[nvidia_waveglow_link]: https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view
[hifi_link]: https://github.com/jik876/hifi-gan


For audio examples, visit our [demo page][demo_page]. A [pre-trained model](pretrained_model_link) is also available.

## Setup and training using LJ Speech
1. Download and extract the [LJ Speech dataset][ljspeech_link]. Place it in the `data` folder such that the directory becomes `data/wavs`.
2. Clone this repository ```git clone https://github.com/gustavo-beck/wavebender-gan.git``` 
   * This project is expected to be run with GPU.

## Synthesis
1. Download our [pre-trained LJ Speech model][pretrained_model_link]. 
2. Download Nvidia's [WaveGlow model][nvidia_waveglow_link].
3. Download Nvidia's [Tacotron 2 model][tacotron2_link].
4. Download [HiFi-GAN model][hifi_link].
5. Run ```vebncejd.py```.

## Support
If you have any questions or comments, please open an [issue][github_new_issue_link] on our GitHub repository.

## Acknowledgements
The code implementation is based on [Nvidia's implementation of Tacotron 2][tacotron2_link] and Nvidia's [WaveGlow model][nvidia_waveglow_link] for mel-spectrograms generation for the training step, and [HiFi-GAN model][hifi_link] for speech synthesis.
