# Wavebender GAN:
### Deep architecture for high-quality and controllable speech synthesis through interpretable features and exchangeable neural synthesizers
##### [Gustavo Teodoro Döhler Beck][gustavo_profile], [Ulme Wennberg][ulme_profile], [Zofia Malisz][zofia_profile], [Gustav Eje Henter][gustav_profile]
---

[paper_link]: https://arxiv.org/abs/2202.10973
[gustav_profile]: https://people.kth.se/~ghe/
[gustavo_profile]: https://www.linkedin.com/in/gustavotbeck/
[ulme_profile]: https://www.kth.se/profile/ulme
[zofia_profile]: https://www.kth.se/profile/malisz
[demo_page]: https://gustavo-beck.github.io/wavebender-gan/
[ljspeech_link]: https://keithito.com/LJ-Speech-Dataset/
[github_link]: https://github.com/gustavo-beck/wavebender-gan
[github_new_issue_link]: https://github.com/gustavo-beck/wavebender-gan/issues/new
[tacotron2_link]: https://github.com/NVIDIA/tacotron2
[nvidia_waveglow_link]: https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view
[hifi_link]: https://github.com/jik876/hifi-gan

This is the official code repository for the paper [Wavebender GAN: An architecture for phonetically meaningful speech manipulation][paper_link].

For audio examples, visit our [demo page][demo_page].


---------

##### Data
All the 13100 audio samples from the [LJ speech][ljspeech_link] data set should be stored in data/wavs/. Then they should be split and the results should be stored in wavebender_features_data/train/, wavebender_features_data/test/. In these folders there are .txt files with the corresponding audios filed for each data set.

##### Tacotron 2
Before start training you need to download waveglow_256channels_universal_v5.pt and save it in the main folder and in the tacotron2 folder. You should have both copies

##### Training
Wavebender Net and GAN are trained separetelly. Therefore, you can train each one of them by running train_wavebender_net.py or train_wavebender_gan.py. Don't forget to have the data already in the correct format to run them.
