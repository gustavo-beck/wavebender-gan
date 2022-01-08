# Wavebender GAN:
Deep architecture for high-quality and controllable speech synthesis through interpretable features and exchangeable neural synthesizers
##### [Gustavo Teodoro Döhler Beck][gustavo_profile], [Ulme Wennberg][ulme_profile], [Zofia Malisz][zofia_profile], [Gustav Eje Henter][gustav_profile]


<head> 
<link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff">
</head>
<!-- This post presents Wavebender GAN a deep architecture for high-quality and controllable speech synthesis through interpretable features and exchangeable neural synthesizers -->

[github_link]: https://github.com/gustavo-beck/wavebender-gan
[gustav_profile]: https://people.kth.se/~ghe/
[gustavo_profile]: https://www.linkedin.com/in/gustavotbeck/
[ulme_profile]: https://www.kth.se/profile/ulme
[zofia_profile]: https://www.kth.se/profile/malisz
[hifi_link]: https://github.com/jik876/hifi-gan
[ljspeech_link]: https://keithito.com/LJ-Speech-Dataset/

## Summary

Modeling humans' speech is a challenging task that originally required a coalition between phoneticians and speech engineers. Yet, the latter, disengaged from phoneticians, have strived for evermore natural speech synthesis in the absence of an awareness of speech modeling due to data-driven and ever-growing deep learning models. By virtue of decades of detachment between phoneticians and speech engineers, this thesis presents a deep learning architecture, alleged Wavebender GAN, that predicts mel-spectrograms that are processed by a vocoder, [HiFi-GAN][hifi_link], to synthesize speech. Wavebender GAN pushes for progress in both speech science and technology, allowing phoneticians to manipulate stimuli and test phonological models supported by high-quality synthesized speeches generated through interpretable low-level signal properties. This work sets a new step of cooperation for phoneticians and speech engineers.

The samples presented in this page are from [LJ speech][ljspeech_link] dataset, which is a public dataset that consists of 13,100 short audio
clips of a single speaker. The same dataset was used to train Wavebender GAN.

## Architecture

![Wavebender GAN](./images/WavebenderGAN.png "Architecture of Wavebender GAN")

## Code

Code is available on our [Github repository][github_link], along with a pre-trained model.

<style type="text/css">
  .tg {
    border-collapse: collapse;
    border-color: #9ABAD9;
    border-spacing: 0;
  }

  .tg td {
    background-color: #EBF5FF;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #444;
    font-family: Arial, sans-serif;
    font-size: 14px;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
  }

  .tg th {
    background-color: #409cff;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #fff;
    font-family: Arial, sans-serif;
    font-size: 14px;
    font-weight: normal;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;

  }

  .tg .tg-0pky {
    border-color: inherit;
    text-align: center;
    vertical-align: top,
  }

  .tg .tg-fymr {
    border-color: inherit;
    font-weight: bold;
    text-align: center;
    vertical-align: top
  }
  .slider {
  -webkit-appearance: none;
  width: 75%;
  height: 15px;
  border-radius: 5px;  
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border-radius: 50%; 
  background: #409cff;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}
</style>

## Reproducibility 

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Type</th>
      <th class="tg-0pky" colspan="1">Original</th>
      <th class="tg-0pky" colspan="1">Wavebender GAN</th>
      <th class="tg-0pky" colspan="1">HiFi-GAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 1</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_1.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_1.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_1.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 2</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_2.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_2.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_2.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 3</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_3.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_3.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_3.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 4</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_4.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_4.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_4.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 5</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_5.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_5.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_5.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 6</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_6.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_6.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_6.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 7</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_7.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_7.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_7.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 8</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_8.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_8.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_8.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 9</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_9.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_9.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_9.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
  
  <tbody>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 10</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/Original/original_10.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Reproduction/wavebendergan_10.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/HiFiGAN/hifi_10.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
</table>

## Manipulation  

It is important to recall that this project has the purpose of achieving naturalness and controllable speech. Whereas the previous examples focused on evaluating the speech naturalness reconstruction against a state-of-the-art system ([HiFi-GAN][hifi_link]), this section leverages the fact that our system can also manipulate speech, which is not possible with other neural synthesizers. The manipulation consists of extracting the low-level signal properties and multiplying one of the properties at a time by a specific scale (e.g., multiplying F0-contour by a factor of 1.3 across the entire signal).

For simplicity, all of the following manipulations were done on a particular sample (LJ026-0014) from the [LJ speech][ljspeech_link] dataset.

### F0-contour (pitch)

For this experiment, we present how Wavebender GAN is capable of reducing and increasing the pitch (F0-contour) of the speaker. 

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky" colspan="1">-30%</th>
      <th class="tg-0pky" colspan="1">-15%</th>
      <th class="tg-0pky" colspan="1">LJ026-0014</th>
      <th class="tg-0pky" colspan="1">+15%</th>
      <th class="tg-0pky" colspan="1">+30%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_low_30_f0_contour.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_low_15_f0_contour.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_high_15_f0_contour.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_high_30_f0_contour.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
</table>

### F1 and F2

Something

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky" colspan="1">Fools (-30% of F1)</th>
      <th class="tg-0pky" colspan="1">Forms</th>
      <th class="tg-0pky" colspan="1">Frogs (+30% of F1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_low_f1.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_forms.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_high_f1.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
</table>

### Spectral centroid

Something

<audio controls>
  <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_high_spectral_centroid.wav" type="audio/wav">
</audio>
        
### Spectral Slope

Something

<audio controls>
  <source src="./audios/WavebenderGAN/Manipulation/LJ026-0014_low_spectral_slope.wav" type="audio/wav">
</audio>
