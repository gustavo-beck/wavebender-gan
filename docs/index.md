# Wavebender GAN:
### Deep architecture for high-quality and controllable speech synthesis through interpretable features and exchangeable neural synthesizers
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

[github_link]: https://https://github.com/gustavo-beck/wavebender-gan
[gustav_profile]: https://people.kth.se/~ghe/
[gustavo_profile]: https://www.linkedin.com/in/gustavotbeck/
[ulme_profile]: https://www.kth.se/profile/ulme
[zofia_profile]: https://www.kth.se/profile/malisz

## Summary

Lorem Ipsum is simply dummy text of the printing and 
typesetting industry. Lorem Ipsum has been the industry's 
standard dummy text ever since the 1500s, when an unknown 
printer took a galley of type and scrambled it to make a 
type specimen book. It has survived not only five centuries, 
but also the leap into electronic typesetting, 
remaining essentially unchanged.

## Architecture

![Wavebender GAN](./images/WavebenderGAN.png "Architecture of Wavebender GAN")

## Code

Code is available on our [Github repository][github_link], along with a pre-trained model.

## Listening examples

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

### Reproducibility 

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Type</th>
      <th class="tg-0pky" colspan="2">Proposed neural HMM TTS</th>
      <th class="tg-0pky" colspan="2">Tacotron 2 baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th class="tg-fymr">Condition</th>
      <th class="tg-fymr">2 states per phone (NH2)</th>
      <th class="tg-fymr">1 state per phone (NH1)</th>
      <th class="tg-fymr">w/o post-net (T2-P)</th>
      <th class="tg-fymr">w/ post-net (T2+P)</th>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 1</b></td>
      <td class="tg-0pky">
        <audio id="audio-small" controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_001.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_001.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_001.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_001.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 2</b></td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_002.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_002.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_002.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_002.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 3</b></td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_003.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_003.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_003.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_003.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 4</b></td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_004.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_004.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_004.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_004.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 5</b></td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_005.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_005.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_005.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_005.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky"><b>Sentence 6</b></td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/2State/NeuralHMM1AR2State_hvd_006.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/NeuralHMM/1State/NeuralHMM1AR1State_hvd_006.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWOPostnet/Tacotron_hvd_006.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/Tacotron/TacotronWPostnet/Tacotron_Postnet_hvd_006.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
</table>

### NEW TABLE
