# AIFriendly-circuits

In the present script AI-Friendly learns how to detect if adding a diode will result in it being burnt for different circuits

---

<p align="center"><b>CIRCUIT 1: RLC-series circuit</b><br></p>
<b> This is the circuit<img width=100, height=200, src="./circuits/AI_circuit_1/schematic/schematic_dioded.png"><br>
This are the currents over each passive component <img width=600, height=300, src="./circuits/AI_circuit_1/gallery/RLC_example.png"><br>
This are some resonance curves: current over resistance vs freq<img width=600, height=300, src="./circuits/AI_circuit_1/gallery/RLC_resonance-curves.png"><br>
This are the parameters used to generate the database<img width=600, height=300, src="./circuits/AI_circuit_1/gallery/database-parameters.png"><br>
This is the ROC of AI-Friendly over the database<img width=600, height=300, src="./circuits/AI_circuit_1/gallery/figPerf.png"><br>
We conclude it has learnt very well</b>

---

<p align="center"><b>CIRCUIT 2: Sallen-Key low-pass-filter</b><br></p>
<b> This is the circuit<img width=100, height=200, src="./circuits/AI_circuit_2/schematic/schematic_dioded.png"><br>
This are the input and output voltage <img width=600, height=300, src="./circuits/AI_circuit_2/gallery/sallen-key-low-pass-filter_example.png"><br>
This are some transmission curves: output voltage over resistance vs freq<img width=600, height=300, src="./circuits/AI_circuit_2/gallery/allen-key-low-pass-filter_attenuation_factor.png"><br>
This are the parameters used to generate the database<img width=600, height=300, src="./circuits/AI_circuit_2/gallery/database-parameters.png"><br>
This is the ROC of AI-Friendly over the database<img width=600, height=300, src="./circuits/AI_circuit_2/gallery/figPerf.png"><br>
We conclude it has learnt very well</b>

---

<p align="center"><b>CIRCUIT 2: RC-series low-pass-filter</b><br></p>
<b> This is the circuit<img width=100, height=200, src="./circuits/AI_circuit_3/schematic/schematic_dioded.png"><br>
This are the input and output voltage <img width=600, height=300, src="./circuits/AI_circuit_3/gallery/RC-low-pass-filter_example.png"><br>
This are some transmission curves: output voltage over resistance vs freq<img width=600, height=300, src="./circuits/AI_circuit_3/gallery/RC-low-pass-filter_attenuation-factor.png"><br>
This are the parameters used to generate the database<img width=600, height=300, src="./circuits/AI_circuit_3/gallery/database-parameters.png"><br>
This is the ROC of AI-Friendly over the database<img width=600, height=300, src="./circuits/AI_circuit_3/gallery/figPerf.png"><br>
We conclude it has learnt very well</b>

---

<p align="center"><b>CIRCUIT 2: Amplifier</b><br></p>
<b> This is the circuit<img width=100, height=200, src="./circuits/AI_circuit_4/schematic/schematic_dioded.png"><br>
This are the input and output voltage <img width=600, height=300, src="./circuits/AI_circuit_4/gallery/amplifier.png"><br>
This are some amplification vs frequency curves: under the current architecture not any group of parameters lead to a viable amplifier; sometimes the output voltage is lower than the input voltage and in general the coefficient depends on the frequency.<img width=600, height=300, src="./circuits/AI_circuit_4/gallery/amplification-factor_vs_frequency.png"><br>
This are the parameters used to generate the database<img width=600, height=300, src="./circuits/AI_circuit_4/gallery/database-parameters.png"><br>
This is the ROC of AI-Friendly over the database<img width=600, height=300, src="./circuits/AI_circuit_4/gallery/figPerf.png"><br>
We conclude it has learnt very well</b>

