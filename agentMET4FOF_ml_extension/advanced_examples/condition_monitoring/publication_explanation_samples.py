import dill
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

data_dict = dill.load(open("publication/"+"sample_zema3_data.p","rb"))
e_nll = data_dict["mean"][:,:-2]

labels = ['Microphone', 'Vibration plain bearing',
               'Vibration piston rod', 'Vibration ball bearing',
               'Axial force', 'Pressure', 'Velocity', 'Active current',
               'Motor current phase']

label_size = "medium"
title_size = "large"
legend_size = "small"

figsize = (8.0,3.25)
fig , (ax1,ax2) = plt.subplots(1,2, figsize=figsize)

ax1.plot(np.arange(0,len(e_nll.mean(-1))), e_nll.mean(-1))
ax2.plot(e_nll)

ax1.set_xticks([0,len(e_nll)])
ax2.set_xticks([0,len(e_nll)])
ax1.set_xticklabels([0,1.0])
ax2.set_xticklabels([0,1.0])

ax1.set_yticks([])
ax2.set_yticks([])

ax1.set_xlabel("Degradation", fontsize=label_size)
ax2.set_xlabel("Degradation", fontsize=label_size)

ax1.set_ylabel("Health Indicator (HI)", fontsize=label_size)
ax2.set_ylabel("Sensor Attribution Score", fontsize=label_size)

ax1.set_title("(a)",fontsize=title_size)
ax2.set_title("(b)",fontsize=title_size)


plt.legend(labels, fontsize=legend_size)

fig.tight_layout()
# fig.savefig("publication/"+"example-sensor-attribution.png",dpi=500)